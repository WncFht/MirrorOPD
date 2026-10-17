"""Run Mirror's projection and loss inside the native FSDP model forward.

The output loss remains on the model's returned logits edge so FSDP backward
hooks run. The actual LM-head module is restored on every exit. Teacher/anchor
payloads leave the device one scoring tile at a time; dense evidence is jagged
in response coordinates, independently of input sequence offsets.
"""

from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

from mirroropd.core.cache import (
    ANCHOR_FIELDS,
    TEACHER_FIELDS,
    WIRE_PREFIX,
    build_anchor,
    score_teacher,
    select_valid,
)
from mirroropd.core.config import CacheContext, Objective
from mirroropd.core.hidden import build_hidden_anchor, sampled_loss, score_hidden_teacher
from mirroropd.core.kernels import projected_loss
from mirroropd.integrations.verl.adapter import (
    DIAGNOSTIC_PREFIX,
    cache_from_data,
    nested_cache_output,
)
from mirroropd.runtime.events import environment_writer, phase
from mirroropd.runtime.frozen_head import load_head, publish_head
from mirroropd.runtime.memory import Memory, check_cache_capacity, scoring_tile_size, tensor_bytes


class Projection:
    def __init__(self, data):
        from verl.utils import tensordict_utils as tu

        self.data = data
        self.stage = tu.get_non_tensor_data(data, key="mirror_stage", default=None)
        spec = tu.get_non_tensor_data(data, key="mirror_objective", default=None)
        self.enabled = spec is not None
        self.result = None
        if not self.enabled:
            return
        self.objective = Objective.from_dict(spec)
        self.memory = Memory(**tu.get_non_tensor_data(data, key="mirror_memory", default={}))
        self.enabled = self.memory.projection == "chunked"
        self.context = CacheContext.from_dict(
            tu.get_non_tensor_data(data, key="mirror_context", default=None)
        )
        if self.memory.anchor_storage != self.context.anchor_storage:
            raise ValueError("Memory storage differs from the frozen cache context")
        self.context.validate(
            self.objective,
            anchor_version=tu.get_non_tensor_data(data, key="mirror_anchor_version", default=None),
        )
        if (
            tu.get_non_tensor_data(data, key="temperature", default=None)
            != self.context.temperature
        ):
            raise ValueError("Projection temperature differs from the frozen cache")

    def forward(self, hidden, weight, bias):
        valid = self.data[WIRE_PREFIX + "valid"]
        valid = valid.values() if valid.is_nested else valid.reshape(-1)
        if hidden.numel() // hidden.shape[-1] != valid.numel():
            raise ValueError("Projected hidden states do not align with prediction positions")
        weight = weight[: self.context.vocab_size]
        bias = bias[: self.context.vocab_size] if bias is not None else None
        source = "worker-" + str(
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        observation = (
            phase(
                "projection." + (self.stage or "actor"),
                source=source,
                profile_memory=False,
                outer=self.context.anchor_version,
            )
            if self.memory.profile_phases
            else nullcontext()
        )
        with observation:
            if self.stage is None:
                frozen = cache_from_data(self.data, self.context)
                if self.context.anchor_storage == "hidden":
                    frozen_weight, frozen_bias = load_head(self.context, hidden.device)
                    result = sampled_loss(
                        hidden,
                        weight,
                        bias,
                        frozen,
                        self.objective,
                        frozen_weight,
                        frozen_bias,
                        token_chunk=self.memory.token_chunk_size,
                        vocab_chunk=self.memory.vocab_chunk_size,
                    )
                else:
                    result = projected_loss(
                        hidden,
                        weight,
                        bias,
                        frozen,
                        self.objective,
                        anchor_version=self.context.anchor_version,
                        token_chunk=self.memory.token_chunk_size,
                        vocab_chunk=self.memory.vocab_chunk_size,
                        temperature=self.context.temperature,
                    )
                self.result = {DIAGNOSTIC_PREFIX + k: v for k, v in result.diagnostics.items()}
                # This exact edge must be returned by the wrapped causal LM.
                return result.loss.reshape(*hidden.shape[:-1], 1)
            self.result = self.score(hidden, weight, bias, valid)
            writer = environment_writer(source)
            if writer is not None and self.memory.profile_phases:
                writer.emit(
                    "cache.scored",
                    outer=self.context.anchor_version,
                    stage=self.stage,
                    valid_tokens=int(valid.sum()),
                    host_storage_bytes=tensor_bytes(self.result.values()),
                )
            return hidden.new_zeros((*hidden.shape[:-1], 1))

    def score(self, hidden, weight, bias, valid):
        actions = self.data[WIRE_PREFIX + "actions"]
        actions = actions.values() if actions.is_nested else actions.reshape(-1)
        count = int(valid.sum())
        check_cache_capacity(
            self.objective,
            self.memory,
            prefixes=count,
            vocab_size=self.context.vocab_size,
            sequence_slots=valid.numel(),
            hidden_size=hidden.shape[-1]
            if self.stage == "anchor"
            else (
                cache_from_data(self.data, self.context, anchor_only=True).anchor_hidden.shape[-1]
                if self.context.anchor_storage == "hidden"
                else None
            ),
            bytes_per_hidden=2
            if torch.is_autocast_enabled(hidden.device.type)
            and torch.get_autocast_dtype(hidden.device.type) == torch.bfloat16
            else hidden.element_size(),
        )
        packed = hidden.reshape(-1, hidden.shape[-1])[valid]
        anchor = None
        if self.stage == "teacher":
            anchor = cache_from_data(self.data, self.context, anchor_only=True)
        frozen_weight = frozen_bias = None
        if self.context.anchor_storage == "hidden":
            if self.stage == "anchor":
                publish_head(self.context, weight, bias, hidden)
            else:
                frozen_weight, frozen_bias = load_head(self.context, hidden.device)
        chosen_actions = actions[valid]
        records = {}
        # Bound *output staging* independently from projection workspace.
        tile = scoring_tile_size(
            self.objective,
            self.memory,
            vocab_size=self.context.vocab_size,
            hidden_size=hidden.shape[-1],
            bytes_per_hidden=hidden.element_size(),
        )
        for start in range(0, count, tile):
            stop = min(start + tile, count)
            logits = None
            if self.context.anchor_storage == "hidden":
                if self.stage == "anchor":
                    result = build_hidden_anchor(
                        packed[start:stop],
                        weight,
                        bias,
                        chosen_actions[start:stop],
                        self.context,
                        vocab_chunk=self.memory.vocab_chunk_size,
                    )
                    names = ANCHOR_FIELDS
                else:
                    result = score_hidden_teacher(
                        packed[start:stop],
                        weight,
                        bias,
                        select_valid(anchor, start, stop),
                        frozen_weight,
                        frozen_bias,
                        diagnostics=self.objective.diagnostics,
                        vocab_chunk=self.memory.vocab_chunk_size,
                    )
                    names = TEACHER_FIELDS
            elif self.stage == "anchor":
                logits = F.linear(packed[start:stop], weight, bias) / self.context.temperature
                result = build_anchor(
                    logits,
                    chosen_actions[start:stop],
                    torch.ones(stop - start, device=valid.device, dtype=torch.bool),
                    self.context,
                )
                names = ANCHOR_FIELDS
            elif self.stage == "teacher":
                logits = F.linear(packed[start:stop], weight, bias) / self.context.temperature
                result = score_teacher(
                    logits,
                    select_valid(anchor, start, stop),
                    diagnostics=self.objective.diagnostics,
                )
                names = TEACHER_FIELDS
            else:
                raise ValueError("Unknown projection scoring stage")
            for name, value in result.tensors().items():
                if name in names:
                    records.setdefault(name, []).append(value.detach().cpu())
            del logits, result
        # Empty responses are permitted for an individual dispatch row.
        if count == 0:
            logits = packed.new_empty((0, self.context.vocab_size))
            if self.context.anchor_storage == "hidden":
                if self.stage == "anchor":
                    empty = build_hidden_anchor(
                        packed,
                        weight,
                        bias,
                        actions[valid],
                        self.context,
                        vocab_chunk=self.memory.vocab_chunk_size,
                    )
                    names = ANCHOR_FIELDS
                else:
                    empty = score_hidden_teacher(
                        packed,
                        weight,
                        bias,
                        select_valid(anchor, 0, 0),
                        frozen_weight,
                        frozen_bias,
                        diagnostics=self.objective.diagnostics,
                        vocab_chunk=self.memory.vocab_chunk_size,
                    )
                    names = TEACHER_FIELDS
            elif self.stage == "anchor":
                empty = build_anchor(logits, actions[valid], valid[valid], self.context)
                names = ANCHOR_FIELDS
            else:
                empty = score_teacher(
                    logits, select_valid(anchor, 0, 0), diagnostics=self.objective.diagnostics
                )
                names = TEACHER_FIELDS
            records = {
                name: [value.cpu()] for name, value in empty.tensors().items() if name in names
            }
        from mirroropd.core.cache import RESPONSE_FIELDS

        output = {}
        cpu_valid = valid.cpu()
        for name, parts in records.items():
            value = torch.cat(parts)
            if name not in RESPONSE_FIELDS:
                dense = value.new_zeros((len(valid), *value.shape[1:]))
                dense[cpu_valid] = value
                value = dense
            output[WIRE_PREFIX + name] = value
        name = "anchor_action" if self.stage == "anchor" else "teacher_action"
        output["log_probs"] = output[WIRE_PREFIX + name]
        return output

    def outputs(self, logits, input_ids):
        if self.stage is None:
            values = logits.squeeze(0).squeeze(-1)
            outputs = {
                "distillation_losses": torch.nested.nested_tensor_from_jagged(
                    values, input_ids.offsets()
                )
            }
            for key, value in self.result.items():
                outputs[key] = torch.nested.nested_tensor_from_jagged(
                    value.reshape(-1), input_ids.offsets()
                )
            return outputs
        offsets = input_ids.offsets()
        valid = self.data[WIRE_PREFIX + "valid"]
        return {
            key: nested_cache_output(key, value, valid, offsets)
            for key, value in self.result.items()
        }


@contextmanager
def project_head(model, data, *, sequence_parallel_size=1):
    projected = Projection(data)
    if not projected.enabled:
        yield projected
        return
    from verl.utils import tensordict_utils as tu

    if sequence_parallel_size != 1:
        raise NotImplementedError("Chunked MirrorOPD requires sequence parallel size 1")
    if not tu.get_non_tensor_data(data, key="use_remove_padding", default=True):
        raise NotImplementedError("Chunked MirrorOPD requires remove-padding")
    if tu.get_non_tensor_data(data, key="use_fused_kernels", default=False):
        raise NotImplementedError("Chunked MirrorOPD requires a standard unfused linear head")
    head = model.get_output_embeddings()
    config = model.config
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if not isinstance(head, torch.nn.Linear) or config.model_type not in {
        "qwen2",
        "qwen3",
        "llama",
        "mistral",
    }:
        raise NotImplementedError(
            "Chunked projection supports standard Qwen2/Qwen3/Llama/Mistral linear heads; select memory.projection=eager for another output contract"
        )
    if getattr(head, "mirror_projection_active", False):
        raise RuntimeError("The LM head is already owned by another projection")
    original = head.forward
    head.mirror_projection_active = True

    def forward(hidden):
        return projected.forward(hidden, head.weight, head.bias)

    head.forward = forward
    try:
        yield projected
    finally:
        head.forward = original
        del head.mirror_projection_active
