"""Narrow verl binding; framework imports occur only inside native callbacks."""

import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache, wraps
from pathlib import Path

from mirroropd.core.cache import ANCHOR_FIELDS, TEACHER_FIELDS, WIRE_PREFIX
from mirroropd.core.config import API_VERSION, CacheContext, Objective, decode
from mirroropd.identity import canonical_json
from mirroropd.integrations.verl.adapter import (
    DIAGNOSTIC_PREFIX,
    cache_outputs,
    loss_outputs,
    prediction_fields,
    restore_cache_output,
)
from mirroropd.runtime.events import environment_writer, phase
from mirroropd.runtime.memory import Memory
from mirroropd.runtime.metrics import finalize_moments, record_update, token_moments


@dataclass(frozen=True, slots=True)
class Settings:
    teacher_model_path: str
    teacher_id: str
    tokenizer_id: str
    vocab_size: int
    objective: Objective
    memory: Memory = field(default_factory=Memory)
    schema_version: int = API_VERSION

    def __post_init__(self):
        if self.schema_version != API_VERSION or not self.teacher_model_path:
            raise ValueError("Backend requires the current mirror schema and explicit teacher")


@lru_cache(maxsize=32)
def _settings(serialized):
    return decode(Settings, json.loads(serialized))


def settings(config):
    from omegaconf import OmegaConf

    value = config.mirror
    value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return _settings(canonical_json(value))


def is_enabled(config):
    return config is not None and config.enabled and config.distillation_loss.loss_mode == "mirror"


def window_context(config, anchor_version, temperature):
    value = settings(config)
    return CacheContext(
        anchor_version,
        value.teacher_id,
        value.tokenizer_id,
        value.vocab_size,
        temperature,
        value.objective.signal,
        value.objective.estimator,
        anchor_storage=value.memory.anchor_storage,
        anchor_head_path=(
            str(
                Path(os.environ["MIRROR_RUN_DIR"])
                / "cache"
                / "anchor-heads"
                / f"{anchor_version}.pt"
            )
            if value.memory.anchor_storage == "hidden"
            else None
        ),
        action_only=(
            value.objective.algorithm == "mdpo"
            and value.objective.estimator.integration == "sampled"
            and value.objective.diagnostics == "none"
        ),
    )


def metadata(data, key):
    from verl.utils import tensordict_utils as tu

    value = tu.get_non_tensor_data(data, key=key, default=None)
    if value is None:
        raise ValueError(f"Native mirror batch is missing {key}")
    return value


def context_from_batch(data, objective):
    context = CacheContext.from_dict(metadata(data, "mirror_context"))
    context.validate(objective, anchor_version=metadata(data, "mirror_anchor_version"))
    if metadata(data, "temperature") != context.temperature:
        raise ValueError("Probability temperature does not match the frozen cache context")
    return context


def cache_callback(logits, data, stage, *, sequence_parallel_size):
    objective = Objective.from_dict(metadata(data, "mirror_objective"))
    context = context_from_batch(data, objective)
    return cache_outputs(
        logits,
        data,
        stage,
        context,
        diagnostics=objective.diagnostics,
        sequence_parallel_size=sequence_parallel_size,
    )


def loss_callback(logits, data, config, data_format, *, sequence_parallel_size):
    value = settings(config)
    context = context_from_batch(data, value.objective)
    if (context.teacher_id, context.tokenizer_id, context.vocab_size) != (
        value.teacher_id,
        value.tokenizer_id,
        value.vocab_size,
    ):
        raise ValueError("Batch teacher/vocabulary identity differs from the configured models")
    return loss_outputs(
        logits,
        data,
        value.objective,
        context,
        anchor_version=metadata(data, "mirror_anchor_version"),
        data_format=data_format,
        sequence_parallel_size=sequence_parallel_size,
    )


def prepare_global_batch(config, data):
    if metadata(data, "batch_num_tokens") <= 0:
        raise ValueError("Distillation batch has no valid response actions")
    for key in ("dp_size", "batch_num_tokens", "global_batch_size"):
        config.global_batch_info[key] = data[key]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor


def aggregate_loss(model_output, data):
    from verl.utils.metric import Metric
    from verl.workers.utils.padding import no_padding_2_padding

    losses = no_padding_2_padding(model_output["distillation_losses"], data)
    mask = data["response_mask"]
    mask = mask.bool().to_padded_tensor(False) if mask.is_nested else mask.bool()
    fields = {
        key.removeprefix(DIAGNOSTIC_PREFIX): no_padding_2_padding(value, data)
        for key, value in model_output.items()
        if key.startswith(DIAGNOSTIC_PREFIX)
    }
    return losses, {
        key: Metric(aggregation=reduction, value=value)
        for key, (reduction, value) in token_moments(fields, mask).items()
    }


def finish_update(metrics, data, *, inner, epochs, dp_size, global_rank, clip_grad):
    objective = Objective.from_dict(metadata(data, "mirror_objective"))
    context = context_from_batch(data, objective)
    if objective.algorithm == "on_policy" and epochs != 1:
        raise ValueError("On-policy cannot reuse a frozen rollout for multiple updates")
    output = finalize_moments(metrics, dp_size)
    record_update(
        output,
        outer=context.anchor_version,
        inner=inner,
        epochs=epochs,
        algorithm=objective.algorithm,
        loss_kind=objective.loss_kind,
        clip_grad=clip_grad,
        global_rank=global_rank,
    )
    return output


def validate_rollout_versions(non_tensor_batch, expected_version, batch_size):
    versions = non_tensor_batch.get("global_steps")
    if (
        versions is None
        or len(versions) != batch_size
        or any(value != expected_version for value in versions)
    ):
        raise ValueError(
            "Rollout server did not report the expected student weight version for every trajectory"
        )
    return {"distill/rollout_weight_version": expected_version}


def anchor_pass(trainer, batch):
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    context = window_context(
        trainer.distillation_config,
        trainer.global_steps - 1,
        trainer.config.actor_rollout_ref.rollout.temperature,
    )
    if context.anchor_storage == "hidden":
        directory = Path(context.anchor_head_path).parent
        directory.mkdir(parents=True, exist_ok=True)
        if Path(context.anchor_head_path).exists():
            raise FileExistsError("Anchor head already exists; equivalent resume is unsupported")
        # No old-window consumer remains after the blocking train update.
        for old in directory.glob("*.pt"):
            old.unlink()
    batch.meta_info.update(
        mirror_context=context.to_dict(),
        mirror_anchor_version=context.anchor_version,
        mirror_objective=settings(trainer.distillation_config).objective.to_dict(),
        mirror_memory=asdict(settings(trainer.distillation_config).memory),
    )
    data = batch.to_tensordict()
    for key, value in prediction_fields(
        data["input_ids"], data["attention_mask"], data["response_mask"]
    ).items():
        data[key] = value
    data = left_right_2_no_padding(data)
    tu.assign_non_tensor(data, mirror_stage="anchor", calculate_entropy=False, compute_loss=False)
    with phase("cache.anchor", outer=context.anchor_version):
        output = trainer.actor_rollout_wg.compute_log_prob(data)
    values = {"old_log_probs": no_padding_2_padding(tu.get(output, "log_probs"), data).float()}
    indices = tu.get_non_tensor_data(data, key="indices", default=None)
    for name in ANCHOR_FIELDS:
        key = WIRE_PREFIX + name
        if tu.get(output, key) is not None:
            values[key] = restore_cache_output(
                key, tu.get(output, key), data["attention_mask"].shape, indices
            )
    writer = environment_writer("driver")
    if writer:
        writer.emit(
            "anchor.created",
            outer=context.anchor_version,
            cache_context=context.to_dict(),
            valid_tokens=int(values[WIRE_PREFIX + "valid"].sum().item()),
            bytes=sum(
                value.numel() * value.element_size()
                for key, value in values.items()
                if key.startswith(WIRE_PREFIX)
            ),
        )
    return DataProto.from_tensordict(tu.get_tensordict(values)), tu.get(output, "metrics")["mfu"]


def teacher_pass(trainer, batch):
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding

    data = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(data, mirror_stage="teacher", compute_loss=False, calculate_entropy=False)
    with phase("cache.teacher", outer=trainer.global_steps - 1):
        output = trainer.ref_policy_wg.compute_ref_log_prob(data)
    indices = tu.get_non_tensor_data(data, key="indices", default=None)
    values = {}
    for name in TEACHER_FIELDS:
        key = WIRE_PREFIX + name
        if tu.get(output, key) is not None:
            values[key] = restore_cache_output(
                key, tu.get(output, key), data["attention_mask"].shape, indices
            )
    return DataProto.from_tensordict(tu.get_tensordict(values))


def batch_metrics(batch):
    import torch

    response = batch.batch["response_mask"].bool()
    counts = response.sum(-1)
    prompt = batch.batch["attention_mask"][:, : -response.shape[1]].sum(-1)
    result = {
        "training/prompt_tokens_mean": prompt.float().mean().item(),
        "training/response_tokens_mean": counts.float().mean().item(),
        "training/response_tokens_max": counts.max().item(),
        "training/response_tokens": counts.sum().item(),
        "training/generated_responses": len(counts),
    }
    references = torch.tensor(
        [bool(value.get("has_reference", True)) for value in batch.non_tensor_batch["extra_info"]],
        device=counts.device,
    )
    result["task/labeled_responses"] = references.sum().item()
    if references.any():
        scores = batch.batch["token_level_scores"].sum(-1)
        result["task/score_mean"] = scores[references].float().mean().item()
    return result


def worker_memory_phase(name):
    """Observe the whole native worker operation; nested projections never reset peaks."""

    def decorate(function):
        @wraps(function)
        def wrapped(self, data, *args, **kwargs):
            import torch
            from verl.utils import tensordict_utils as tu

            context = tu.get_non_tensor_data(data, key="mirror_context", default=None)
            if context is None:
                return function(self, data, *args, **kwargs)
            memory = Memory(**tu.get_non_tensor_data(data, key="mirror_memory", default={}))
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            stage = tu.get_non_tensor_data(data, key="mirror_stage", default=None)
            label = name + ("." + stage if stage else "")
            with phase(
                label,
                source="worker-" + str(rank),
                profile_memory=memory.profile_phases,
                outer=context["anchor_version"],
            ):
                return function(self, data, *args, **kwargs)

        return wrapped

    return decorate
