"""Native metadata, jagged transport, and the real causal-LM output gradient edge."""

import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mirroropd.core.cache import build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective
from mirroropd.core.objectives import compute_loss
from mirroropd.integrations.verl.adapter import nested_cache_output
from mirroropd.runtime.memory import Memory


@pytest.fixture(scope="module")
def native():
    import sys

    path = Path(os.environ.get("MIRROR_TEST_BACKEND", "upstream/TA-OPD-v2")) / "verl"
    sys.path.insert(0, str(path.resolve()))
    from verl.utils import tensordict_utils as tu

    return tu


def batch(native, spec, context, fields, stage=None):
    offsets = torch.tensor([0, 6])
    data = {}
    valid = fields["mirror_cache_valid"]
    for key, value in fields.items():
        data[key] = nested_cache_output(
            key,
            value.reshape(-1, *value.shape[2:])
            if key not in {"mirror_cache_anchor_full", "mirror_cache_target_full"}
            and value.ndim >= 2
            else value,
            valid,
            offsets,
        )
    data["input_ids"] = torch.nested.nested_tensor_from_jagged(
        torch.tensor([2, 4, 5, 6, 7, 8]), offsets
    )
    return native.get_tensordict(
        tensor_dict=data,
        non_tensor_dict={
            "mirror_objective": spec.to_dict(),
            "mirror_context": context.to_dict(),
            "mirror_anchor_version": 0,
            "temperature": context.temperature,
            "mirror_stage": stage,
            "mirror_memory": asdict(Memory(token_chunk_size=2, vocab_chunk_size=5)),
        },
    )


@pytest.mark.parametrize("algorithm", ["on_policy", "mdpo", "bpo"])
@pytest.mark.parametrize("integration", ["sampled", "full"])
def test_real_causal_lm_projected_forward_returns_the_actual_loss_edge(
    native, algorithm, integration
):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from mirroropd.integrations.verl.projection import project_head

    torch.manual_seed(444)
    model = (
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                attention_dropout=0.0,
            )
        )
        .double()
        .eval()
    )
    spec = Objective(algorithm, estimator=Estimator("full", integration, 4), diagnostics="basic")
    context = CacheContext(0, "t", "v", 19, 1.0, spec.signal, spec.estimator)
    valid = torch.tensor([[False, False, True, True, True, False]])
    actions = torch.tensor([[0, 0, 6, 7, 8, 0]])
    frozen = build_cache(
        torch.randn(1, 6, 19, dtype=torch.float64),
        torch.randn(1, 6, 19, dtype=torch.float64),
        actions,
        valid,
        context,
        diagnostics="basic",
    )
    data = batch(native, spec, context, frozen.to_wire())
    ids = torch.tensor([[2, 4, 5, 6, 7, 8]])
    original = model.get_output_embeddings().forward
    reference = compute_loss(model(ids).logits[..., :19], frozen, spec, anchor_version=0)
    expected = torch.autograd.grad(reference.loss.sum(), tuple(model.parameters()))
    with project_head(model, data) as projected:
        output = model(ids)
        result = projected.outputs(output.logits, data["input_ids"])
    assert model.get_output_embeddings().forward == original
    actual_loss = result["distillation_losses"].values().sum()
    torch.testing.assert_close(actual_loss, reference.loss.sum(), atol=2e-12, rtol=3e-11)
    actual = torch.autograd.grad(actual_loss, tuple(model.parameters()))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=4e-11)
    assert not actual[-1][19:].any()


def test_full_fields_remain_jagged_through_dataproto_reorder_and_chunks(native):
    from verl import DataProto

    valid = torch.tensor([[False, True, True, False], [False, False, True, False]])
    offsets = torch.tensor([0, 4, 8])
    packed = torch.arange(21.0).reshape(3, 7)
    full = nested_cache_output("mirror_cache_anchor_full", packed, valid, offsets)
    assert full.values().numel() == 3 * 7
    td = native.get_tensordict({"mirror_cache_anchor_full": full, "label": torch.tensor([5, 9])})
    proto = DataProto.from_tensordict(td)
    changed = proto[[1, 0]]
    assert torch.equal(changed.batch["mirror_cache_anchor_full"].unbind()[0], packed[2:])
    assert torch.equal(changed.chunk(2)[1].batch["mirror_cache_anchor_full"].values(), packed[:2])


def test_head_forward_is_restored_on_scoring_failure(native):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from mirroropd.integrations.verl.projection import project_head

    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    spec = Objective(estimator=Estimator("full", "full", 4))
    context = CacheContext(0, "t", "v", 19, 1.0, spec.signal, spec.estimator)
    fields = {
        "mirror_cache_valid": torch.ones(1, 6, dtype=torch.bool),
        "mirror_cache_actions": torch.zeros(1, 6, dtype=torch.long),
    }
    data = batch(native, spec, context, fields, "anchor")
    native.assign_non_tensor(data, mirror_memory=asdict(Memory(cache_bytes=1)))
    original = model.get_output_embeddings().forward
    with pytest.raises(MemoryError), torch.no_grad(), project_head(model, data):
        model(torch.tensor([[2, 4, 5, 6, 7, 8]]))
    assert model.get_output_embeddings().forward == original


@pytest.mark.parametrize("algorithm", ["on_policy", "mdpo", "bpo"])
@pytest.mark.parametrize("integration", ["sampled", "full"])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_projected_anchor_teacher_loss_pipeline(native, algorithm, integration, temperature):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from mirroropd.integrations.verl.projection import project_head

    torch.manual_seed(446)

    def model(width):
        return (
            Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=width,
                    intermediate_size=width * 2,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    attention_dropout=0.0,
                )
            )
            .double()
            .eval()
        )

    student, teacher = model(16), model(24)
    spec = Objective(algorithm, estimator=Estimator("full", integration, 4), diagnostics="full")
    context = CacheContext(0, "t", "v", 19, temperature, spec.signal, spec.estimator)
    valid = torch.tensor([[False, False, True, True, True, False]])
    actions = torch.tensor([[0, 0, 6, 7, 8, 0]])
    ids = torch.tensor([[2, 4, 5, 6, 7, 8]])
    direct = build_cache(
        student(ids).logits[..., :19] / temperature,
        teacher(ids).logits[..., :19] / temperature,
        actions,
        valid,
        context,
    )
    fields = {"mirror_cache_valid": valid, "mirror_cache_actions": actions}
    anchor = batch(native, spec, context, fields, "anchor")
    with torch.no_grad(), project_head(student, anchor) as operation:
        output = student(ids)
        raw = operation.outputs(output.logits, anchor["input_ids"])
    for key, value in raw.items():
        if key != "log_probs":
            anchor[key] = value
    native.assign_non_tensor(anchor, mirror_stage="teacher")
    with torch.no_grad(), project_head(teacher, anchor) as operation:
        output = teacher(ids)
        raw = operation.outputs(output.logits, anchor["input_ids"])
    for key, value in raw.items():
        if key != "log_probs":
            anchor[key] = value
    native.assign_non_tensor(anchor, mirror_stage=None)
    from mirroropd.integrations.verl.adapter import cache_from_data

    transported = cache_from_data(anchor, context)
    for key, value in direct.tensors().items():
        actual = transported.tensors()[key]
        torch.testing.assert_close(actual.reshape(value.shape), value, atol=2e-12, rtol=3e-11)
    with project_head(student, anchor) as operation:
        output = student(ids)
        result = operation.outputs(output.logits, anchor["input_ids"])
    reference = compute_loss(
        student(ids).logits[..., :19] / temperature, direct, spec, anchor_version=0
    )
    torch.testing.assert_close(
        result["distillation_losses"].values(), reference.loss.reshape(-1), atol=3e-12, rtol=4e-11
    )
