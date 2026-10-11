"""CPU regressions for tokenizer support across rollout, cache and training."""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.losses import compute_mirror_opd_token_loss
from mirroropd.training.tokenization import align_rollout_logits


def test_valid_action_normalizer_matches_vllm_mask():
    logits = torch.tensor([[1.2, -0.7, 2.1, 4.0, 3.0]], dtype=torch.float64)
    valid_vocab = 3
    old = logits.softmax(-1)
    assert old[:, valid_vocab:].sum() > 0.8
    masked = logits.clone()
    masked[:, valid_vocab:] = -torch.inf
    aligned = align_rollout_logits(logits, valid_vocab)
    torch.testing.assert_close(aligned.log_softmax(-1), masked.log_softmax(-1)[:, :valid_vocab])
    torch.testing.assert_close(aligned.softmax(-1).sum(-1), torch.ones(1, dtype=torch.float64))
    # A valid sampled action had the wrong denominator before alignment.
    assert not torch.isclose(old[0, 2].log(), aligned.log_softmax(-1)[0, 2])


@pytest.mark.parametrize("size", [0, -1, 6, True, 2.5, None])
def test_invalid_support_rejected(size):
    with pytest.raises(ValueError, match="Tokenizer action support"):
        align_rollout_logits(torch.zeros(2, 5), size)


def test_scalar_logits_rejected():
    with pytest.raises(ValueError, match="Tokenizer action support"):
        align_rollout_logits(torch.tensor(1.0), 1)


@pytest.mark.parametrize("method", ["opd_onpolicy", "mdpo", "bpo"])
def test_padded_logits_do_not_change_cache_loss_or_gradients(method):
    generator = torch.Generator().manual_seed(712)
    anchor = torch.randn(3, 7, generator=generator)
    teacher = torch.randn(3, 7, generator=generator)
    student = torch.randn(3, 7, generator=generator)
    actions = torch.tensor([0, 3, 1])
    valid = torch.ones(3, dtype=torch.bool)

    def evaluate(padded_shift):
        a, t, s = [x.clone() for x in (anchor, teacher, student)]
        a[:, 4:] += padded_shift
        t[:, 4:] -= padded_shift
        s[:, 4:] += 2 * padded_shift
        s.requires_grad_()
        cache = build_mirror_opd_cache_from_full_logits(
            align_rollout_logits(a, 4), align_rollout_logits(t, 4), actions, valid, 2
        )
        kwargs = {key.removeprefix("mirror_"): value for key, value in cache.items()}
        kwargs.pop("valid")
        result = compute_mirror_opd_token_loss(
            align_rollout_logits(s, 4), method=method, eta=0.7, **kwargs
        )
        result["distillation_losses"].sum().backward()
        assert torch.isfinite(s.grad).all()
        assert torch.equal(s.grad[:, 4:], torch.zeros_like(s.grad[:, 4:]))
        assert torch.count_nonzero(s.grad[:, :4]) > 0
        assert torch.isfinite(result["mirror_student_entropy"]).all()
        return cache, result, s.grad

    first, changed = evaluate(0), evaluate(100)
    for original, perturbed in zip(first[:2], changed[:2], strict=True):
        assert original.keys() == perturbed.keys()
        for key in original:
            torch.testing.assert_close(original[key], perturbed[key], rtol=0, atol=0)
    torch.testing.assert_close(first[2], changed[2], rtol=0, atol=0)


def extract_backend_normalization(source_path, remove_padding):
    """Execute actual worker prefix and normalization without Ray/FSDP startup.

    The extraction stops before sampled-logprob kernels and nested-tensor
    output assembly. Both real branch bodies and the shared alignment guard
    are retained, so moving alignment after normalization fails this check.
    """
    tree = ast.parse(Path(source_path).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_model_outputs"
    )
    method = copy.deepcopy(method)
    branch_index = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "use_remove_padding"
    )
    outer = method.body[branch_index]
    branch = outer.body if remove_padding else outer.orelse
    fused = next(
        node
        for node in branch
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "use_fused_kernels"
    )
    before_fused = branch[: branch.index(fused)]
    normalization = []
    for node in fused.orelse:
        normalization.append(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value.func
            if isinstance(call, ast.Attribute) and call.attr == "div_":
                break
    else:
        raise AssertionError("Backend branch has no temperature normalization")
    variable = "logits_rmpad" if remove_padding else "logits"
    method.body = (
        method.body[:branch_index]
        + before_fused
        + normalization
        + [ast.Return(value=ast.Name(id=variable, ctx=ast.Load()))]
    )
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    namespace = {
        "torch": torch,
        "os": os,
        "tu": SimpleNamespace(
            get_non_tensor_data=lambda data, key, default=None: data.get(key, default)
        ),
        "DatasetPadMode": SimpleNamespace(NO_PADDING="no_padding"),
    }
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["prepare_model_outputs"]


@pytest.mark.parametrize("remove_padding", [True, False])
def test_live_backend_aligns_before_both_normalization_branches(monkeypatch, remove_padding):
    source = os.getenv("MIRROR_BACKEND_SOURCE") or str(
        Path(__file__).resolve().parents[1]
        / "upstream/TA-OPD/verl/verl/workers/engine/fsdp/transformer_impl.py"
    )
    if not Path(source).is_file():
        pytest.skip("Restore the upstream backend or set MIRROR_BACKEND_SOURCE")
    fn = extract_backend_normalization(source, remove_padding)
    monkeypatch.setenv("MIRROR_ALIGN_ROLLOUT_VOCAB", "1")
    worker = SimpleNamespace(model_config=SimpleNamespace(tokenizer=range(3)), use_ulysses_sp=False)
    raw = torch.tensor([[[0.2, 0.5, -0.8, 7.0, 8.0], [0.1, 0.8, 1.0, 4.0, 9.0]]])
    args = {
        "input_ids_rmpad_rolled": torch.tensor([1, 2]),
        "temperature_rmpad": torch.tensor([2.0, 2.0]),
        "temperature": torch.tensor([2.0]),
    }
    data = {"input_ids": torch.tensor([[1, 2]]), "use_remove_padding": remove_padding}
    result = fn(worker, SimpleNamespace(logits=raw.clone()), args, data, None)
    expected = raw[..., :3] / 2
    if remove_padding:
        expected = expected.squeeze(0)
    torch.testing.assert_close(result, expected)
    assert result.shape[-1] == 3
    data["use_fused_kernels"] = True
    with pytest.raises((ValueError, NotImplementedError), match=r"[Ff]used"):
        fn(worker, SimpleNamespace(logits=raw.clone()), args, data, None)
