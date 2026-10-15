"""Independent execution parity and bounded saved-state contracts."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from mirroropd.core.cache import FrozenCache, build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective
from mirroropd.core.distributions import Distribution
from mirroropd.core.kernels import projected_loss
from mirroropd.core.objectives import compute_loss


@pytest.mark.parametrize("algorithm", ["on_policy", "mdpo", "bpo"])
@pytest.mark.parametrize(
    ("center", "integration"), [("topj_tail", "sampled"), ("full", "sampled"), ("full", "full")]
)
@pytest.mark.parametrize("diagnostics", ["none", "basic", "full"])
@pytest.mark.parametrize("chunk", [1, 3])
def test_projected_value_diagnostics_and_model_gradients(
    algorithm, center, integration, diagnostics, chunk
):
    torch.manual_seed(220)
    h = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    w = torch.randn(19, 7, dtype=torch.float64, requires_grad=True)
    b = torch.randn(19, dtype=torch.float64, requires_grad=True)
    q = torch.randn(2, 5, 19, dtype=torch.float64)
    teacher = torch.randn_like(q)
    valid = torch.tensor([[False, True, True, True, False], [False, False, True, True, True]])
    actions = torch.randint(19, (2, 5))
    spec = Objective(
        algorithm, estimator=Estimator(center, integration, 4), diagnostics=diagnostics
    )
    context = CacheContext(4, "teacher", "tokenizer", 19, 0.7, spec.signal, spec.estimator)
    frozen = build_cache(q, teacher, actions, valid, context, diagnostics=diagnostics)
    reference = compute_loss(F.linear(h, w, b) / 0.7, frozen, spec, anchor_version=4)
    expected = torch.autograd.grad(reference.loss.sum(), (h, w, b), retain_graph=True)
    result = projected_loss(
        h, w, b, frozen, spec, anchor_version=4, token_chunk=chunk, vocab_chunk=5, temperature=0.7
    )
    torch.testing.assert_close(result.loss, reference.loss, atol=2e-12, rtol=3e-11)
    actual = torch.autograd.grad(result.loss.sum(), (h, w, b))
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=2e-12, rtol=3e-11)
    assert result.kind == reference.kind
    for name in reference.diagnostics:
        torch.testing.assert_close(
            result.diagnostics[name], reference.diagnostics[name], atol=2e-12, rtol=3e-11
        )
    assert not actual[0][~valid].any()


def test_projected_backward_does_not_save_logits_or_softmax():
    torch.manual_seed(221)
    h = torch.randn(13, 7, requires_grad=True)
    w = torch.randn(101, 7, requires_grad=True)
    spec = Objective(diagnostics="none")
    context = CacheContext(0, "t", "v", 101, 1.0, spec.signal, spec.estimator)
    frozen = build_cache(
        torch.randn(13, 101),
        torch.randn(13, 101),
        torch.randint(101, (13,)),
        torch.ones(13, dtype=torch.bool),
        context,
        diagnostics="none",
    )
    saved = []

    def pack(x):
        saved.append((tuple(x.shape), x.requires_grad))
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        result = projected_loss(h, w, None, frozen, spec, anchor_version=0, token_chunk=3)
        result.loss.sum().backward()
    assert not any(grad and shape in {(13, 101), (3, 101)} for shape, grad in saved)


def test_full_bpo_minimal_path_needs_no_student_normalization(monkeypatch):
    spec = Objective("bpo", estimator=Estimator("full", "full", 2), diagnostics="none")
    ctx = CacheContext(0, "t", "v", 7, 1.0, spec.signal, spec.estimator)
    frozen = build_cache(
        torch.randn(3, 7),
        torch.randn(3, 7),
        torch.arange(3),
        torch.ones(3, dtype=torch.bool),
        ctx,
        diagnostics="none",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Full BPO must not compute student softmax/normalizer")

    monkeypatch.setattr(Distribution, "from_logits", forbidden)
    z = torch.randn(3, 7, requires_grad=True)
    compute_loss(z, frozen, spec, anchor_version=0, vocab_chunk_size=2).loss.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_action_only_mdpo_cache_has_no_unused_distribution_fields(monkeypatch):
    spec = Objective("mdpo", estimator=Estimator("full", "sampled", 2), diagnostics="none")
    ctx = CacheContext(0, "t", "v", 7, 1.0, spec.signal, spec.estimator, action_only=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Action-only scoring cannot compute head/tail/full statistics")

    monkeypatch.setattr(Distribution, "tail", forbidden)
    monkeypatch.setattr(Distribution, "full", forbidden)
    frozen = build_cache(
        torch.randn(3, 7),
        torch.randn(3, 7),
        torch.arange(3),
        torch.ones(3, dtype=torch.bool),
        ctx,
        diagnostics="none",
    )
    assert set(frozen.tensors()) == {"valid", "actions", "anchor_action", "teacher_action"}
    restored = FrozenCache.from_wire(frozen.to_wire(), ctx)
    z = torch.randn(3, 7, requires_grad=True)
    compute_loss(z, restored, spec, anchor_version=0).loss.sum().backward()
    with pytest.raises(ValueError, match="Action-only"):
        compute_loss(z, restored, replace(spec, diagnostics="basic"), anchor_version=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_selected_logprobs_backward_matches_dense_normalization(dtype):
    torch.manual_seed(231)
    logits = torch.randn(5, 4103, dtype=dtype, requires_grad=True)
    ids = torch.randint(4103, (5, 3))
    weights = torch.randn(5, 3, dtype=torch.float64 if dtype == torch.float64 else torch.float32)
    dist = Distribution.from_logits(logits)
    selected = dist.selected(ids)
    reference = logits.to(selected.dtype).log_softmax(-1).gather(-1, ids)
    expected = torch.autograd.grad((reference * weights).sum(), logits, retain_graph=True)[0]
    actual = torch.autograd.grad((selected * weights).sum(), logits)[0]
    torch.testing.assert_close(
        selected,
        reference,
        atol=2e-6 if dtype != torch.float64 else 2e-13,
        rtol=2e-6 if dtype != torch.float64 else 2e-13,
    )
    torch.testing.assert_close(
        actual,
        expected,
        atol=4e-3 if dtype == torch.bfloat16 else 2e-6,
        rtol=2e-2 if dtype == torch.bfloat16 else 2e-5,
    )


@pytest.mark.parametrize("shift", [0.0, 10000.0, -10000.0])
def test_tiled_full_bpo_matches_independent_softmax_objective(shift):
    from mirroropd.core.kernels import full_bpo

    torch.manual_seed(232)
    q = torch.randn(4, 37, dtype=torch.float64).log_softmax(-1)
    target = torch.randn_like(q).log_softmax(-1)
    z = (torch.randn_like(q) + shift).requires_grad_()
    residual = z.log_softmax(-1) - target
    residual = residual - (q.exp() * residual).sum(-1, keepdim=True)
    oracle = (q.exp() * residual.square()).sum(-1) / 0.4
    actual = full_bpo(z, q, target, 0.2, token_chunk=2, vocab_chunk=7)
    g1 = torch.autograd.grad(oracle.sum(), z, retain_graph=True)[0]
    g2 = torch.autograd.grad(actual.sum(), z)[0]
    torch.testing.assert_close(actual, oracle, atol=2e-11, rtol=3e-12)
    torch.testing.assert_close(g2, g1, atol=2e-12, rtol=3e-11)


def test_full_bpo_projection_under_bfloat16_autocast():
    torch.manual_seed(229)
    h = torch.randn(11, 32, requires_grad=True)
    w = torch.randn(131, 32, requires_grad=True) / 32**0.5
    spec = Objective("bpo", estimator=Estimator("full", "full", 8), diagnostics="none")
    context = CacheContext(0, "t", "v", 131, 0.7, spec.signal, spec.estimator)
    frozen = build_cache(
        torch.randn(11, 131),
        torch.randn(11, 131),
        torch.randint(131, (11,)),
        torch.ones(11, dtype=torch.bool),
        context,
        diagnostics="none",
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reference = compute_loss(F.linear(h, w) / 0.7, frozen, spec, anchor_version=0)
        current = projected_loss(
            h,
            w,
            None,
            frozen,
            spec,
            anchor_version=0,
            token_chunk=3,
            vocab_chunk=17,
            temperature=0.7,
        )
    expected = torch.autograd.grad(reference.loss.sum(), (h, w))
    actual = torch.autograd.grad(current.loss.sum(), (h, w))
    torch.testing.assert_close(current.loss, reference.loss, atol=2e-4, rtol=1e-3)
    for left, right in zip(actual, expected, strict=True):
        assert (left - right).norm() / right.norm() < 0.01
        # BF16 GEMM and tile accumulation differ most near zero. Compare
        # vector error and the largest coordinate against gradient scale.
        assert (left - right).abs().max() / right.abs().max() < 0.01
