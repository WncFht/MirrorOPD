"""Hidden evidence must preserve sampled BPO centers and all student gradients."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from mirroropd.core.cache import FrozenCache, build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective, Signal
from mirroropd.core.hidden import build_hidden_anchor, sampled_loss, score_hidden_teacher
from mirroropd.core.objectives import compute_loss


@pytest.mark.parametrize("diagnostics", ["none", "basic", "full"])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
@pytest.mark.parametrize("signal", [Signal(), Signal(0, 0.2), Signal(0.5, 1)])
def test_hidden_centers_and_sampled_gradients_match_eager(diagnostics, temperature, signal):
    torch.manual_seed(74)
    n, v, width = 5, 19, 7
    hq, wq, bq = (
        torch.randn(n, width, dtype=torch.double),
        torch.randn(v, width, dtype=torch.double),
        torch.randn(v, dtype=torch.double),
    )
    ht, wt, bt = (
        torch.randn(n, 11, dtype=torch.double),
        torch.randn(v, 11, dtype=torch.double),
        torch.randn(v, dtype=torch.double),
    )
    h, w, b = [
        x.requires_grad_()
        for x in (torch.randn_like(hq), torch.randn_like(wq), torch.randn_like(bq))
    ]
    actions, valid = torch.tensor([0, 3, 18, 5, 8]), torch.ones(n, dtype=torch.bool)
    spec = Objective("bpo", signal, Estimator("full", "sampled", 4), diagnostics)
    context = CacheContext(3, "teacher", "vocab", v, temperature, signal, spec.estimator)
    eager = build_cache(
        F.linear(hq, wq, bq) / temperature,
        F.linear(ht, wt, bt) / temperature,
        actions,
        valid,
        context,
        diagnostics=diagnostics,
    )
    compact_context = replace(context, anchor_storage="hidden", anchor_head_path="test-window-3.pt")
    anchor = build_hidden_anchor(hq, wq, bq, actions, compact_context, vocab_chunk=6)
    cache = score_hidden_teacher(ht, wt, bt, anchor, wq, bq, diagnostics=diagnostics, vocab_chunk=6)
    torch.testing.assert_close(cache.reward_center, eager.reward_center, atol=2e-13, rtol=2e-12)
    assert cache.anchor_full is None
    assert cache.anchor_hidden.shape == (n, width)
    assert all(not t.requires_grad for t in cache.tensors().values())
    actual = sampled_loss(h, w, b, cache, spec, wq, bq, token_chunk=2, vocab_chunk=6)
    expected = compute_loss(F.linear(h, w, b) / temperature, eager, spec, anchor_version=3)
    torch.testing.assert_close(actual.loss, expected.loss, atol=2e-11, rtol=2e-11)
    assert actual.diagnostics.keys() == expected.diagnostics.keys()
    for key in actual.diagnostics:
        torch.testing.assert_close(
            actual.diagnostics[key], expected.diagnostics[key], atol=2e-11, rtol=2e-11
        )
    upstream = torch.tensor([0.0, 2.0, -0.5, 1.0, -3.0], dtype=torch.double)
    gradients = torch.autograd.grad(actual.loss, (h, w, b), upstream)
    reference = torch.autograd.grad(expected.loss, (h, w, b), upstream)
    for a, e in zip(gradients, reference, strict=True):
        torch.testing.assert_close(a, e, atol=3e-11, rtol=3e-11)
    restored = FrozenCache.from_wire(cache.to_wire(), compact_context)
    assert torch.equal(restored.anchor_hidden, hq)


def test_hidden_cache_rejects_missing_or_misaligned_evidence():
    spec = Objective(estimator=Estimator("full", "sampled", 2))
    context = CacheContext(
        0,
        "t",
        "v",
        7,
        1.0,
        spec.signal,
        spec.estimator,
        anchor_storage="hidden",
        anchor_head_path="head.pt",
    )
    anchor = build_hidden_anchor(
        torch.randn(3, 4), torch.randn(7, 4), None, torch.tensor([0, 1, 6]), context, vocab_chunk=3
    )
    with pytest.raises(ValueError, match="normalizers"):
        replace(anchor, anchor_logz=None)
    with pytest.raises(ValueError, match="align"):
        replace(anchor, anchor_hidden=torch.randn(2, 4))
    with pytest.raises(ValueError, match="Stale"):
        context.validate(spec, anchor_version=1)


@pytest.mark.parametrize("low_precision", [False, True])
def test_fp32_and_bf16_projection_error_is_bounded(low_precision):
    import runpy
    from pathlib import Path

    module = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "benchmarks/hidden_anchor.py")
    )
    errors = module["precision_errors"](low_precision)
    tolerance = 2e-2 if low_precision else 5e-5
    assert errors["loss_max_abs"] < tolerance
    assert errors["teacher_center_max_abs"] < tolerance
    assert errors["student_kl_max_abs"] < tolerance
    assert max(errors["gradient_relative_l2"]) < tolerance


@pytest.mark.parametrize("topj", [0, 7])
def test_tail_and_masked_global_token_reduction(topj):
    from mirroropd.core.cache import RESPONSE_FIELDS, scatter_valid

    torch.manual_seed(12)
    valid = torch.tensor([[False, True, True, False], [False, True, False, False]])
    actions = torch.tensor([[0, 0, 6, 0], [0, 4, 0, 0]])
    spec = Objective(estimator=Estimator("full", "sampled", topj), diagnostics="full")
    context = CacheContext(0, "t", "v", 7, 1.0, spec.signal, spec.estimator)
    compact = replace(context, anchor_storage="hidden", anchor_head_path="test.pt")
    hq, ht = torch.randn(3, 4, dtype=torch.double), torch.randn(3, 4, dtype=torch.double)
    wq, wt = torch.randn(7, 4, dtype=torch.double), torch.randn(7, 4, dtype=torch.double)
    anchor = build_hidden_anchor(hq, wq, None, actions[valid], compact, vocab_chunk=3)
    packed = score_hidden_teacher(ht, wt, None, anchor, wq, None, diagnostics="full", vocab_chunk=3)
    cache = replace(
        packed,
        **{
            key: value if key in RESPONSE_FIELDS else scatter_valid(value, valid, value.shape[1:])
            for key, value in packed.tensors().items()
        },
    )
    cache = replace(cache, valid=valid)
    h = torch.randn(2, 4, 4, dtype=torch.double)
    h[~valid] = float("nan")
    h.requires_grad_()
    w = torch.randn(7, 4, dtype=torch.double, requires_grad=True)
    qa, ta = torch.zeros(2, 4, 7, dtype=torch.double), torch.zeros(2, 4, 7, dtype=torch.double)
    qa[valid], ta[valid] = F.linear(hq, wq), F.linear(ht, wt)
    direct = build_cache(qa, ta, actions, valid, context)
    actual = sampled_loss(h, w, None, cache, spec, wq, None, token_chunk=2, vocab_chunk=3)
    expected = compute_loss(F.linear(h.nan_to_num(), w), direct, spec, anchor_version=0)
    ga = torch.autograd.grad(actual.loss.sum() / valid.sum(), (h, w), retain_graph=True)
    ge = torch.autograd.grad(expected.loss.sum() / valid.sum(), (h, w))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-11)
    assert not ga[0][~valid].any()
