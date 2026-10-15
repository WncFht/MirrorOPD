"""Independent objective, gradient, cache and padding checks for power mirrors."""

from types import SimpleNamespace

import pytest
import torch

from mirroropd.integrations.verl.fsdp import compute_mirror_opd_from_cache
from mirroropd.proximal import normalized_power_signal
from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.losses import compute_mirror_opd_token_loss, full_vocab_mirror_loss
from mirroropd.training.support import topk_pair_overlap


def reference(logits, qlog, tlog, method, eta, kappa):
    """Construct the normalized power target independently of production signals."""
    if kappa == 0:
        raw = (1 - eta) * qlog + eta * tlog
    else:
        raw = ((1 - eta) * qlog.exp().pow(kappa) + eta * tlog.exp().pow(kappa)).log() / kappa
    mlog = raw - raw.logsumexp(-1, keepdim=True)
    logp = logits.log_softmax(-1)
    if method == "mdpo":
        return (logp.exp() * (logp - mlog)).sum(-1) / eta
    h = logp - mlog
    residual = h - (qlog.exp() * h).sum(-1, keepdim=True)
    return (qlog.exp() * residual.square()).sum(-1) / (2 * eta)


def fields(logits, cache, method, eta, kappa, action_mode="sampled"):
    return compute_mirror_opd_token_loss(
        logits,
        cache["mirror_actions"],
        cache["mirror_anchor_ids"],
        cache["mirror_anchor_head_log_probs"],
        cache["mirror_anchor_tail_log_prob"],
        cache["mirror_anchor_action_log_prob"],
        cache["mirror_teacher_action_log_prob"],
        cache["mirror_teacher_center"],
        method,
        eta,
        full_anchor_log_probs=cache.get("mirror_full_anchor_log_probs"),
        full_teacher_log_probs=cache.get("mirror_full_teacher_log_probs"),
        kappa=kappa,
        action_mode=action_mode,
    )


@pytest.mark.parametrize("method", ["bpo", "mdpo"])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.03, 0.3, 1.0])
def test_complete_objective_and_enumerated_score_match_independent_target(method, kappa, eta):
    qlog = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64).log()
    tlog = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64).log()
    z = torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    expected = reference(z, qlog, tlog, method, eta, kappa)
    gradient = torch.autograd.grad(expected, z)[0]
    actual = full_vocab_mirror_loss(z, qlog, tlog, method, eta, kappa)
    torch.testing.assert_close(torch.autograd.grad(actual, z)[0], gradient, atol=2e-13, rtol=2e-12)
    if method == "bpo":
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)
    actions = torch.arange(3)
    for center_mode, topj in (("full_vocab", 1), ("topj_tail", 3)):
        cache = build_mirror_opd_cache_from_full_logits(
            qlog.expand(3, -1),
            tlog.expand(3, -1),
            actions,
            torch.ones(3, dtype=torch.bool),
            topj,
            center_mode=center_mode,
            kappa=kappa,
            eta=eta,
        )
        sampled = fields(z.expand(3, -1), cache, method, eta, kappa)
        enumerated = (qlog.exp() * sampled["distillation_losses"]).sum()
        torch.testing.assert_close(
            torch.autograd.grad(enumerated, z)[0], gradient, atol=2e-13, rtol=2e-12
        )
        assert not sampled["mirror_score_coefficient"].requires_grad


@pytest.mark.parametrize("topj", [0, 1, 2, 3])
def test_grouped_bpo_bias_and_mdpo_center_independence(topj):
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    t = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64)
    z = torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    inputs = (
        q.log().expand(3, -1),
        t.log().expand(3, -1),
        torch.arange(3),
        torch.ones(3, dtype=torch.bool),
    )
    compact = build_mirror_opd_cache_from_full_logits(*inputs, topj, kappa=0.5, eta=0.3)
    complete = build_mirror_opd_cache_from_full_logits(
        *inputs, topj, center_mode="full_vocab", kappa=0.5, eta=0.3
    )
    assert (compact["mirror_teacher_center"] <= complete["mirror_teacher_center"] + 1e-13).all()
    for method in ("bpo", "mdpo"):
        grouped = fields(z.expand(3, -1), compact, method, 0.3, 0.5)
        full = fields(z.expand(3, -1), complete, method, 0.3, 0.5)
        difference = (q * (grouped["distillation_losses"] - full["distillation_losses"])).sum()
        actual = torch.autograd.grad(difference, z, retain_graph=True)[0]
        if method == "bpo":
            delta = (
                grouped["mirror_student_center"][0] - full["mirror_student_center"][0]
            ) / 0.3 - (compact["mirror_teacher_center"][0] - complete["mirror_teacher_center"][0])
            expected = delta * (q - z.detach().softmax(-1))
        else:
            expected = torch.zeros_like(z)
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=2e-12)


def test_signal_small_shape_small_strength_and_extreme_ratios_are_stable():
    r = torch.tensor([-10000.0, -0.1, 0.0, 0.1, 10000.0], dtype=torch.float64)
    for kappa in (1e-12, 0.5, 1.0):
        for eta in (1e-12, 0.3, 1.0):
            assert torch.isfinite(normalized_power_signal(r, kappa, eta)).all()
    torch.testing.assert_close(normalized_power_signal(r, 1e-12, 0.3), r, rtol=1e-8, atol=1e-8)
    for kappa in (0.0, 0.5, 1.0):
        torch.testing.assert_close(normalized_power_signal(r, kappa, 1.0), r)
    with pytest.raises(ValueError, match="eta at most one"):
        normalized_power_signal(r, 0.5, 1.1)


def test_large_head_overlap_matches_set_intersection_without_quadratic_storage():
    left = torch.arange(151665).reshape(1, -1)
    right = left.flip(-1)
    output = topk_pair_overlap(left, right, "full")
    assert output["mirror_overlap_full_intersection_count"].item() == 151665
    assert output["mirror_overlap_full_jaccard"].item() == 1
    left = torch.arange(256).reshape(1, -1)
    right = left + 123
    output = topk_pair_overlap(left, right, "partial")
    assert output["mirror_overlap_partial_intersection_count"].item() == 133


def test_complete_mdpo_does_not_compute_unneeded_overflowing_importance_ratios():
    qlog = torch.tensor([[-1000.0, 0.0]], dtype=torch.float64)
    tlog = torch.zeros_like(qlog).log_softmax(-1)
    cache = build_mirror_opd_cache_from_full_logits(
        qlog,
        tlog,
        torch.tensor([0]),
        torch.tensor([True]),
        1,
        center_mode="full_vocab",
        kappa=0.5,
        eta=0.3,
    )
    z = torch.zeros_like(qlog, requires_grad=True)
    output = fields(z, cache, "mdpo", 0.3, 0.5, "full_vocab")
    assert "mirror_importance_ratio" not in output
    assert all(torch.isfinite(value).all() for value in output.values())
    output["distillation_losses"].sum().backward()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize(
    ("center_mode", "action_mode"),
    [("topj_tail", "sampled"), ("full_vocab", "sampled"), ("full_vocab", "full_vocab")],
)
def test_padding_cache_freezing_student_refresh_and_signal_contract(center_mode, action_mode):
    torch.manual_seed(11)
    q = torch.randn(1, 3, 7, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(q, requires_grad=True)
    valid = torch.tensor([[True, False, True]])
    cache = build_mirror_opd_cache_from_full_logits(
        q,
        t,
        torch.tensor([[2, -99, 5]]),
        valid,
        3,
        center_mode=center_mode,
        kappa=0.5,
        eta=0.3,
    )
    frozen = {key: value.clone() for key, value in cache.items()}
    settings = SimpleNamespace(
        loss_mode="mirror_bpo",
        mirror_eta=0.3,
        mirror_kappa=0.5,
        mirror_center_mode=center_mode,
        mirror_action_mode=action_mode,
    )
    config = SimpleNamespace(distillation_loss=settings)
    z = q.detach().clone().requires_grad_()
    output = compute_mirror_opd_from_cache(z, cache, config, "bshd")
    output["distillation_losses"].sum().backward()
    assert q.grad is None
    assert t.grad is None
    assert not z.grad[~valid].any()
    assert all(torch.equal(cache[key], value) for key, value in frozen.items())
    perturbed = z.detach().clone()
    perturbed[..., 0] += 0.7
    updated = compute_mirror_opd_from_cache(perturbed, cache, config, "bshd")
    assert (updated["mirror_student_center"][valid] > output["mirror_student_center"][valid]).all()
    torch.testing.assert_close(updated["mirror_teacher_center"], output["mirror_teacher_center"])
    settings.mirror_eta = 0.7
    with pytest.raises(ValueError, match="frozen teacher signal"):
        compute_mirror_opd_from_cache(z, cache, config, "bshd")
    settings.mirror_eta = 0.3
    settings.mirror_center_mode = "full_vocab" if center_mode == "topj_tail" else "topj_tail"
    with pytest.raises(ValueError, match="frozen anchor cache"):
        compute_mirror_opd_from_cache(z, cache, config, "bshd")
