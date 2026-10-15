"""Power rewards apply to fresh-policy score updates as well as mirror reuse."""

from types import SimpleNamespace

import pytest
import torch

from mirroropd.integrations.verl.fsdp import compute_mirror_opd_from_cache
from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.losses import compute_mirror_opd_token_loss


def evaluate(z, q, teacher, method, eta, kappa, topj, center_mode="topj_tail"):
    cache = build_mirror_opd_cache_from_full_logits(
        q.log().expand(3, -1),
        teacher.log().expand(3, -1),
        torch.arange(3),
        torch.ones(3, dtype=torch.bool),
        topj,
        center_mode=center_mode,
        kappa=kappa,
        eta=eta,
    )
    result = compute_mirror_opd_token_loss(
        z.expand(3, -1),
        cache["mirror_actions"],
        cache["mirror_anchor_ids"],
        cache["mirror_anchor_head_log_probs"],
        cache["mirror_anchor_tail_log_prob"],
        cache["mirror_anchor_action_log_prob"],
        cache["mirror_teacher_action_log_prob"],
        cache["mirror_teacher_center"],
        method,
        eta,
        teacher_head_log_probs=cache["mirror_teacher_head_log_probs"],
        teacher_tail_log_prob=cache["mirror_teacher_tail_log_prob"],
        full_anchor_log_probs=cache.get("mirror_full_anchor_log_probs"),
        full_teacher_log_probs=cache.get("mirror_full_teacher_log_probs"),
        kappa=kappa,
        signal_kappa=cache["mirror_signal_kappa"],
        signal_eta=cache["mirror_signal_eta"],
    )
    gradient = torch.autograd.grad((q * result["distillation_losses"]).sum(), z)[0]
    return result, gradient


@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.03, 0.2, 1.0])
@pytest.mark.parametrize(
    ("topj", "center"), [(0, "topj_tail"), (1, "topj_tail"), (3, "topj_tail"), (1, "full_vocab")]
)
def test_fresh_onpolicy_matches_independent_fixed_mirror_target(kappa, eta, topj, center):
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64)
    z = q.log().requires_grad_()
    raw = (
        (1 - eta) * q.log() + eta * teacher.log()
        if kappa == 0
        else ((1 - eta) * q.pow(kappa) + eta * teacher.pow(kappa)).log() / kappa
    )
    target = raw - raw.logsumexp(-1)
    logp = z.log_softmax(-1)
    objective = (logp.exp() * (logp - target)).sum() / eta
    exact = torch.autograd.grad(objective, z)[0]
    opd, actual = evaluate(z, q, teacher, "opd_onpolicy", eta, kappa, topj, center)
    torch.testing.assert_close(actual, exact, atol=2e-14, rtol=2e-12)
    for method in ("bpo", "mdpo"):
        mirror, gradient = evaluate(z, q, teacher, method, eta, kappa, topj, center)
        torch.testing.assert_close(actual, gradient, atol=2e-14, rtol=2e-12)
        torch.testing.assert_close(opd["mirror_sampled_reward"], mirror["mirror_sampled_reward"])
        if method == "bpo":
            torch.testing.assert_close(
                opd["mirror_score_coefficient"], mirror["mirror_score_coefficient"]
            )
    assert not opd["mirror_sampled_reward"].requires_grad
    assert not opd["mirror_score_coefficient"].requires_grad


@pytest.mark.parametrize("topj", [0, 1, 3])
@pytest.mark.parametrize("method", ["opd_is", "opd_is_no_baseline"])
def test_power_is_corrects_changed_policy_and_baseline_only_changes_variance(topj, method):
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64)
    z = torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    p = z.detach().softmax(-1)
    # Independent elementary expression at the current policy, frozen for PG.
    reward = ((1 - 0.2) + 0.2 * (teacher / p).sqrt()).log() / (0.5 * 0.2)
    exact = torch.autograd.grad(-(p * reward * z.log_softmax(-1)).sum(), z)[0]
    result, actual = evaluate(z, q, teacher, method, 0.2, 0.5, topj)
    torch.testing.assert_close(actual, exact, atol=2e-14, rtol=2e-12)
    torch.testing.assert_close(result["mirror_sampled_reward"], reward)
    onpolicy, uncorrected = evaluate(z, q, teacher, "opd_onpolicy", 0.2, 0.5, topj)
    assert not torch.allclose(uncorrected, actual)
    if method == "opd_is":
        torch.testing.assert_close(
            result["mirror_score_coefficient"], onpolicy["mirror_score_coefficient"] * p / q
        )


def test_power_onpolicy_adapter_masks_padding_and_freezes_teacher_cache():
    torch.manual_seed(17)
    q = torch.randn(1, 3, 7, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(q, requires_grad=True)
    valid = torch.tensor([[True, False, True]])
    cache = build_mirror_opd_cache_from_full_logits(
        q, teacher, torch.tensor([[2, -99, 5]]), valid, 3, kappa=0.5, eta=0.2
    )
    saved = {key: value.clone() for key, value in cache.items()}
    config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            loss_mode="mirror_opd_onpolicy",
            mirror_eta=0.2,
            mirror_kappa=0.5,
            mirror_center_mode="topj_tail",
            mirror_action_mode="sampled",
        )
    )
    z = q.detach().clone().requires_grad_()
    outputs = compute_mirror_opd_from_cache(z, cache, config, "bshd")
    outputs["distillation_losses"].sum().backward()
    assert torch.isfinite(z.grad).all()
    assert not z.grad[~valid].any()
    assert q.grad is None
    assert teacher.grad is None
    assert all(torch.equal(cache[key], value) for key, value in saved.items())
    config.distillation_loss.mirror_eta = 0.3
    with pytest.raises(ValueError, match="frozen teacher signal"):
        compute_mirror_opd_from_cache(z, cache, config, "bshd")
