"""Mathematical target identities, unbiased score gradients and fitting certificates."""

import math

import pytest
import torch

from mirroropd.proximal import (
    minimax_velocity_shape,
    power_log_target,
    renyi_inner_bound,
    sampled_target_loss,
    target_divergence,
    worst_fisher_speed_squared,
)


@pytest.mark.parametrize("kappa", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("fit", ["reverse_kl", "power", "bpo"])
@pytest.mark.parametrize("strength", [0.2, 1.0])
def test_anchor_enumerated_score_matches_true_fitting_gradient(kappa, fit, strength):
    qlog = torch.tensor([[0.2, 0.3, 0.5]]).log()
    tlog = torch.tensor([[0.6, 0.35, 0.05]]).log()
    target = power_log_target(qlog, tlog, kappa, strength)
    logits = torch.tensor([[-0.7, 0.2, 0.8]], requires_grad=True)
    true = (
        target_divergence(logits.log_softmax(-1), target, kappa, fit, anchor_logp=qlog).sum()
        / strength
    )
    expected = torch.autograd.grad(true, logits)[0]
    fields = sampled_target_loss(
        logits, qlog, target, torch.tensor([[0, 1, 2]]), kappa, strength, fit
    )
    # Enumerating categorical outcomes requires q weights, not an unweighted average.
    surrogate = (qlog.exp() * fields["coefficient"] * logits.log_softmax(-1)).sum()
    actual = torch.autograd.grad(surrogate, logits)[0]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_shape_endpoints_square_root_and_native_mdpo_scale():
    qlog = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64).log()
    tlog = torch.tensor([[0.6, 0.35, 0.05]], dtype=torch.float64).log()
    arithmetic = power_log_target(qlog, tlog, 1, 0.2).exp()
    torch.testing.assert_close(arithmetic, 0.8 * qlog.exp() + 0.2 * tlog.exp())
    sqrt_target = (0.8 * qlog.exp().sqrt() + 0.2 * tlog.exp().sqrt()).square()
    sqrt_target /= sqrt_target.sum(-1, keepdim=True)
    torch.testing.assert_close(power_log_target(qlog, tlog, 0.5, 0.2).exp(), sqrt_target)
    geometric = power_log_target(qlog, tlog, 0, 0.2)
    logp = torch.tensor([[0.25, 0.25, 0.5]], dtype=torch.float64).log()
    original = (logp.exp() * ((logp - qlog) / 0.2 - (tlog - qlog))).sum(-1)
    normalized = target_divergence(logp, geometric, 0) / 0.2
    logz = torch.logsumexp(0.8 * qlog + 0.2 * tlog, -1)
    torch.testing.assert_close(normalized, original + logz / 0.2)


@pytest.mark.parametrize("seed", range(4))
def test_renyi_certificate_upper_bounds_actual_teacher_reverse_kl(seed):
    torch.manual_seed(seed)
    qlog = torch.randn(5, 11, dtype=torch.float64).log_softmax(-1)
    tlog = torch.randn(5, 11, dtype=torch.float64).log_softmax(-1)
    plog = torch.randn(5, 11, dtype=torch.float64).log_softmax(-1)
    for kappa in (0.0, 0.5, 1.0):
        mlog = power_log_target(qlog, tlog, kappa, 0.2)
        bound = renyi_inner_bound(plog, mlog, tlog)
        actual = target_divergence(plog, tlog, 0)
        assert torch.all(bound >= actual - 1e-10)


def test_power_target_stays_normalized_for_extreme_teacher_ratios():
    qlog = torch.tensor([[-math.log(2), -math.log(2)]], dtype=torch.float64)
    tlog = torch.tensor([[-1000.0, 0.0]], dtype=torch.float64).log_softmax(-1)
    for kappa in (0.0, 0.5, 1.0):
        mlog = power_log_target(qlog, tlog, kappa, 0.1)
        assert torch.isfinite(mlog).all()
        torch.testing.assert_close(mlog.logsumexp(-1), torch.zeros(1, dtype=torch.float64))


@pytest.mark.parametrize("kappa", [0.25, 0.5])
def test_current_student_sampled_power_gradient_has_error_scaled_second_moment(kappa):
    torch.manual_seed(42)
    logits = torch.randn(1, 23, requires_grad=True)
    target = torch.randn(1, 23).log_softmax(-1)
    proposal = logits.detach().log_softmax(-1)
    fields = sampled_target_loss(
        logits, proposal, target, torch.arange(23).unsqueeze(0), kappa, 0.1, "power"
    )
    loss = target_divergence(logits.log_softmax(-1), target, kappa, "power")
    coefficient_moment = (proposal.exp() * (0.1 * fields["coefficient"]).square()).sum(-1)
    assert coefficient_moment <= loss / kappa + 1e-5
    enumerated = (proposal.exp() * fields["coefficient"] * logits.log_softmax(-1)).sum()
    direct = loss.sum() / 0.1
    torch.testing.assert_close(
        torch.autograd.grad(enumerated, logits, retain_graph=True)[0],
        torch.autograd.grad(direct, logits)[0],
        atol=3e-6,
        rtol=3e-6,
    )


def test_sharp_velocity_optimum_is_an_interior_minimum():
    optimum = minimax_velocity_shape()
    assert 0.43 < optimum < 0.46
    value = worst_fisher_speed_squared(optimum)
    assert value < worst_fisher_speed_squared(0.5)
    assert value < worst_fisher_speed_squared(optimum - 0.01)
    assert value < worst_fisher_speed_squared(optimum + 0.01)


@pytest.mark.parametrize("kappa", [0.1, 0.25, 0.49, 0.5])
def test_error_relative_coefficient_constant_has_a_sharp_boundary_witness(kappa):
    p = torch.tensor([[1e-8, 1 - 1e-8]], dtype=torch.float64)
    m = torch.tensor([[1e-16, 1 - 1e-16]], dtype=torch.float64)
    rho = m / p
    a = (p * rho.pow(kappa)).sum(-1)
    moment = (p * (rho.pow(kappa) - a.unsqueeze(-1)).square()).sum(-1) / kappa**2
    loss = target_divergence(p.log(), m.log(), kappa, "power")
    assert float(moment / loss) <= 1 / kappa + 1e-5
    assert float(moment / loss) >= 0.7 / kappa


@pytest.mark.parametrize("kappa", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.03, 0.3, 1.0])
def test_anchor_fisher_bpo_steps_are_mirror_steps_with_finite_teacher_progress(kappa, eta):
    q = torch.tensor([0.35, 0.25, 0.18, 0.12, 0.07, 0.03], dtype=torch.float64)
    teacher = torch.tensor([0.04, 0.1, 0.13, 0.2, 0.25, 0.28], dtype=torch.float64)
    mlog = power_log_target(q.log(), teacher.log(), kappa, eta)
    fisher = torch.diag(q) - q[:, None] * q[None, :]
    inverse = torch.linalg.pinv(fisher)
    tau = 0.4
    z = q.log().requires_grad_()

    def metrics(logp):
        p = logp.exp()
        return torch.stack(
            (
                target_divergence(logp, teacher.log(), 0),
                target_divergence(teacher.log(), logp, 0),
                (p - teacher).abs().sum() / 2,
                (p.sqrt() - teacher.sqrt()).square().sum(),
            )
        )

    previous = metrics(z.log_softmax(-1))
    for _ in range(3):
        logp = z.log_softmax(-1)
        loss = target_divergence(logp, mlog, kappa, "bpo", anchor_logp=q.log()) / eta
        gradient = torch.autograd.grad(loss, z)[0]
        next_z = (z - tau * eta * (inverse @ gradient)).detach().requires_grad_()
        next_logp = next_z.log_softmax(-1)
        expected = ((1 - tau) * logp + tau * mlog).log_softmax(-1)
        torch.testing.assert_close(next_logp, expected, atol=1e-12, rtol=1e-12)
        current = metrics(next_logp)
        assert torch.all(current <= previous + 1e-12)
        next_loss = target_divergence(next_logp, mlog, kappa, "bpo", anchor_logp=q.log()) / eta
        torch.testing.assert_close(next_loss, (1 - tau) ** 2 * loss, atol=1e-12, rtol=1e-12)
        previous, z = current.detach(), next_z
