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


@pytest.mark.parametrize("eta", [0.03, 0.3, 0.7])
def test_teacher_center_is_bounded_monotone_and_has_certified_tail_bounds(eta):
    from mirroropd.proximal import power_signal

    q = torch.tensor([0.35, 0.25, 0.18, 0.12, 0.07, 0.03], dtype=torch.float64)
    teacher = torch.tensor([0.04, 0.1, 0.13, 0.2, 0.25, 0.28], dtype=torch.float64)
    rho = teacher / q
    previous = (q * -rho.log()).sum()
    delta, tail_teacher = q[2:].sum(), teacher[2:].sum()
    for kappa in (0.25, 0.5, 0.75, 1):
        a = power_signal(rho.log(), kappa, eta) / eta
        center = -(q * a).sum()
        cap = -math.log1p(-eta) / (kappa * eta)
        assert 0 <= center <= previous + 1e-12
        assert center <= cap + 1e-12
        phi = rho - 1 - a
        assert torch.all(phi >= -1e-12)
        torch.testing.assert_close((q * phi).sum(), center, atol=1e-12, rtol=1e-12)
        tail_a = power_signal((tail_teacher / delta).log(), kappa, eta) / eta
        lower = -(q[:2] * a[:2]).sum() - delta * tail_a
        upper = -(q[:2] * a[:2]).sum() + delta * cap
        assert lower <= center + 1e-12
        assert center <= upper + 1e-12
        if kappa <= 0.5:
            affinity = (q * rho.pow(kappa)).sum()
            variance = (q * (a + center).square()).sum()
            assert variance <= (1 - affinity.square()) / (kappa**2 * (1 - eta) ** 2) + 1e-12
            assert variance <= 2 * center / (kappa * (1 - eta) ** 2) + 1e-12
        previous = center


def test_square_root_teacher_center_and_variance_have_hellinger_limits():
    from mirroropd.proximal import power_signal

    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    teacher = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    h2 = (q.sqrt() - teacher.sqrt()).square().sum()
    eta = 1e-6
    a = power_signal((teacher / q).log(), 0.5, eta) / eta
    center = -(q * a).sum()
    variance = (q * (a + center).square()).sum()
    torch.testing.assert_close(center, h2, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(variance, h2 * (4 - h2), atol=2e-5, rtol=2e-5)


def test_nonnegative_center_estimator_can_have_diverging_variance():
    from mirroropd.proximal import power_signal

    moments = []
    teacher = torch.tensor([0.5, 0.5], dtype=torch.float64)
    for epsilon in (1e-3, 1e-8):
        q = torch.tensor([epsilon, 1 - epsilon], dtype=torch.float64)
        rho = teacher / q
        a = power_signal(rho.log(), 0.5, 0.3) / 0.3
        center = -(q * a).sum()
        phi = rho - 1 - a
        signed_variance = (q * (a + center).square()).sum()
        positive_variance = (q * (phi - center).square()).sum()
        torch.testing.assert_close((q * phi).sum(), center, atol=1e-10, rtol=1e-10)
        moments.append((signed_variance, positive_variance))
    assert moments[1][0] < moments[0][0]
    assert moments[1][1] > 10000 * moments[0][1]


@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_same_action_positive_center_estimate_cancels_teacher_shape(kappa):
    from mirroropd.proximal import power_signal

    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    teacher = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    rho = teacher / q
    a = power_signal(rho.log(), kappa, 0.3) / 0.3
    positive_estimate = rho - 1 - a
    torch.testing.assert_close(a + positive_estimate, rho - 1, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kappa", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.03, 0.3, 0.7])
def test_envelope_center_estimator_is_nonnegative_unbiased_and_bounded(kappa, eta):
    from mirroropd.proximal import power_signal

    q = torch.tensor([1e-9, 0.7, 0.3 - 1e-9], dtype=torch.float64)
    teacher = torch.tensor([0.6, 0.1, 0.3], dtype=torch.float64)
    rho = teacher / q
    a = power_signal(rho.log(), kappa, eta) / eta
    center = -(q * a).sum()
    phi = rho - 1 - a
    cap = -math.log1p(-eta) / (kappa * eta)
    proposal = (1 - 1 / cap) * q + teacher / cap
    z = q * phi / proposal
    assert torch.all(z >= -1e-12)
    assert torch.all(z <= cap + 1e-12)
    torch.testing.assert_close((proposal * z).sum(), center, atol=1e-12, rtol=1e-12)
    variance = (proposal * (z - center).square()).sum()
    assert variance <= center * (cap - center) + 1e-12


def test_small_step_hellinger_mixture_estimator_halves_signed_variance_bound():
    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    teacher = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    proposal = (q + teacher) / 2
    z = 2 * (q.sqrt() - teacher.sqrt()).square() / (q + teacher)
    center = (q.sqrt() - teacher.sqrt()).square().sum()
    torch.testing.assert_close((proposal * z).sum(), center, atol=1e-12, rtol=1e-12)
    variance = (proposal * (z - center).square()).sum()
    signed_variance = center * (4 - center)
    assert variance <= center * (2 - center) + 1e-12
    assert variance <= signed_variance / 2 + 1e-12


def test_teacher_center_error_only_biases_uncorrected_score_form():
    from mirroropd.proximal import power_signal

    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    teacher = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    eta = 0.3
    f = power_signal((teacher / q).log(), 0.5, eta)
    center = -(q * f).sum() / eta
    delta = 0.2
    z = torch.tensor([0.1, -0.4, 0.9], dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    u = lp - q.log()
    residual = u - (q * u).sum() - f - eta * (center + delta)
    loss = (q * residual.square()).sum() / (2 * eta)
    actual = torch.autograd.grad(loss, z, retain_graph=True)[0]
    target = power_log_target(q.log(), teacher.log(), 0.5, eta)
    exact_loss = target_divergence(lp, target, 0.5, "bpo", anchor_logp=q.log()) / eta
    exact = torch.autograd.grad(exact_loss, z, retain_graph=True)[0]
    torch.testing.assert_close(actual, exact, atol=1e-12, rtol=1e-12)
    raw = (q * residual.detach() * lp).sum() / eta
    raw_gradient = torch.autograd.grad(raw, z, retain_graph=True)[0]
    torch.testing.assert_close(
        raw_gradient - exact, -delta * (q - lp.exp()), atol=1e-12, rtol=1e-12
    )
    centered_score = (q * residual.detach() * (lp - (q * lp).sum())).sum() / eta
    corrected = torch.autograd.grad(centered_score, z)[0]
    torch.testing.assert_close(corrected, exact, atol=1e-12, rtol=1e-12)


def test_small_kl_does_not_bound_geometric_teacher_advantage_variance():
    from mirroropd.proximal import power_signal

    results = []
    for epsilon in (1e-3, 1e-6):
        q = torch.tensor([epsilon, 1 - epsilon], dtype=torch.float64)
        logt = torch.tensor([-(epsilon ** (-0.75)), 0.0], dtype=torch.float64).log_softmax(-1)
        r = logt - q.log()
        kl = -(q * r).sum()
        raw_variance = (q * (r + kl).square()).sum()
        a = power_signal(r, 0.5, 0.3) / 0.3
        center = -(q * a).sum()
        shaped_variance = (q * (a + center).square()).sum()
        results.append((kl, raw_variance, shaped_variance))
    assert results[1][0] < results[0][0]
    assert results[1][1] > 10 * results[0][1]
    assert results[1][2] < results[0][2]


@pytest.mark.parametrize("sample_count", [2, 3, 4])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_sample_variance_bpo_has_exact_population_loss_and_gradient(sample_count, kappa):
    from itertools import product

    from mirroropd.proximal import sample_variance_bpo_loss

    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64)
    target = power_log_target(q.log(), teacher.log(), kappa, 0.3)
    z = torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    exact = target_divergence(z.log_softmax(-1), target, kappa, "bpo", anchor_logp=q.log()) / 0.3
    empirical_mean = 0.0
    for outcome in product(range(3), repeat=sample_count):
        actions = torch.tensor(outcome)
        empirical_mean += (
            q[actions].prod() * sample_variance_bpo_loss(z, target, actions, 0.3)["loss"]
        )
    torch.testing.assert_close(empirical_mean, exact, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(empirical_mean, z)[0],
        torch.autograd.grad(exact, z)[0],
        atol=1e-12,
        rtol=1e-12,
    )
    fields = sample_variance_bpo_loss(z, target, torch.tensor([0, 1]), 0.3)
    shifted = sample_variance_bpo_loss(z + 123.0, target - 37.0, torch.tensor([0, 1]), 0.3)
    torch.testing.assert_close(fields["loss"], shifted["loss"], atol=1e-11, rtol=1e-11)
    at_target = target.clone().requires_grad_()
    zero = sample_variance_bpo_loss(at_target, target, torch.tensor([0, 1, 2]), 0.3)["loss"]
    assert float(zero.detach()) == 0
    torch.testing.assert_close(torch.autograd.grad(zero, at_target)[0], torch.zeros_like(at_target))


@pytest.mark.parametrize("sample_count", [2, 4])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
def test_mixture_weighted_bpo_preserves_population_target(sample_count, kappa):
    from itertools import product

    from mirroropd.proximal import sample_variance_bpo_loss

    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64)
    proposal = (q + teacher) / 2
    target = power_log_target(q.log(), teacher.log(), kappa, 0.3)
    z = torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    exact = target_divergence(z.log_softmax(-1), target, kappa, "bpo", anchor_logp=q.log()) / 0.3
    expected = 0.0
    for outcome in product(range(3), repeat=sample_count):
        actions = torch.tensor(outcome)
        weights = q[actions] / proposal[actions]
        assert torch.all(weights <= 2)
        loss = sample_variance_bpo_loss(z, target, actions, 0.3, importance_weights=weights)["loss"]
        expected += proposal[actions].prod() * loss
    torch.testing.assert_close(expected, exact, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(expected, z)[0],
        torch.autograd.grad(exact, z)[0],
        atol=1e-12,
        rtol=1e-12,
    )


def test_distinct_pair_bpo_is_unbiased_and_reduces_parameter_covariance():
    from mirroropd.proximal import sample_variance_bpo_loss

    q = torch.tensor([0.8, 0.15, 0.05], dtype=torch.float64)
    teacher = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    target = power_log_target(q.log(), teacher.log(), 0.5, 0.3)
    z = torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    exact = target_divergence(z.log_softmax(-1), target, 0, "bpo", anchor_logp=q.log()) / 0.3
    exact_gradient = torch.autograd.grad(exact, z, retain_graph=True)[0]
    expected = 0.0
    iid_moment = torch.zeros((), dtype=torch.float64)
    distinct_moment = torch.zeros_like(iid_moment)
    for a in range(3):
        for b in range(3):
            actions = torch.tensor([a, b])
            base = sample_variance_bpo_loss(z, target, actions, 0.3)["loss"]
            base_gradient = torch.autograd.grad(base, z, retain_graph=True)[0]
            iid_moment += q[a] * q[b] * base_gradient.square().sum()
            if a != b:
                probability = q[a] * q[b] / (1 - q[a])
                weights = torch.stack((torch.ones_like(q[a]), 1 - q[a]))
                loss = sample_variance_bpo_loss(
                    z, target, actions, 0.3, importance_weights=weights
                )["loss"]
                expected += probability * loss
                gradient = torch.autograd.grad(loss, z, retain_graph=True)[0]
                distinct_moment += probability * gradient.square().sum()
    torch.testing.assert_close(expected, exact, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.autograd.grad(expected, z)[0], exact_gradient, atol=1e-12, rtol=1e-12
    )
    assert distinct_moment <= iid_moment + 1e-12
