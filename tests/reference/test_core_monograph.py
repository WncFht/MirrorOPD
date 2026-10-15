"""Independent finite-simplex audits for the core monograph's new derivations."""

import itertools
import math

import numpy as np
import pytest
import torch

from mirroropd.reference.categorical import mean_target
from mirroropd.reference.targets import normalized_power_signal, power_log_target

DTYPE = torch.float64


def kl(p, q):
    return (p * (p.log() - q.log())).sum()


def power_divergence(p, q, kappa):
    if kappa == 0:
        return kl(q, p)
    if kappa == 1:
        return kl(p, q)
    return -(q * torch.expm1(kappa * (p.log() - q.log()))).sum() / (kappa * (1 - kappa))


def teacher_objective_by_quadrature(p, teacher, kappa, eta):
    """Integrate the manuscript's generator independently of its closed gradient."""
    nodes, weights = np.polynomial.legendre.leggauss(96)
    nodes = torch.tensor(nodes, dtype=DTYPE)
    weights = torch.tensor(weights, dtype=DTYPE)
    x = teacher / p
    u = 1 + (x[:, None] - 1) * (nodes + 1) / 2
    if kappa == 0:
        signal = u.log()
    else:
        signal = torch.log1p(eta * torch.expm1(kappa * u.log())) / (kappa * eta)
    integral = (weights * signal / u.square()).sum(-1) * (x - 1) / 2
    return (p * x * integral).sum()


@pytest.mark.parametrize("kappa", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.03, 0.3, 0.85])
def test_integrated_teacher_objective_gradient_and_md_step(kappa, eta):
    p = torch.tensor([0.21, 0.32, 0.47], dtype=DTYPE, requires_grad=True)
    teacher = torch.tensor([0.4, 0.25, 0.35], dtype=DTYPE)
    objective = teacher_objective_by_quadrature(p, teacher, kappa, eta)
    gradient = torch.autograd.grad(objective, p)[0]
    expected = -normalized_power_signal((teacher / p).log(), kappa, eta)
    torch.testing.assert_close(gradient, expected, atol=1e-11, rtol=1e-11)
    md = (p.log() - eta * gradient).softmax(-1)
    target = power_log_target(p.log(), teacher.log(), kappa, eta).exp()
    torch.testing.assert_close(md, target, atol=1e-12, rtol=1e-12)
    signal = -gradient
    advantage = signal - p @ signal
    centered_step = (p.log() + eta * advantage).softmax(-1)
    torch.testing.assert_close(centered_step, target, atol=1e-12, rtol=1e-12)
    candidate = torch.tensor([0.3, 0.4, 0.3], dtype=DTYPE)
    mirror_candidate = kl(candidate, p) / eta - candidate @ advantage
    mirror_optimum = kl(target, p) / eta - target @ advantage
    torch.testing.assert_close(
        mirror_candidate - mirror_optimum,
        kl(candidate, target) / eta,
        atol=1e-11,
        rtol=1e-11,
    )
    next_objective = teacher_objective_by_quadrature(target, teacher, kappa, eta)
    assert objective - next_objective >= kl(p, target) / eta - 1e-12


@pytest.mark.parametrize("kappa", [0.0, 0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("seed", range(3))
def test_barycenter_gap_progress_and_common_divergence_improvement(kappa, seed):
    generator = torch.Generator().manual_seed(seed)
    q, teacher, candidate = torch.randn(3, 7, generator=generator, dtype=DTYPE).softmax(-1)
    eta = 0.3
    m = power_log_target(q.log(), teacher.log(), kappa, eta).exp()
    if kappa in (0, 1):
        scale = torch.tensor(1.0, dtype=DTYPE)
    else:
        z = ((1 - eta) * q.pow(kappa) + eta * teacher.pow(kappa)).pow(1 / kappa).sum()
        scale = z.pow(kappa)

    def objective(p):
        return (1 - eta) * power_divergence(q, p, kappa) + eta * power_divergence(teacher, p, kappa)

    torch.testing.assert_close(
        objective(candidate) - objective(m),
        scale * power_divergence(m, candidate, kappa),
        atol=1e-12,
        rtol=1e-12,
    )
    left = eta * (power_divergence(teacher, q, kappa) - power_divergence(teacher, candidate, kappa))
    right = (1 - eta) * power_divergence(q, candidate, kappa) + scale * (
        power_divergence(m, q, kappa) - power_divergence(m, candidate, kappa)
    )
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    for shape in (0, 0.5, 1):
        assert power_divergence(teacher, m, shape) <= power_divergence(teacher, q, shape) + 1e-12


@pytest.mark.parametrize("seed", range(4))
def test_mdpo_exact_hessian_contains_covariance_residual(seed):
    generator = torch.Generator().manual_seed(seed)
    logits, target_logits, direction = torch.randn(3, 6, generator=generator, dtype=DTYPE)
    target = target_logits.softmax(-1)
    hessian = torch.autograd.functional.hessian(lambda z: kl(z.softmax(-1), target), logits)
    p = logits.softmax(-1)
    residual = p.log() - target.log()
    centered_v = direction - p @ direction
    variance = p @ centered_v.square()
    covariance = p @ ((residual - p @ residual) * centered_v.square())
    torch.testing.assert_close(
        direction @ hessian @ direction, variance + covariance, atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("count", [2, 3, 4])
def test_center_free_u_statistic_variance_by_exhaustive_enumeration(count):
    q = torch.tensor([0.2, 0.3, 0.5], dtype=DTYPE)
    p = torch.tensor([0.4, 0.35, 0.25], dtype=DTYPE)
    teacher = torch.tensor([0.6, 0.1, 0.3], dtype=DTYPE)
    eta = 0.4
    m = power_log_target(q.log(), teacher.log(), 0.5, eta).exp()
    h = p.log() - m.log()
    score = torch.eye(3, dtype=DTYPE) - p
    centered_h = h - q @ h
    centered_score = score - q @ score
    oracle = centered_h[:, None] * centered_score / eta
    expected = q @ oracle
    vc = q @ (oracle - expected).square().sum(-1)
    pair = (
        (h[:, None] - h[None, :])[:, :, None] * (score[:, None, :] - score[None, :, :]) / (2 * eta)
    )
    pair_weights = q[:, None] * q[None, :]
    v2 = (pair_weights * (pair - expected).square().sum(-1)).sum()
    mean = torch.zeros(3, dtype=DTYPE)
    second = torch.zeros((), dtype=DTYPE)
    for sample in itertools.product(range(3), repeat=count):
        ids = torch.tensor(sample)
        weight = q[ids].prod()
        hs = h[ids]
        estimate = ((hs - hs.mean())[:, None] * score[ids]).sum(0) / (eta * (count - 1))
        mean += weight * estimate
        second += weight * estimate.square().sum()
    torch.testing.assert_close(mean, expected, atol=1e-12, rtol=1e-12)
    predicted = ((count - 2) * vc + 2 * v2) / (count * (count - 1))
    torch.testing.assert_close(second - mean.square().sum(), predicted, atol=1e-12, rtol=1e-12)
    assert predicted >= vc / count - 1e-12


def test_no_repeat_pair_covariance_improves_in_psd_order():
    q = torch.tensor([0.1, 0.25, 0.65], dtype=DTYPE)
    h = torch.tensor([-0.6, 1.1, 0.3], dtype=DTYPE)
    score = torch.tensor([[1.0, 0.2], [-0.3, 0.5], [0.4, -1.0]], dtype=DTYPE)
    old_mean = torch.zeros(2, dtype=DTYPE)
    new_mean = torch.zeros(2, dtype=DTYPE)
    old_second = torch.zeros(2, 2, dtype=DTYPE)
    new_second = torch.zeros_like(old_second)
    predicted_gap = torch.zeros_like(old_second)
    for a, b in itertools.product(range(3), repeat=2):
        kernel = (h[a] - h[b]) * (score[a] - score[b]) / 2
        weight = q[a] * q[b]
        old_mean += weight * kernel
        old_second += weight * torch.outer(kernel, kernel)
        predicted_gap += weight * q[a] * torch.outer(kernel, kernel)
        if a != b:
            new_weight = weight / (1 - q[a])
            new_kernel = (1 - q[a]) * kernel
            new_mean += new_weight * new_kernel
            new_second += new_weight * torch.outer(new_kernel, new_kernel)
    torch.testing.assert_close(old_mean, new_mean)
    torch.testing.assert_close(old_second - new_second, predicted_gap)
    assert torch.linalg.eigvalsh(old_second - new_second).min() >= -1e-12


@pytest.mark.parametrize("eta", [0.03, 0.2, 0.7, 0.97])
def test_topd_target_js_and_contraction(eta):
    q = torch.tensor([0.4, 0.35, 0.25], dtype=DTYPE)
    teacher = torch.tensor([0.1, 0.2, 0.7], dtype=DTYPE)
    m = power_log_target(q.log(), teacher.log(), 1, eta).exp()
    torch.testing.assert_close(m, (1 - eta) * q + eta * teacher)
    signal = eta * normalized_power_signal((teacher / q).log(), 1, eta)
    torch.testing.assert_close(signal, (m / q).log(), atol=1e-12, rtol=1e-12)
    objective = teacher_objective_by_quadrature(q, teacher, 1, eta)
    js = ((1 - eta) * kl(q, m) + eta * kl(teacher, m)) / (eta * (1 - eta))
    torch.testing.assert_close(objective, js, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close((m - teacher).abs().sum(), (1 - eta) * (q - teacher).abs().sum())
    for shape in (0, 0.5, 1):
        assert power_divergence(m, teacher, shape) <= (
            (1 - eta) * power_divergence(q, teacher, shape) + 1e-12
        )


def test_topd_uniform_variance_scale_has_matching_binary_witness():
    raw_variance = []
    normalized_variance = []
    for eta in (1e-3, 1e-5, 1e-7):
        q = torch.tensor([eta / 2, 1 - eta / 2], dtype=DTYPE)
        teacher = torch.tensor([0.5, 0.5], dtype=DTYPE)
        f = torch.log1p(eta * (teacher / q - 1))
        variance = q @ (f - q @ f).square()
        raw_variance.append(float(variance / eta))
        normalized_variance.append(float(eta * variance / eta**2))
    expected = math.log(2) ** 2 / 2
    assert abs(raw_variance[-1] - expected) < 1e-6
    assert abs(normalized_variance[-1] - expected) < 1e-6


def test_restricted_bpo_projection_is_not_part_stationary():
    q = torch.ones(3, dtype=DTYPE) / 3
    m = torch.tensor([0.1, 0.7, 0.2], dtype=DTYPE)
    z_b = (m[0].log() - m[1:].log().mean()).requires_grad_()
    p = torch.stack((z_b, z_b * 0, z_b * 0)).softmax(-1)
    residual = p.log() - m.log()
    bpo = (q @ (residual - q @ residual).square()) / 2
    part = (q @ residual.square()) / 2
    assert abs(float(torch.autograd.grad(bpo, z_b, retain_graph=True)[0])) < 1e-12
    torch.testing.assert_close(
        torch.autograd.grad(part, z_b)[0],
        torch.tensor(0.03543884357, dtype=DTYPE),
        atol=1e-10,
        rtol=1e-10,
    )


def test_small_bpo_loss_can_coexist_with_larger_teacher_kl():
    q = torch.tensor([0.4999995, 0.4999995, 1e-6], dtype=DTYPE)
    teacher = torch.tensor([0.8, 0.199999, 1e-6], dtype=DTYPE)
    eta = 0.3
    logm = power_log_target(q.log(), teacher.log(), 0.5, eta)
    p = (logm + torch.tensor([0, 0, 12], dtype=DTYPE)).softmax(-1)

    def bpo(candidate):
        residual = candidate.log() - logm
        return q @ (residual - q @ residual).square() / (2 * eta)

    assert bpo(p) < bpo(q) / 100
    assert kl(p, teacher) > 5 * kl(q, teacher)


@pytest.mark.parametrize(("kappa", "expected"), [(0, 6.07445), (0.5, 7.09792), (1, 10.44581)])
def test_conditional_improvement_does_not_imply_sequence_improvement(kappa, expected):
    q = torch.tensor([[0.99, 0.01], [0.5, 0.5], [1 - 1e-6, 1e-6]], dtype=DTYPE)
    teacher = torch.tensor([[0.01, 0.99], [0.5, 0.5], [1e-30, 1 - 1e-30]], dtype=DTYPE)
    m = power_log_target(q.log(), teacher.log(), kappa, 0.1).exp()

    def joint(rows):
        return torch.cat([rows[0, i] * rows[i + 1] for i in range(2)])

    for row in range(3):
        assert kl(m[row], teacher[row]) <= kl(q[row], teacher[row]) + 1e-12
    before, after = kl(joint(q), joint(teacher)), kl(joint(m), joint(teacher))
    assert after > before
    assert abs(float(after) - expected) < 6e-6


def test_mean_cubic_teacher_kl_coefficient():
    teacher = torch.tensor([0.2, 0.3, 0.5], dtype=DTYPE)
    direction = torch.tensor([-1.0, 0.2, 0.8], dtype=DTYPE)
    ratios = []

    def stable_kl(p, t):
        delta = p / t - 1
        return (t * ((1 + delta) * torch.log1p(delta) - delta)).sum()

    for epsilon in (0.1, 0.03, 0.01):
        q = (teacher.log() + epsilon * direction).softmax(-1)
        p, _ = mean_target(q, (teacher / q).log(), 1.0)
        ratios.append(float(stable_kl(p, teacher) / stable_kl(q, teacher) ** 3))
    assert abs(ratios[-1] - 1) < 0.02
    assert abs(ratios[-1] - 1) < abs(ratios[0] - 1)
