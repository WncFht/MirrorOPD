"""Finite-simplex checks for the consolidated analytical appendices."""

import itertools

import numpy as np
import pytest
import torch

from mirroropd.reference.categorical import mean_target

D = torch.float64


def t(values):
    return torch.tensor(values, dtype=D)


def variance(p, x):
    return p @ (x - p @ x).square()


def bpo(z, q, m, eta):
    return variance(q, z - m.log()) / (2 * eta)


def test_mean_and_gibbs_third_order_difference():
    q, reward = t([0.2, 0.3, 0.5]), t([-0.8, 0.4, 0.7])
    errors = []
    for eta in [0.04, 0.02, 0.01]:
        mean, _ = mean_target(q, reward, eta)
        gibbs = (q.log() + eta * reward).softmax(0)
        errors.append((mean - gibbs).abs().sum().item())
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 8, rtol=0.04)


def test_pair_gradient_second_moment_and_expected_contraction():
    q, m, z = t([0.1, 0.3, 0.6]), t([0.4, 0.2, 0.4]), t([0.4, -0.2, 0.1])
    eta = 0.4
    alpha = eta * q.min()
    loss = bpo(z, q, m, eta)
    mean, moment, after = torch.zeros(3, dtype=D), 0.0, 0.0
    basis = torch.eye(3, dtype=D)
    residual = z - m.log()
    for a, b in itertools.product(range(3), repeat=2):
        g = (residual[a] - residual[b]) * (basis[a] - basis[b]) / (2 * eta)
        weight = q[a] * q[b]
        mean += weight * g
        moment += weight * g.square().sum()
        after += weight * bpo(z - alpha * g, q, m, eta)
    exact = torch.autograd.functional.jacobian(lambda x: bpo(x, q, m, eta), z)
    torch.testing.assert_close(mean, exact)
    torch.testing.assert_close(moment, 2 * loss / eta)
    assert after <= (1 - alpha * q.min() / eta) * loss
    gamma = 0.7
    diagonal_step = (z - gamma * eta * exact / q).softmax(0)
    torch.testing.assert_close(diagonal_step, ((1 - gamma) * z + gamma * m.log()).softmax(0))


@pytest.mark.parametrize("eta", [0.001, 0.2, 0.5, 0.9, 1.0])
def test_joint_square_root_fitting_bound(eta):
    q, teacher = t([0.001, 0.099, 0.9]), t([0.8, 0.19, 0.01])
    root = (1 - eta) * q.sqrt() + eta * teacher.sqrt()
    normalizer = root.square().sum()
    m = root.square() / normalizer
    a, b = q.sqrt() @ teacher.sqrt(), q.sqrt() @ m.sqrt()
    coefficient = 2 * (b - (m / q).sqrt()) / eta
    torch.testing.assert_close(q @ coefficient.square(), 4 * (1 - a * a) / normalizer)
    assert q @ coefficient.square() <= 8


def test_envelope_center_and_nested_head_variances():
    q, teacher = t([0.01, 0.19, 0.3, 0.5]), t([0.6, 0.1, 0.2, 0.1])
    mu = (q + teacher) / 2
    phi = ((teacher / q).sqrt() - 1).square()
    value = q * phi / mu
    center = q @ phi
    assert variance(mu, value) <= center * (2 - center) + 1e-14
    signed = 2 * ((teacher / q).sqrt() - 1)
    assert variance(mu, value) <= variance(q, signed) / 2 + 1e-14
    for count in [1, 2, 5]:
        assert variance(q, signed) + variance(mu, value) / count <= 4 + 1 / (count * (count + 1))
    previous = float("inf")
    for head in range(4):
        u = (q + teacher)[head:]
        nu = u / u.sum()
        sampled = u.sum() * (q * phi)[head:] / u
        current = variance(nu, sampled)
        assert current <= previous + 1e-14
        torch.testing.assert_close(nu @ sampled + (q * phi)[:head].sum(), center)
        previous = current


def test_tailmatch_resopd_noise_and_offpolicy_control_coupling():
    p, teacher = t([0.4, 0.2, 0.4]), t([0.7, 0.1, 0.2])
    score = torch.eye(3, dtype=D) - p
    tail_mean = (p[1:, None] * score[1:]).sum(0) / p[1:].sum()
    ell = (p[1:] / teacher[1:]).log()
    residual = ell[:, None] * (score[1:] - tail_mean)
    trace = (p[1:] * residual.square().sum(1)).sum()
    torch.testing.assert_close(trace, t(4 / 15) * t(2).log().square())
    torch.testing.assert_close((p[1:, None] * residual).sum(0), torch.zeros(3, dtype=D))
    probability, proposal, chi = t([0.5, 0.5]), t([0.25, 0.75]), t([1, -1])
    w = probability / proposal
    cross = proposal @ ((w - 1) * w * chi)
    torch.testing.assert_close(cross, t(2 / 3))


def test_empirical_gradient_gaps_and_frozen_center_bias():
    q, m = t([0.1, 0.3, 0.6]), t([0.4, 0.2, 0.4])
    z = t([0.4, -0.2, 0.1]).requires_grad_()
    p = z.softmax(0)
    score = torch.eye(3, dtype=D) - p
    eta = 0.3
    pool = torch.tensor([0, 0, 2])
    raw = p.log() - m.log()
    residual = raw - q @ raw
    empirical = residual[pool].square().mean() / (2 * eta)
    actual = torch.autograd.grad(empirical, z, retain_graph=True)[0]
    pg = (residual[pool, None] * score[pool]).mean(0) / eta
    torch.testing.assert_close(actual - pg, -residual[pool].mean() * (q @ score) / eta)
    for shift in [0.2, -0.3]:
        original = q @ raw.square() / 2
        changed = q @ (raw + shift).square() / 2
        bias = torch.autograd.grad(changed - original, z, retain_graph=True)[0]
        torch.testing.assert_close(bias, shift * (q @ score))


@pytest.mark.parametrize("eta", [0.2, 0.8, 1.3, 1.8])
def test_biased_geometric_stationary_and_stale_target(eta):
    q, teacher, err = t([0.1, 0.3, 0.6]), t([0.4, 0.2, 0.4]), t([0.03, -0.02, 0.01])

    def span(v):
        return v.max() - v.min()

    fixed = (teacher.log() + err / eta).softmax(0)
    out = ((1 - eta) * fixed.log() + eta * teacher.log() + err).softmax(0)
    torch.testing.assert_close(out, fixed)
    torch.testing.assert_close(span((fixed / teacher).log()), span(err) / eta)
    m1 = ((1 - eta) * q.log() + eta * teacher.log()).softmax(0)
    m2 = ((1 - eta) * fixed.log() + eta * teacher.log()).softmax(0)
    torch.testing.assert_close(span((m1 / m2).log()), abs(1 - eta) * span((q / fixed).log()))


def test_equal_rate_convolution_and_mean_noise_bound():
    c, budget, initial = 0.7, 0.03, 0.5
    b = initial
    for k in range(1, 12):
        b = c * b + budget * c ** (k - 1)
        assert b == pytest.approx(c**k * initial + budget * k * c ** (k - 1))
    error = 0.02

    def h(x):
        return np.exp(x) * x * x / (8 + np.exp(x) * x * x)

    b = 2 * error
    for _ in range(100):
        b = h(b) * b + error
        assert b <= 2 * error
    assert b <= error / (1 - h(2 * error))


@pytest.mark.parametrize("beta", [0.0, 0.5, 0.9])
def test_fixed_momentum_jury_boundary(beta):
    bound = 2 * (1 + beta) / (1 - beta)
    for factor, stable in [(0.9, True), (1.1, False)]:
        step = factor * bound
        matrix = np.array([[1 - step * (1 - beta), -step * beta], [1 - beta, beta]])
        assert bool(np.max(np.abs(np.linalg.eigvals(matrix))) < 1) == stable
