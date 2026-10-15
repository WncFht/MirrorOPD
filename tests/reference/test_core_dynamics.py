"""Independent finite-simplex checks of the analytical dynamics extensions."""

import numpy as np
import pytest
import torch


def target(q, teacher, kappa, eta):
    """Mathematical positive-branch map, including extrapolation test cases."""
    if kappa == 0:
        return ((1 - eta) * q.log() + eta * teacher.log()).softmax(0)
    power = (1 - eta) * q.pow(kappa) + eta * teacher.pow(kappa)
    if torch.any(power <= 0):
        raise ValueError("Extrapolation leaves the positive branch")
    return (power.log() / kappa).softmax(0)


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def fisher(p):
    return p.diag() - p.outer(p)


def kl(p, q):
    return p @ (p.log() - q.log())


def hellinger2(p, q):
    return (p.sqrt() - q.sqrt()).square().sum()


@pytest.mark.parametrize("kappa", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("eta", [0.02, 0.3, 0.5, 0.9])
def test_common_forward_kl_contraction(kappa, eta):
    generator = torch.Generator().manual_seed(32)
    for dimension in [2, 5, 11]:
        for _ in range(8):
            q = (3 * torch.randn(dimension, generator=generator, dtype=torch.float64)).softmax(0)
            teacher = (
                3 * torch.randn(dimension, generator=generator, dtype=torch.float64)
            ).softmax(0)
            m = target(q, teacher, kappa, eta)
            assert kl(teacher, m) <= (1 - eta) * kl(teacher, q) + 1e-12


@pytest.mark.parametrize("eta", [0.1, 0.5, 0.9, 1.1, 1.5, 2.0])
def test_root_two_anchor_identity_and_lipschitz(eta):
    teacher = tensor([0.2, 0.3, 0.5])
    q, other = tensor([0.3, 0.2, 0.5]), tensor([0.15, 0.4, 0.45])
    first, second = target(q, teacher, 0.5, eta), target(other, teacher, 0.5, eta)
    z1 = ((1 - eta) * q.sqrt() + eta * teacher.sqrt()).square().sum()
    z2 = ((1 - eta) * other.sqrt() + eta * teacher.sqrt()).square().sum()
    expected = ((1 - eta) ** 2 * hellinger2(q, other) - (z1.sqrt() - z2.sqrt()).square()) / (
        z1 * z2
    ).sqrt()
    torch.testing.assert_close(hellinger2(first, second), expected, atol=1e-14, rtol=1e-12)
    bound = (1 - eta) ** 2 / ((1 - eta) ** 2 + eta**2) if eta < 1 else (eta - 1) ** 2
    assert hellinger2(first, second) <= bound * hellinger2(q, other) + 1e-14


@pytest.mark.parametrize("kappa", [0.25, 0.5, 1])
def test_extrapolation_domain_threshold(kappa):
    q, teacher = tensor([0.8, 0.2]), tensor([0.2, 0.8])
    eta_max = float(1 / (1 - (teacher / q).min().pow(kappa)))
    valid = target(q, teacher, kappa, 1 + 0.99 * (eta_max - 1))
    assert torch.all(valid > 0)
    with pytest.raises(ValueError, match="positive branch"):
        target(q, teacher, kappa, 1 + 1.01 * (eta_max - 1))


@pytest.mark.parametrize("eta", [0.5, 1.2, 1.9, 2.0])
def test_root_angle_recurrence_and_positivity(eta):
    q, teacher = tensor([0.75, 0.2, 0.05]), tensor([0.6, 0.3, 0.1])
    for _ in range(12):
        angle = torch.acos(q.sqrt() @ teacher.sqrt())
        m = target(q, teacher, 0.5, eta)
        expected = torch.atan2(abs(1 - eta) * angle.sin(), eta + (1 - eta) * angle.cos())
        actual = torch.acos(torch.clamp(m.sqrt() @ teacher.sqrt(), max=1))
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-7)
        assert actual <= angle + 1e-12
        q = m


@pytest.mark.parametrize("kappa", [0.25, 0.5, 0.75])
def test_general_boundary_cubic_coefficient(kappa):
    teacher = tensor([0.5, 0.3, 0.2])
    direction = tensor([0.4, -0.5, 0.2])
    direction -= teacher @ direction
    errors = []
    for scale in [0.04, 0.02, 0.01]:
        y = scale * direction
        q = (teacher.log() + y).softmax(0)
        q2 = target(target(q, teacher, kappa, 2), teacher, kappa, 2)
        y2 = (q2 / teacher).log()
        y2 -= teacher @ y2
        predicted = y - 2 * kappa * (1 - kappa) * (teacher @ y.square()) * y
        errors.append(float(torch.linalg.vector_norm(y2 - predicted)))
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 16, rtol=0.12)


def test_root_boundary_rate_and_arithmetic_cycle():
    teacher, initial = tensor([0.5, 0.5]), tensor([0.9, 0.1])
    q = initial.clone()
    for _ in range(6000):
        q = target(q, teacher, 0.5, 2)
    assert float(6000 * kl(q, teacher)) == pytest.approx(1, rel=0.01)
    assert float(6000 * kl(teacher, q)) == pytest.approx(1, rel=0.01)
    torch.testing.assert_close(target(target(initial, teacher, 1, 2), teacher, 1, 2), initial)


def test_root_nonzero_cycle_above_two():
    teacher = tensor([0.5, 0.5])
    x = 0.5 + 11 * np.sqrt(23) / 144
    q = tensor([x, 1 - x])
    after = target(q, teacher, 0.5, 2.2)
    torch.testing.assert_close(after, q.flip(0))
    torch.testing.assert_close(target(after, teacher, 0.5, 2.2), q)
    assert kl(after, teacher) > 0.1


@pytest.mark.parametrize("algorithm", ["bpo", "mdpo"])
@pytest.mark.parametrize("kappa", [0, 0.5, 1])
@pytest.mark.parametrize("eta", [0.35, 1.5, 2.2])
def test_joint_outer_jacobian_with_shared_parameters(algorithm, kappa, eta):
    teacher = tensor([0.55, 0.3, 0.14, 0.01])
    features = tensor([[0.2, 0.5], [-0.7, 0.1], [0.4, -0.3], [0.3, 0.8]])
    alpha, steps = 0.12, 7

    def outer(anchor):
        q = (teacher.log() + features @ anchor).softmax(0)
        m = target(q, teacher, kappa, eta)
        current = anchor.clone()
        for _ in range(steps):
            logp = (teacher.log() + features @ current).log_softmax(0)
            if algorithm == "bpo":
                residual = logp - m.log()
                loss = (q @ (residual - q @ residual).square()) / (2 * eta)
            else:
                loss = (logp.exp() @ (logp - m.log())) / eta
            gradient = torch.autograd.grad(loss, current, create_graph=True)[0]
            current = current - alpha * gradient
        return current

    actual = torch.autograd.functional.jacobian(outer, torch.zeros(2, dtype=torch.float64))
    metric = features.T @ fisher(teacher) @ features
    identity = torch.eye(2, dtype=torch.float64)
    expected = (1 - eta) * identity + eta * torch.linalg.matrix_power(
        identity - alpha * metric / eta, steps
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)


def test_bpo_exact_flow_matches_loss_autograd_and_spectrum():
    q, teacher = tensor([0.7, 0.2, 0.09, 0.01]), tensor([0.3, 0.35, 0.3, 0.05])
    eta = 0.4
    m = target(q, teacher, 0.5, eta)
    h0 = q.log() - m.log()
    metric = fisher(q) / eta
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    coefficients = eigenvectors.T @ h0
    previous_rate = float("inf")
    for time in [0.0, 1.0, 5.0, 20.0]:
        h = torch.matrix_exp(-time * metric) @ h0
        z = (m.log() + h).requires_grad_()
        residual = z.log_softmax(0) - m.log()
        loss = (q @ (residual - q @ residual).square()) / (2 * eta)
        gradient = torch.autograd.grad(loss, z)[0]
        torch.testing.assert_close(gradient, metric @ h)
        terms = eigenvalues * coefficients.square() * (-2 * eigenvalues * time).exp() / 2
        torch.testing.assert_close(loss, terms.sum())
        rate = float(2 * (terms @ eigenvalues) / terms.sum())
        assert rate <= previous_rate
        previous_rate = rate


def test_finite_inner_stability_boundary_and_shape_counterexamples():
    eta, maximum = 3.0, 0.7
    threshold = eta / maximum * np.log(eta / (eta - 2))
    for factor, stable in [(0.9, True), (1.1, False)]:
        r = 1 - eta + eta * np.exp(-maximum * threshold * factor / eta)
        assert bool(abs(r) < 1) == stable
    for x, t, increasing in [(0.05, 0.2, False), (0.6, 0.95, True)]:
        q, teacher = tensor([x, 1 - x]), tensor([t, 1 - t])
        losses = [float(kl(target(q, teacher, k, 0.5), teacher)) for k in [0, 0.5, 1]]
        differences = np.diff(losses)
        assert np.all(differences > 0) if increasing else np.all(differences < 0)


def test_time_varying_kernel_rate_derivative():
    q, m = tensor([0.2, 0.3, 0.5]), tensor([0.5, 0.35, 0.15])
    eta = 0.4
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher(q))
    root = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T

    def logits(theta):
        x, y = theta.unbind()
        return torch.stack([x.square() + y, x.sin(), x * y])

    def geometry(theta):
        residual = root @ (logits(theta) - m.log())
        jacobian = torch.autograd.functional.jacobian(logits, theta, create_graph=True)
        kernel = root @ jacobian @ jacobian.T @ root
        return residual, kernel

    theta = tensor([0.4, -0.2]).requires_grad_()
    b, kernel = geometry(theta)
    norm = b @ b
    loss = norm / (2 * eta)
    velocity = -torch.autograd.grad(loss, theta, create_graph=True)[0]
    rate = 2 * (b @ kernel @ b) / (eta * norm)
    actual = torch.autograd.grad(rate, theta)[0] @ velocity
    derivative = torch.autograd.functional.jacobian(lambda x: geometry(x)[1], theta)
    kernel_dot = torch.einsum("ijk,k->ij", derivative, velocity)
    expected = (
        -4 / eta**2 * ((b @ kernel @ kernel @ b) / norm - ((b @ kernel @ b) / norm).square())
        + 2 / eta * (b @ kernel_dot @ b) / norm
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
