"""Check OPD Mean bounds, outer contraction, loss identity and LOO centering."""

import itertools

import torch

from mirroropd.reference.categorical import mean_target, pmd_mean_loss


def kl(p, q):
    return p @ (p.log() - q.log())


def span(p, t):
    v = p.log() - t.log()
    return v.max() - v.min()


def test_reference_identities():
    torch.manual_seed(20261008)
    count = 0
    for dimension, eta, _ in itertools.product((2, 4, 8), (0.1, 0.5, 1.0, 2.0), range(8)):
        q = torch.randn(dimension, dtype=torch.float64).softmax(-1)
        t = torch.randn(dimension, dtype=torch.float64).softmax(-1)
        reward = t.log() - q.log()
        p, multiplier = mean_target(q, reward, eta)
        g = eta * (reward - q @ reward)
        a, b = q @ g.exp(), q @ (2 * g).exp()
        assert a * (a - 1) / b <= multiplier + 1e-10
        assert multiplier <= a.log() + 1e-10
        candidate = torch.randn(dimension, dtype=q.dtype).softmax(-1)
        excess = pmd_mean_loss(candidate.log(), q, reward, eta) - pmd_mean_loss(
            p.log(), q, reward, eta
        )
        identity = 0.5 * (q @ (candidate.log() - p.log()).square()) + multiplier * kl(p, candidate)
        torch.testing.assert_close(excess, identity, atol=2e-10, rtol=2e-9)
        if eta <= 1:
            bound = kl(q, t) - (1 / eta - 1) * kl(p, q)
            bound -= multiplier * (q @ (p / q - 1).square()) / (2 * eta)
            assert kl(p, t) <= bound + 1e-10
            old_span = span(q, t)
            assert span(p, t) <= (1 - eta / (1 + eta * old_span)) * old_span + 1e-10
            assert kl(p, t) < kl(q, t)
        count += 1
    # Enumerate LOO loss, not just a Monte Carlo approximation.
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    t = torch.tensor([0.5, 0.4, 0.1], dtype=q.dtype)
    reward = t.log() - q.log()
    logits = torch.tensor([0.1, -0.8, 0.4], dtype=q.dtype, requires_grad=True)
    u = logits.log_softmax(-1) - q.log()
    eta, n = 0.7, 3
    sampled = torch.zeros((), dtype=q.dtype)
    for actions in itertools.product(range(3), repeat=n):
        loss = sum(
            (
                u[actions[i]]
                - eta
                * (
                    reward[actions[i]]
                    - sum(reward[actions[j]] for j in range(n) if j != i) / (n - 1)
                )
            ).square()
            / (2 * n)
            for i in range(n)
        )
        sampled = sampled + q[list(actions)].prod() * loss
    exact = pmd_mean_loss(logits, q, reward, eta)
    expected = exact + eta**2 * (q @ (reward - q @ reward).square()) / (2 * (n - 1))
    torch.testing.assert_close(sampled, expected)
    torch.testing.assert_close(
        torch.autograd.grad(sampled, logits)[0], torch.autograd.grad(exact, logits)[0]
    )
    # High-temperature limit keeps all positive centered reward actions.
    h = reward - q @ reward
    limit = q * h.clamp_min(0)
    limit /= limit.sum()
    errors = []
    for eta in (10.0, 50.0, 200.0):
        p, multiplier = mean_target(q, reward, eta)
        errors.append(float((p - limit).abs().max()))
        assert abs(multiplier / eta - float(q @ h.clamp_min(0))) < 0.2
    assert errors[2] < errors[1] < errors[0]
    print(f"PASS: {count} random OPD cases; multiplier bounds, exact excess-loss identity,")
    print("teacher KL descent, span contraction, enumerated LOO loss/gradient and large-eta limit.")
    print(f"Large-eta max probability errors: {errors}")
