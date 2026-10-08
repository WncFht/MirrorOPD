"""Categorical oracle checks for reward-only Mean and centered-residual BPO."""

import itertools

import torch

from mirroropd.token_inner import bpo_loss, mdpo_loss, mean_target, pmd_mean_loss, pmd_part_loss


def close(actual, expected, atol=2e-10):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-9)


def check_case(q, reward, eta, logits):
    logits = logits.detach().requires_grad_(True)
    p = logits.softmax(-1)
    u = logits.log_softmax(-1) - q.log()
    g = eta * (reward - q @ reward)
    kl = -(q @ u)
    mean = pmd_mean_loss(logits, q, reward, eta)
    bpo = bpo_loss(logits, q, reward, eta)
    gm = torch.autograd.grad(mean, logits, retain_graph=True)[0]
    gb = torch.autograd.grad(bpo, logits, retain_graph=True)[0]
    gkl = torch.autograd.grad(kl, logits)[0]
    score = torch.eye(len(q), dtype=q.dtype) - p[None, :]
    close(gm, (q * (u - g)) @ score)
    close(gb, (q * (u + kl - g)) @ score / eta)
    close(mean, eta * bpo + 0.5 * kl.square())
    close(gm, eta * gb + kl * gkl)
    for loss, gradient in ((pmd_mean_loss, gm), (bpo_loss, gb)):
        for j in range(len(q)):
            shift = torch.zeros_like(logits)
            shift[j] = 1e-5
            finite = (
                loss(logits + shift, q, reward, eta) - loss(logits - shift, q, reward, eta)
            ) / 2e-5
            close(finite, gradient[j], atol=2e-9)
    target, multiplier = mean_target(q, reward, eta)
    ut = target.log() - q.log()
    close(target.sum(), torch.tensor(1.0, dtype=q.dtype))
    close(ut - g + multiplier * target / q, torch.zeros_like(q))
    close(torch.tensor(multiplier, dtype=q.dtype), q @ (q.log() - target.log()))
    stationary = torch.autograd.functional.jacobian(
        lambda x: pmd_mean_loss(x, q, reward, eta), target.log()
    )
    close(stationary, torch.zeros_like(q))
    gibbs = (q.log() + eta * reward).softmax(-1)
    for loss in (pmd_part_loss, bpo_loss):
        close(loss(gibbs.log(), q, reward, eta), torch.tensor(0.0, dtype=q.dtype))
    # At the anchor, the extra KL penalty has zero gradient.
    at_anchor = q.log().requires_grad_(True)
    close(
        torch.autograd.grad(pmd_mean_loss(at_anchor, q, reward, eta), at_anchor)[0],
        eta * torch.autograd.grad(bpo_loss(at_anchor, q, reward, eta), at_anchor)[0],
    )


def check_sample_correction():
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    reward = torch.tensor([-0.4, 0.7, 1.2], dtype=torch.float64)
    logits = torch.tensor([0.1, -0.8, 0.4], dtype=torch.float64, requires_grad=True)
    eta = 0.6
    z = eta * reward - logits.log_softmax(-1) + q.log()
    exact = bpo_loss(logits, q, reward, eta)
    for k in (1, 2, 3):
        sampled = torch.zeros((), dtype=q.dtype)
        for indices in itertools.product(range(len(q)), repeat=k + 1):
            probability = q[list(indices)].prod()
            residual = z[indices[0]] - z[list(indices[1:])].mean()
            sampled = sampled + probability * (k / (k + 1)) * residual.square() / (2 * eta)
        close(sampled, exact)
        close(
            torch.autograd.grad(sampled, logits, retain_graph=True)[0],
            torch.autograd.grad(exact, logits, retain_graph=True)[0],
        )
    beta = torch.tensor([0.5, 0.4, 0.1], dtype=q.dtype)
    weighted = sum(
        beta[a] * beta[b] * (q[a] / beta[a]) * (q[b] / beta[b]) * (z[a] - z[b]).square() / (4 * eta)
        for a, b in itertools.product(range(3), repeat=2)
    )
    close(weighted, exact)
    close(torch.autograd.grad(weighted, logits)[0], torch.autograd.grad(exact, logits)[0])


def check_inner_geometry():
    q = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    teacher = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    reward = teacher.log() - q.log()
    x = torch.tensor([0.2, -0.4, 0.8], dtype=q.dtype, requires_grad=True)
    projector = torch.eye(3, dtype=q.dtype) - torch.ones(3, 3, dtype=q.dtype) / 3
    anchor_fisher = torch.diag(q) - q[:, None] * q[None, :]
    for eta in (0.1, 0.3, 1.0, 1.5):
        target_logits = q.log() + eta * reward
        hessian = torch.autograd.functional.hessian(
            lambda z, step_eta=eta: bpo_loss(z, q, reward, step_eta), x
        )
        close(hessian, anchor_fisher / eta)
        p = x.softmax(-1)
        current_fisher = torch.diag(p) - p[:, None] * p[None, :]
        for objective, fisher in ((bpo_loss, anchor_fisher), (mdpo_loss, current_fisher)):
            gradient = torch.autograd.grad(objective(x, q, reward, eta), x)[0]
            natural = torch.linalg.pinv(fisher) @ gradient
            error = projector @ (x - target_logits)
            for ratio in (0.5, 1.0, 1.9, 2.0, 2.2):
                updated_error = projector @ (x - eta * ratio * natural - target_logits)
                close(updated_error, (1 - ratio) * error)
    mean, _ = mean_target(q, reward, 1.0)
    assert float((mean - teacher).abs().sum()) > 0.47
    print("PASS: BPO Hessian and exact Fisher dynamics, stable/oscillating/divergent step regimes")


def main():
    torch.manual_seed(20261008)
    cases = 0
    for dimension, eta, _ in itertools.product((2, 4, 8), (0.1, 0.5, 1.0, 2.0), range(4)):
        q = torch.randn(dimension, dtype=torch.float64).softmax(-1)
        reward = torch.randn(dimension, dtype=torch.float64)
        logits = torch.randn(dimension, dtype=torch.float64)
        check_case(q, reward, eta, logits)
        cases += 1
    q = torch.tensor([0.5, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.75, 0.25], dtype=torch.float64)
    target, multiplier = mean_target(q, teacher.log() - q.log(), 1.0)
    close(target[0], torch.tensor(0.729186955209, dtype=q.dtype), atol=1e-10)
    close(mean_target(q, torch.ones_like(q), 0.7)[0], q)
    check_sample_correction()
    check_inner_geometry()
    print(
        f"PASS: {cases} random cases; analytic gradients, finite differences, loss/gradient identities,"
    )
    print(
        "Lambert-W normalization/KKT/stationarity, Gibbs zeros, first-step scaling, constant reward,"
    )
    print("enumerated finite-sample correction and off-behavior pair weighting.")
    print(f"Two-action Mean={target.tolist()}, Gibbs={teacher.tolist()}, lambda={multiplier:.12f}")


if __name__ == "__main__":
    main()
