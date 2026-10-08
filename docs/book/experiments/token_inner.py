"""Reusable fixed-measure categorical inner updates for MirrorOPD."""

import math

import torch


def mirror_target(anchor, teacher, eta):
    return torch.softmax(
        (1.0 - eta) * torch.log(anchor) + eta * torch.log(teacher), dim=-1
    )


def pmd_part_loss(logits, anchor, reward, eta):
    logp = torch.log_softmax(logits, -1)
    logz = torch.logsumexp(torch.log(anchor) + eta * reward, dim=-1)
    residual = logp - torch.log(anchor) - eta * reward + logz
    return 0.5 * torch.sum(anchor * residual.square())


def pmd_mean_loss(logits, anchor, reward, eta):
    u = torch.log_softmax(logits, -1) - torch.log(anchor)
    centered_r = reward - torch.sum(anchor * reward)
    return 0.5 * torch.sum(anchor * (u - eta * centered_r).square())


def bpo_loss(logits, anchor, reward, eta):
    """Unlinearized BPO: centered optimality residual under frozen anchor."""
    u = torch.log_softmax(logits, -1) - torch.log(anchor)
    z = u - eta * reward
    residual = z - torch.sum(anchor * z)
    return torch.sum(anchor * residual.square()) / (2 * eta)


def mdpo_loss(logits, anchor, reward, eta):
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    return torch.sum(p * (logp - torch.log(anchor))) / eta - torch.sum(p * reward)


def mean_target(anchor, reward, eta):
    """Solve the finite categorical Lambert-W normalization, in float64.

    The scalar multiplier belongs to the unscaled half-square loss. This is
    a one-history oracle, not a differentiable training layer.
    """
    q = anchor.detach().to(dtype=torch.float64)
    g = eta * (reward.detach().to(dtype=torch.float64) - torch.sum(q * reward.detach()))
    values = g.tolist()
    weights = q.tolist()

    def lambert_w(x):
        if x == 0:
            return 0.0
        low, high = 0.0, max(1.0, math.log1p(x))
        for _ in range(100):
            mid = (low + high) / 2
            if mid + math.log(mid) < math.log(x):
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def probabilities(multiplier):
        return [
            weight * math.exp(value - lambert_w(multiplier * math.exp(value)))
            for weight, value in zip(weights, values, strict=True)
        ]

    if max(values) - min(values) == 0:
        return q.clone(), 0.0
    low, high = 0.0, 1.0
    while sum(probabilities(high)) > 1:
        high *= 2
    for _ in range(80):
        mid = (low + high) / 2
        if sum(probabilities(mid)) > 1:
            low = mid
        else:
            high = mid
    multiplier = (low + high) / 2
    return torch.tensor(
        probabilities(multiplier), dtype=q.dtype, device=q.device
    ), multiplier


def mdpo_step(logits, anchor, reward, eta, step_size=1.0):
    """One exact categorical natural-gradient step on the common objective."""
    logits = logits.detach().requires_grad_(True)
    loss = mdpo_loss(logits, anchor, reward, eta)
    grad = torch.autograd.grad(loss, logits)[0]
    p = torch.softmax(logits, -1)
    fisher = torch.diag(p) - torch.outer(p, p)
    nat = torch.linalg.pinv(fisher) @ grad
    return (logits - step_size * nat).detach()
