"""Reusable fixed-measure categorical inner updates for MirrorOPD."""
import torch


def mirror_target(anchor, teacher, eta):
    return torch.softmax((1.0 - eta) * torch.log(anchor) + eta * torch.log(teacher), dim=-1)


def pmd_part_loss(logits, anchor, reward, eta):
    p = torch.softmax(logits, -1)
    residual = torch.log(p) - torch.log(anchor) - eta * reward
    return 0.5 * torch.sum(anchor * residual.square())


def pmd_mean_loss(logits, anchor, reward, eta):
    p = torch.softmax(logits, -1)
    u = torch.log(p) - torch.log(anchor)
    centered_u = u - torch.sum(anchor * u)
    centered_r = reward - torch.sum(anchor * reward)
    return 0.5 * torch.sum(anchor * (centered_u - eta * centered_r).square())


def bpo_objective(logits, anchor, reward, eta):
    p = torch.softmax(logits, -1)
    return torch.sum(p * reward) - torch.sum(p * (torch.log(p) - torch.log(anchor))) / eta


def mdpo_step(logits, anchor, reward, eta, step_size=1.0):
    """One exact categorical natural-gradient step on the common objective."""
    logits = logits.detach().requires_grad_(True)
    objective = bpo_objective(logits, anchor, reward, eta)
    grad = torch.autograd.grad(objective, logits)[0]
    p = torch.softmax(logits, -1)
    fisher = torch.diag(p) - torch.outer(p, p)
    nat = torch.linalg.pinv(fisher) @ grad
    return (logits + step_size * nat).detach()
