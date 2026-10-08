#!/usr/bin/env python3
"""Enumerated token-level inner audit.

Every method sees one frozen categorical anchor q and teacher T.  There is no
rollout or sequence-level objective: the experiment separates the Lambert-W
PMD-Mean target from the Gibbs target and measures finite-step optimization.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from token_inner import (
    bpo_loss,
    mdpo_loss,
    mean_target,
    pmd_mean_loss,
    pmd_part_loss,
)


def softmax(z: torch.Tensor) -> torch.Tensor:
    return torch.softmax(z, dim=-1)


def run(
    method: str, eta: float, steps: int = 400, lr: float = 0.15
) -> dict[str, float]:
    torch.manual_seed(20261008)
    q = torch.tensor([0.62, 0.23, 0.10, 0.05], dtype=torch.float64)
    teacher = torch.tensor([0.08, 0.17, 0.25, 0.50], dtype=torch.float64)
    logits = torch.log(torch.tensor([0.70, 0.15, 0.10, 0.05], dtype=torch.float64))
    target = softmax((1 - eta) * torch.log(q) + eta * torch.log(teacher))
    reward = torch.log(teacher) - torch.log(q)
    lambert_target, multiplier = mean_target(q, reward, eta)
    own_target = lambert_target if method == "pmd_mean" else target
    for _ in range(steps):
        logits = logits.detach().requires_grad_(True)
        p = softmax(logits)
        if method == "pmd_part":
            loss = pmd_part_loss(logits, q, reward, eta)
            loss.backward()
            grad = logits.grad
            logits = logits - lr * grad
        elif method == "pmd_mean":
            loss = pmd_mean_loss(logits, q, reward, eta)
            loss.backward()
            logits = logits - lr * logits.grad
        elif method == "bpo":
            loss = bpo_loss(logits, q, reward, eta)
            loss.backward()
            logits = logits - lr * logits.grad
        elif method == "mdpo":
            loss = mdpo_loss(logits, q, reward, eta)
            loss.backward()
            # Exact categorical Fisher natural gradient in centered logit coordinates.
            g = logits.grad
            fisher = torch.diag(p) - p[:, None] * p[None, :]
            nat = torch.linalg.pinv(fisher) @ g
            logits = logits - min(lr, 0.5) * nat
        else:
            raise ValueError(method)
    p = softmax(logits.detach())
    return {
        "method": method,
        "eta": eta,
        "target_tv": float(0.5 * torch.abs(p - target).sum()),
        "target_kl": float(torch.sum(p * (torch.log(p) - torch.log(target)))),
        "own_target_tv": float(0.5 * torch.abs(p - own_target).sum()),
        "own_target_kl": float(torch.sum(p * (torch.log(p) - torch.log(own_target)))),
        "structural_tv": float(0.5 * torch.abs(own_target - target).sum()),
        "mean_multiplier": multiplier if method == "pmd_mean" else 0.0,
        "teacher_kl": float(torch.sum(p * (torch.log(p) - torch.log(teacher)))),
    }


def main() -> None:
    rows = [
        run(m, eta)
        for eta in (0.25, 0.5, 0.8)
        for m in ("pmd_part", "pmd_mean", "bpo", "mdpo")
    ]
    out = Path(__file__).with_name("toy_token_inner.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
