"""Finite-vocabulary OPD runner used to validate outer/inner dynamics.

The categorical backend mirrors the production contract: an outer policy
rolls out prefixes, the anchor and teacher distributions are frozen for the
whole inner solve, and a new rollout is made after synchronization. Tail-
aware KL is used only as a scalar center; its probability support is detached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mirroropd.opd_pg import coefficients, sampled_pg_loss


def tail_aware_kl(log_p: torch.Tensor, log_q: torch.Tensor, top_k: int) -> torch.Tensor:
    """Aggregate reverse KL on a detached top-k support plus one tail token."""
    p, q = log_p.exp(), log_q.exp()
    if top_k <= 0:
        return torch.zeros((), dtype=log_p.dtype, device=log_p.device)
    k = min(top_k, p.numel())
    ids = torch.topk(p.detach(), k=k).indices
    head = (p[ids] * (log_p[ids] - log_q[ids])).sum()
    mask = torch.ones_like(p, dtype=torch.bool)
    mask[ids] = False
    p_tail, q_tail = p[mask].sum(), q[mask].sum()
    tail = p_tail * (torch.log(p_tail) - torch.log(q_tail)) if p_tail > 0 else p_tail
    return head + tail


def exact_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    p = log_p.exp()
    return (p * (log_p - log_q)).sum()


@dataclass
class CategoricalOPDConfig:
    method: Literal["pmd_mean", "bpo", "mdpo"]
    eta: float = 0.5
    top_k: int = 4
    inner_steps: int = 40
    inner_lr: float = 0.08
    outer_steps: int = 8
    rollout_batch: int = 256
    soft_sync: float | None = None
    sampled_inner: bool = False


def _rollout(policy: torch.Tensor, batch: int) -> torch.Tensor:
    return torch.multinomial(policy.detach(), batch, replacement=True)


def run_categorical_opd(
    anchor0: torch.Tensor,
    teacher: torch.Tensor,
    config: CategoricalOPDConfig,
) -> list[dict[str, float]]:
    """Run a one-prefix outer loop and return auditable scalar diagnostics."""
    anchor = anchor0.detach().double().clone()
    if config.eta <= 0 or config.rollout_batch < 1 or config.top_k < 0:
        raise ValueError("invalid eta, rollout batch or support size")
    if config.soft_sync is not None and not 0 < config.soft_sync <= 1:
        raise ValueError("soft sync weight must be in (0,1]")
    teacher = teacher.detach().double()
    history: list[dict[str, float]] = []
    for outer in range(config.outer_steps):
        actions = _rollout(anchor, config.rollout_batch)
        logits = anchor.log().detach().clone().requires_grad_(True)
        frozen_anchor = anchor.detach()
        log_anchor = frozen_anchor.log()
        log_teacher = teacher.log()
        # Tail-aware centers are evaluated from the frozen outer anchor.
        teacher_kl = tail_aware_kl(log_anchor, log_teacher, config.top_k)
        for _ in range(config.inner_steps):
            log_p = torch.log_softmax(logits, -1)
            student_kl = tail_aware_kl(log_anchor, log_p, config.top_k)
            coefficient = coefficients(
                config.method,
                log_p,
                log_anchor,
                log_teacher,
                teacher_kl,
                student_kl,
                config.eta,
            )
            weights = log_p.exp().detach() if config.method == "mdpo" else frozen_anchor
            # This is an analytic score-function surrogate, not the original
            # squared loss: both residual and integration weights stop here.
            if config.sampled_inner:
                loss = sampled_pg_loss(
                    config.method,
                    log_p[actions],
                    log_anchor[actions],
                    log_teacher[actions],
                    teacher_kl,
                    student_kl,
                    config.eta,
                )
            else:
                loss = (weights * coefficient.detach() * log_p).sum()
            grad = torch.autograd.grad(loss, logits)[0]
            with torch.no_grad():
                logits -= config.inner_lr * grad
            logits = logits.detach().requires_grad_(True)
        inner_policy = logits.detach().softmax(-1)
        if config.soft_sync is None:
            next_anchor = inner_policy
        else:
            rho = config.soft_sync
            next_anchor = (1.0 - rho) * anchor + rho * inner_policy
        history.append(
            {
                "outer": float(outer),
                "rollout_unique": float(actions.unique().numel()),
                "teacher_tail_aware": float(teacher_kl),
                "teacher_exact": float(exact_kl(anchor.log(), log_teacher)),
                "student_anchor_exact": float(exact_kl(anchor.log(), inner_policy.log())),
                "inner_to_teacher_exact": float(exact_kl(inner_policy.log(), log_teacher)),
                "inner_loss": float(loss.detach()),
            }
        )
        anchor = next_anchor.detach()
    return history
