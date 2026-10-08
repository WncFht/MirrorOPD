"""Shared stopped-weight OPD score surrogates for token batches.

Inputs are log probabilities on the SAME temperature-adjusted vocabulary.
Rollout actions must come from the frozen anchor (no top-p truncation).
KL centers are computed separately on anchor top-k plus aggregate tail.
"""

import torch


def coefficients(method, log_student, log_anchor, log_teacher, teacher_center, student_center, eta):
    if eta <= 0:
        raise ValueError("eta must be positive")
    u = log_student - log_anchor
    reward = log_teacher - log_anchor
    if method == "pmd_mean":
        return u - eta * (reward + teacher_center)
    if method == "bpo":
        return (u + student_center) / eta - reward - teacher_center
    if method == "mdpo":
        return u / eta - reward
    raise ValueError(f"unknown method: {method}")


def sampled_pg_loss(
    method,
    log_student,
    log_anchor,
    log_teacher,
    teacher_center,
    student_center,
    eta,
    mask=None,
):
    """Frozen-anchor action MC surrogate; only log_student receives gradient."""
    coefficient = coefficients(
        method,
        log_student,
        log_anchor,
        log_teacher,
        teacher_center,
        student_center,
        eta,
    ).detach()
    ratio = (
        (log_student - log_anchor).detach().exp()
        if method == "mdpo"
        else torch.ones_like(log_student)
    )
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("nonfinite MDPO importance ratio")
    terms = coefficient * ratio * log_student
    if mask is None:
        return terms.mean()
    valid = mask.bool()
    if not valid.any():
        raise ValueError("empty response mask")
    return terms[valid].mean()
