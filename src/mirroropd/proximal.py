"""Full-support power-mean targets and exact or sampled conditional fitting losses."""

import math

import torch


def worst_fisher_speed_squared(kappa):
    """Sharp full-simplex supremum of the infinitesimal power-target velocity."""
    if kappa <= 0 or kappa > 0.5:
        return math.inf
    if kappa == 0.5:
        return 4.0
    mass = (1 - 2 * kappa) / (2 * (1 - kappa))
    return mass ** (1 - 2 * kappa) * (1 - mass) / kappa**2


def minimax_velocity_shape():
    """Unique minimizer of the sharp worst-case Fisher velocity ceiling."""
    low, high = 0.01, 0.499999
    for _ in range(80):
        middle = (low + high) / 2
        mass = (1 - 2 * middle) / (2 * (1 - middle))
        if math.log(mass) + 1 / middle > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def power_log_target(anchor_logp, teacher_logp, kappa, strength):
    """Freeze a normalized target; shape zero is the geometric limit."""
    if not 0 <= kappa <= 1 or not 0 <= strength <= 1:
        raise ValueError("Power shape and interpolation strength must lie in [0,1]")
    if anchor_logp.shape != teacher_logp.shape:
        raise ValueError("Anchor and teacher must have the same complete support")
    if strength == 0:
        return anchor_logp.detach()
    if strength == 1:
        return teacher_logp.detach()
    if kappa == 0:
        raw = (1 - strength) * anchor_logp + strength * teacher_logp
    else:
        raw = (
            torch.logaddexp(
                math.log1p(-strength) + kappa * anchor_logp,
                math.log(strength) + kappa * teacher_logp,
            )
            / kappa
        )
    return (raw - torch.logsumexp(raw, -1, keepdim=True)).detach()


def power_signal(log_ratio, kappa, strength):
    """Unnormalized frozen signal; additive target normalizers cancel in scores."""
    if not 0 <= kappa <= 1 or not 0 < strength < 1:
        raise ValueError("Positive interior interpolation strength required")
    if kappa == 0:
        return strength * log_ratio
    return (
        torch.logaddexp(
            torch.full_like(log_ratio, math.log1p(-strength)),
            math.log(strength) + kappa * log_ratio,
        )
        / kappa
    )


def target_divergence(student_logp, target_logp, kappa, fit="reverse_kl", anchor_logp=None):
    """Return one true conditional loss per prefix, retaining student derivatives."""
    if not 0 <= kappa <= 1 or fit not in ("reverse_kl", "power", "bpo"):
        raise ValueError("Unknown target-fitting divergence")
    if fit == "bpo":
        if anchor_logp is None:
            raise ValueError("BPO needs the frozen anchor measure")
        residual = student_logp - target_logp
        centered = residual - (anchor_logp.exp() * residual).sum(-1, keepdim=True)
        return (anchor_logp.exp() * centered.square()).sum(-1) / 2
    if fit == "reverse_kl" or kappa == 0:
        return (student_logp.exp() * (student_logp - target_logp)).sum(-1)
    if kappa == 1:
        return (target_logp.exp() * (target_logp - student_logp)).sum(-1)
    return -(student_logp.exp() * torch.expm1(kappa * (target_logp - student_logp))).sum(-1) / (
        kappa * (1 - kappa)
    )


def sampled_target_loss(
    student_logits, anchor_logp, target_logp, actions, kappa, strength, fit="reverse_kl"
):
    """Unbiased proposal-sampled gradient with the fitting loss derived analytically.

    Actions have shape (..., M). The proposal may be the frozen anchor or current
    student. BPO requires the frozen anchor proposal and its exact residual centre. Importance ratios and score coefficients stop
    gradient after analytic differentiation. The objective is divided by the
    interpolation strength so shape-zero matches the original MDPO scale.
    """
    if not 0 < strength <= 1 or not 0 <= kappa <= 1:
        raise ValueError("Positive strength at most one and valid shape required")
    logp = student_logits.float().log_softmax(-1)
    qlog, mlog = anchor_logp.detach(), target_logp.detach()
    lp = logp.gather(-1, actions)
    with torch.no_grad():
        log_ratio = lp.detach() - qlog.gather(-1, actions)
        weight = log_ratio.exp()
        if fit == "bpo":
            residual = logp.detach() - mlog
            center = (qlog.exp() * residual).sum(-1)
            coefficient = lp.detach() - mlog.gather(-1, actions) - center.unsqueeze(-1)
        elif fit == "reverse_kl" or kappa == 0:
            center = target_divergence(logp.detach(), mlog, 0)
            coefficient = weight * (lp.detach() - mlog.gather(-1, actions) - center.unsqueeze(-1))
        elif fit == "power":
            centered_power = (
                logp.detach().exp() * torch.expm1(kappa * (mlog - logp.detach()))
            ).sum(-1)
            selected_power = torch.expm1(kappa * (mlog.gather(-1, actions) - lp.detach()))
            coefficient = weight * (centered_power.unsqueeze(-1) - selected_power) / kappa
        else:
            raise ValueError("Unknown target-fitting divergence")
        coefficient = coefficient / strength
        if not torch.isfinite(coefficient).all():
            raise FloatingPointError("Nonfinite power-target score; no silent ratio clipping")
    return {
        "loss": (coefficient * lp).mean(-1),
        "true_loss": target_divergence(logp.detach(), mlog, kappa, fit, anchor_logp=qlog),
        "coefficient": coefficient,
        "importance_ratio": weight,
    }


def renyi_inner_bound(student_logp, target_logp, teacher_logp, orders=(0.01, 0.03, 0.1, 0.3, 1.0)):
    """A finite-order upper certificate for student-to-teacher reverse KL."""
    epsilon = target_divergence(student_logp, target_logp, 0)
    bounds = [
        (1 + 1 / order) * epsilon
        + torch.logsumexp((1 + order) * target_logp - order * teacher_logp, -1) / order
        for order in orders
    ]
    return torch.stack(bounds).amin(0)


def sample_variance_bpo_loss(
    student_logits, target_logp, actions, strength, importance_weights=None
):
    """Unbiased BPO sample-variance loss for iid frozen-anchor actions, M >= 2.

    Actions are auxiliary tokens at one frozen prefix, not additional responses.
    Optional detached weights apply to each unordered sample pair; for M=2
    they may factor a correlated proposal joint weight.
    Both conditional centers and logit normalization cancel. The M-1 denominator
    is necessary. Fresh draws give a population gradient at fixed parameters;
    reusing a finite draw optimizes its empirical loss only.
    """
    count = actions.shape[-1]
    if count < 2 or not 0 < strength <= 1:
        raise ValueError("BPO sample variance requires M >= 2 and positive strength <= 1")
    if student_logits.shape != target_logp.shape:
        raise ValueError("Student logits and target log weights must have the same shape")
    logits = (
        student_logits.float()
        if student_logits.dtype in (torch.float16, torch.bfloat16)
        else student_logits
    )
    residual = logits.gather(-1, actions) - target_logp.detach().gather(-1, actions)
    if importance_weights is not None:
        if importance_weights.shape != actions.shape:
            raise ValueError("Proposal weights must align with auxiliary actions")
        weights = importance_weights.detach()
        difference = residual.unsqueeze(-1) - residual.unsqueeze(-2)
        pair_weights = weights.unsqueeze(-1) * weights.unsqueeze(-2)
        return {
            "loss": (pair_weights * difference.square()).sum((-1, -2))
            / (4 * strength * count * (count - 1)),
            "coefficient": (
                weights * (weights.unsqueeze(-2) * difference).sum(-1) / (strength * (count - 1))
            ).detach(),
            "importance_ratio": weights,
        }
    centered = residual - residual.mean(-1, keepdim=True)
    return {
        "loss": centered.square().sum(-1) / (2 * strength * (count - 1)),
        "coefficient": centered.detach() * count / (strength * (count - 1)),
        "importance_ratio": torch.ones_like(centered),
    }
