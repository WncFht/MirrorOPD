"""Production PMD-mean, BPO and MDPO token-score surrogates."""

import math

import torch


def compute_mirror_opd_token_loss(
    student_logits: torch.Tensor,
    actions: torch.Tensor,
    anchor_ids: torch.Tensor,
    anchor_head_log_probs: torch.Tensor,
    anchor_tail_log_prob: torch.Tensor,
    anchor_action_log_prob: torch.Tensor,
    teacher_action_log_prob: torch.Tensor,
    teacher_center: torch.Tensor,
    method: str,
    eta: float,
) -> dict[str, torch.Tensor]:
    """Token score surrogate on frozen anchor actions (no PPO clipping).

    Logits must already have the rollout temperature applied. All cache
    tensors refer to the same prefix and tokenizer. The anchor support stays
    fixed throughout inner updates. Unlike TA-OPD's direct KL loss, centers
    and residuals here stop gradient after deriving the score formula.
    This primitive operates on a local complete vocabulary, not TP shards.
    """
    if not math.isfinite(eta) or eta <= 0 or method not in ("pmd_mean", "bpo", "mdpo"):
        raise ValueError("positive eta and a supported MirrorOPD method required")
    expected = student_logits.shape[:-1]
    if any(
        t.shape != expected
        for t in (
            actions,
            anchor_tail_log_prob,
            anchor_action_log_prob,
            teacher_action_log_prob,
            teacher_center,
        )
    ):
        raise ValueError("all scalar cache fields must align with prefix dimensions")
    if anchor_ids.shape != anchor_head_log_probs.shape or anchor_ids.shape[:-1] != expected:
        raise ValueError("anchor head ids/log-probs must align with prefixes")
    log_p = torch.log_softmax(student_logits.float(), dim=-1)
    action_lp = log_p.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        q_head = anchor_head_log_probs.detach().float()
        p_head = log_p.detach().gather(-1, anchor_ids)
        head = (q_head.exp() * (q_head - p_head)).sum(-1)
        outside = torch.ones_like(log_p, dtype=torch.bool)
        outside.scatter_(-1, anchor_ids, False)
        p_tail_lp = torch.logsumexp(log_p.detach().masked_fill(~outside, float("-inf")), -1)
        q_tail_lp = anchor_tail_log_prob.detach().float()
        tail = torch.zeros_like(head)
        nonempty = torch.isfinite(q_tail_lp)
        tail[nonempty] = q_tail_lp[nonempty].exp() * (q_tail_lp[nonempty] - p_tail_lp[nonempty])
        center = head + tail
        u = action_lp.detach() - anchor_action_log_prob.detach()
        reward = teacher_action_log_prob.detach() - anchor_action_log_prob.detach()
        kt = teacher_center.detach()
        if method == "pmd_mean":
            coefficient = u - eta * (reward + kt)
        elif method == "bpo":
            coefficient = (u + center) / eta - reward - kt
        else:
            coefficient = (u / eta - reward) * u.exp()
        if not torch.isfinite(coefficient).all():
            raise FloatingPointError("nonfinite MirrorOPD weight; no silent ratio clipping")
    return {
        "distillation_losses": coefficient * action_lp,
        "mirror_student_center": center,
        "mirror_teacher_center": teacher_center.detach(),
        "mirror_importance_ratio": u.exp(),
        "mirror_score_coefficient": coefficient,
        "mirror_sampled_reward": reward,
        "mirror_sampled_log_ratio": u,
        "mirror_student_entropy": -(log_p.detach().exp() * log_p.detach()).sum(-1),
        "mirror_anchor_head_mass": q_head.exp().sum(-1),
        "mirror_student_head_mass": p_head.exp().sum(-1),
        "mirror_anchor_tail_mass": q_tail_lp.exp(),
        "mirror_student_tail_mass": p_tail_lp.exp(),
    }
