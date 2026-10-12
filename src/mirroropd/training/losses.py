"""Production PMD-mean, BPO and MDPO token-score surrogates."""

import math

import torch

from mirroropd.training.support import topk_pair_overlap


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
    teacher_head_log_probs: torch.Tensor | None = None,
    teacher_tail_log_prob: torch.Tensor | None = None,
    teacher_topk_ids: torch.Tensor | None = None,
    full_anchor_log_probs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Token score surrogate on frozen anchor actions (no PPO clipping).

    Logits must already have the rollout temperature applied. All cache
    tensors refer to the same prefix and tokenizer. The anchor support stays
    fixed throughout inner updates. Unlike TA-OPD's direct KL loss, centers
    and residuals here stop gradient after deriving the score formula.
    This primitive operates on a local complete vocabulary, not TP shards.
    Supplying full_anchor_log_probs replaces only the student center with the
    exact full-vocabulary KL; the caller must supply the corresponding frozen
    full teacher center. Score coefficients still stop gradient.
    """
    if (
        math.isnan(eta)
        or eta <= 0
        or method not in ("pmd_mean", "bpo", "mdpo", "opd_onpolicy", "opd_is", "opd_is_no_baseline")
        or (
            math.isinf(eta)
            and method not in ("mdpo", "opd_onpolicy", "opd_is", "opd_is_no_baseline")
        )
    ):
        raise ValueError("positive eta required; infinity is supported only for MDPO")
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
        # Bound diagnostic scratch space independently of response length.
        flat_lp = log_p.detach().reshape(-1, log_p.shape[-1])
        flat_ids = anchor_ids.reshape(flat_lp.shape[0], anchor_ids.shape[-1])
        flat_tail = torch.empty(flat_lp.shape[0], device=log_p.device)
        flat_entropy = torch.empty_like(flat_tail)
        for start in range(0, flat_lp.shape[0], 64):
            stop = start + 64
            block = flat_lp[start:stop]
            tail_block = block.clone()
            tail_block.scatter_(-1, flat_ids[start:stop], float("-inf"))
            flat_tail[start:stop] = torch.logsumexp(tail_block, -1)
            del tail_block
            flat_entropy[start:stop] = -(block.exp() * block).sum(-1)
        p_tail_lp = flat_tail.reshape(expected)
        entropy = flat_entropy.reshape(expected)
        q_tail_lp = anchor_tail_log_prob.detach().float()
        tail = torch.zeros_like(head)
        nonempty = torch.isfinite(q_tail_lp)
        tail[nonempty] = q_tail_lp[nonempty].exp() * (q_tail_lp[nonempty] - p_tail_lp[nonempty])
        center = head + tail
        if full_anchor_log_probs is not None:
            if full_anchor_log_probs.shape != log_p.shape:
                raise ValueError(
                    "full anchor probabilities must align with the complete vocabulary"
                )
            full_q = full_anchor_log_probs.detach().float()
            center = (full_q.exp() * (full_q - log_p.detach())).sum(-1)
        u = action_lp.detach() - anchor_action_log_prob.detach()
        reward = teacher_action_log_prob.detach() - anchor_action_log_prob.detach()
        kt = teacher_center.detach()
        if method == "pmd_mean":
            coefficient = u - eta * (reward + kt)
        elif method == "bpo":
            coefficient = (u + center) / eta - reward - kt
        elif method == "mdpo":
            coefficient = (u / eta - reward) * u.exp()
        else:
            if teacher_head_log_probs is None or teacher_tail_log_prob is None:
                raise ValueError("OPD baseline needs teacher head and tail probabilities")
            current_reward = teacher_action_log_prob.detach() - action_lp.detach()
            baseline = (p_head.exp() * (p_head - teacher_head_log_probs.detach())).sum(-1)
            tail_valid = torch.isfinite(p_tail_lp)
            baseline[tail_valid] += p_tail_lp[tail_valid].exp() * (
                p_tail_lp[tail_valid] - teacher_tail_log_prob.detach()[tail_valid]
            )
            if method == "opd_is_no_baseline":
                baseline = torch.zeros_like(baseline)
            weight = torch.ones_like(u) if method == "opd_onpolicy" else u.exp()
            coefficient = -(current_reward + baseline) * weight
            reward = current_reward
            kt = baseline
        if not torch.isfinite(coefficient).all():
            raise FloatingPointError("nonfinite MirrorOPD weight; no silent ratio clipping")
        # Each pair compares native Top-J sets, independently of the loss support.
        support_k = anchor_ids.shape[-1]
        current_ids = log_p.topk(support_k, dim=-1).indices
        overlap_fields = topk_pair_overlap(current_ids, anchor_ids, "student_anchor")
        if teacher_topk_ids is not None:
            if teacher_topk_ids.shape != anchor_ids.shape:
                raise ValueError("teacher Top-J ids must align with anchor Top-J ids")
            overlap_fields.update(
                topk_pair_overlap(current_ids, teacher_topk_ids, "student_teacher")
            )
            overlap_fields.update(topk_pair_overlap(anchor_ids, teacher_topk_ids, "anchor_teacher"))
        support_overlap = overlap_fields["mirror_overlap_student_anchor_intersection_fraction"]
    return {
        **overlap_fields,
        "distillation_losses": coefficient * action_lp,
        "mirror_student_center": center,
        "mirror_teacher_center": kt,
        "mirror_importance_ratio": u.exp(),
        "mirror_score_coefficient": coefficient,
        "mirror_token_advantage": coefficient,
        "mirror_support_overlap": support_overlap,
        "mirror_action_nll": -action_lp.detach(),
        "mirror_sampled_reward": reward,
        "mirror_sampled_log_ratio": u,
        "mirror_student_entropy": entropy,
        "mirror_anchor_head_mass": q_head.exp().sum(-1),
        "mirror_student_head_mass": p_head.exp().sum(-1),
        "mirror_anchor_tail_mass": q_tail_lp.exp(),
        "mirror_student_tail_mass": p_tail_lp.exp(),
    }
