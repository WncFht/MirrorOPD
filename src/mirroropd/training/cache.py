"""Frozen outer-anchor and teacher caches for production MirrorOPD."""

import torch


def build_mirror_anchor_cache(
    anchor_logits: torch.Tensor,
    actions: torch.Tensor,
    valid: torch.Tensor,
    topk: int,
) -> dict[str, torch.Tensor]:
    """Anchor-worker output: O(prefixes * topk), with no full logits transfer.

    Logits already include the rollout temperature. The worker can release
    them before a separate teacher forward, retaining this frozen cache only.
    """
    if anchor_logits.ndim < 2:
        raise ValueError("anchor logits must have (..., vocab) shape")
    if actions.shape != anchor_logits.shape[:-1] or valid.shape != actions.shape:
        raise ValueError("actions and valid must align with prediction positions")
    vocab = anchor_logits.shape[-1]
    if not 0 <= topk <= vocab:
        raise ValueError(f"topk must lie in [0, {vocab}]")
    valid = valid.bool()
    safe_actions = torch.where(valid, actions, torch.zeros_like(actions))
    anchor_logp = torch.log_softmax(
        anchor_logits.detach().float().masked_fill(~valid.unsqueeze(-1), 0), dim=-1
    )
    anchor_ids = anchor_logp.topk(topk, dim=-1).indices
    anchor_head = anchor_logp.gather(-1, anchor_ids).detach()
    outside = torch.ones_like(anchor_logp, dtype=torch.bool)
    outside.scatter_(-1, anchor_ids, False)
    anchor_tail = anchor_logp.masked_fill(~outside, float("-inf")).logsumexp(-1)
    anchor_action = anchor_logp.gather(-1, safe_actions.unsqueeze(-1)).squeeze(-1)
    result = {
        "mirror_valid": valid.bool(),
        "mirror_actions": actions,
        "mirror_anchor_ids": anchor_ids,
        "mirror_anchor_head_log_probs": anchor_head,
        "mirror_anchor_tail_log_prob": anchor_tail,
        "mirror_anchor_action_log_prob": anchor_action.detach(),
        "mirror_vocab_size": torch.full_like(actions, vocab),
    }
    # Do not allow invalid prompt/terminal locations to leak arbitrary cache
    # values into diagnostics; loss adapter still masks them before gather.
    for key, value in result.items():
        if key == "mirror_valid":
            continue
        result[key] = torch.where(
            valid.unsqueeze(-1) if value.ndim == valid.ndim + 1 else valid,
            value,
            torch.zeros_like(value),
        )
    return result


def complete_mirror_teacher_cache(
    teacher_logits: torch.Tensor,
    anchor_cache: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Score anchor-selected ids locally and return compact frozen fields.

    The teacher must share the anchor's tokenizer and vocabulary semantics.
    Native teacher top-k results cannot substitute for these requested ids.
    """
    valid = anchor_cache["mirror_valid"].bool()
    actions = anchor_cache["mirror_actions"]
    ids = anchor_cache["mirror_anchor_ids"]
    qhead = anchor_cache["mirror_anchor_head_log_probs"].detach().float()
    qtail = anchor_cache["mirror_anchor_tail_log_prob"].detach().float()
    if teacher_logits.shape[:-1] != valid.shape or actions.shape != valid.shape:
        raise ValueError("teacher logits and anchor cache must align with prediction positions")
    if ids.shape != qhead.shape or ids.shape[:-1] != valid.shape or qtail.shape != valid.shape:
        raise ValueError("anchor head and tail shapes must align with prediction positions")
    vocab = teacher_logits.shape[-1]
    if ids.shape[-1] > vocab:
        raise ValueError("anchor support is larger than the teacher vocabulary")
    if ((anchor_cache["mirror_vocab_size"] != vocab) & valid).any():
        raise ValueError("anchor and teacher vocabulary sizes must match")
    teacher_logp = torch.log_softmax(
        teacher_logits.detach().float().masked_fill(~valid.unsqueeze(-1), 0), dim=-1
    )
    teacher_head = teacher_logp.gather(-1, ids)
    outside = torch.ones_like(teacher_logp, dtype=torch.bool)
    outside.scatter_(-1, ids, False)
    teacher_tail = teacher_logp.masked_fill(~outside, float("-inf")).logsumexp(-1)
    center = (qhead.exp() * (qhead - teacher_head)).sum(-1)
    nonempty = torch.isfinite(qtail) & valid
    tail = torch.zeros_like(center)
    tail[nonempty] = qtail[nonempty].exp() * (qtail[nonempty] - teacher_tail[nonempty])
    center = center + tail
    safe_actions = torch.where(valid, actions, torch.zeros_like(actions))
    action_lp = teacher_logp.gather(-1, safe_actions.unsqueeze(-1)).squeeze(-1)
    result = dict(anchor_cache)
    result.pop("mirror_vocab_size")
    result["mirror_teacher_action_log_prob"] = torch.where(valid, action_lp, 0).detach()
    result["mirror_teacher_center"] = torch.where(valid, center, 0).detach()
    teacher_ids = teacher_logp.topk(ids.shape[-1], dim=-1).indices
    result["mirror_teacher_topk_ids"] = torch.where(valid.unsqueeze(-1), teacher_ids, 0).detach()
    result["mirror_teacher_head_log_probs"] = torch.where(
        valid.unsqueeze(-1), teacher_head, 0
    ).detach()
    result["mirror_teacher_tail_log_prob"] = torch.where(valid, teacher_tail, 0).detach()
    return result


def build_mirror_opd_cache_from_full_logits(
    anchor_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    actions: torch.Tensor,
    valid: torch.Tensor,
    topk: int,
) -> dict[str, torch.Tensor]:
    """Full-logit oracle for the sequential compact worker cache contract."""
    if anchor_logits.shape != teacher_logits.shape:
        raise ValueError("anchor and teacher logits must have identical (..., vocab) shape")
    anchor_cache = build_mirror_anchor_cache(anchor_logits, actions, valid, topk)
    return complete_mirror_teacher_cache(teacher_logits, anchor_cache)
