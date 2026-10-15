"""Optional per-prefix measurements, separate from the gradient program."""

import torch

from mirroropd.core.distributions import grouped_kl, support_overlap


@torch.no_grad()
def student_statistics(distribution, values, context, *, logp=None):
    head = distribution.selected(values["head_ids"])
    tail = distribution.tail(values["head_ids"], context.estimator.chunk_size)
    if context.estimator.center == "full":
        qlog = values["anchor_full"]
        plog = distribution.full() if logp is None else logp
        center = (qlog.exp() * (qlog - plog)).sum(-1)
    else:
        center = grouped_kl(values["anchor_head"], values["anchor_tail"], head, tail)
    return center, head, tail


@torch.no_grad()
def full_diagnostics(distribution, values, context, head, tail):
    teacher_ids = values.get("teacher_ids")
    if teacher_ids is None:
        raise ValueError("Full diagnostics require cached teacher support")
    ids = distribution.logits.topk(context.estimator.topj, -1).indices
    return {
        "student_entropy": distribution.entropy(context.estimator.chunk_size),
        "anchor_head_mass": values["anchor_head"].exp().sum(-1),
        "anchor_tail_mass": values["anchor_tail"].exp(),
        "student_head_mass": head.exp().sum(-1),
        "student_tail_mass": tail.exp(),
        "student_anchor_support_jaccard": support_overlap(ids, values["head_ids"]),
        "student_teacher_support_jaccard": support_overlap(ids, teacher_ids),
        "anchor_teacher_support_jaccard": support_overlap(values["head_ids"], teacher_ids),
    }
