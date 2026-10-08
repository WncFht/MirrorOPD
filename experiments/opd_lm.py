"""Causal-LM token scoring/cache contract for Tail-aware MirrorOPD.

No padding is required: process one trajectory or sequence microbatch at a
time. Production rollout workers return prompt/generated ids; EOS belongs
to generated ids, padding never does. Logits are temperature-adjusted before
both scoring and generation. Frozen cache contains O(tokens * top_k) values.
"""

from dataclasses import dataclass

import torch
from opd_pg import sampled_pg_loss


@dataclass
class PrefixCache:
    input_ids: torch.Tensor
    actions: torch.Tensor
    prompt_length: int
    support: torch.Tensor
    anchor_head: torch.Tensor
    anchor_tail: torch.Tensor
    anchor_action: torch.Tensor
    teacher_action: torch.Tensor
    teacher_center: torch.Tensor


def log_probs(model, ids, prompt_length, temperature=1.0):
    if temperature <= 0:
        raise ValueError("positive sampling temperature required")
    logits = model(input_ids=ids.unsqueeze(0), use_cache=False).logits[0]
    # Input is prompt + generated[:-1], hence last logits predicts last action.
    return (logits[prompt_length - 1 :].float() / temperature).log_softmax(-1)


def aggregate_kl(anchor_head, anchor_tail, other_log_probs, support):
    other_head = other_log_probs.gather(-1, support)
    mask = torch.ones_like(other_log_probs, dtype=torch.bool)
    mask.scatter_(-1, support, False)
    other_tail_log = torch.logsumexp(other_log_probs.masked_fill(~mask, float("-inf")), -1)
    head = (anchor_head.exp() * (anchor_head - other_head)).sum(-1)
    positive = anchor_tail > 0
    # Do not compute 0 * infinity for full-vocabulary support.
    tail = torch.zeros_like(head)
    tail[positive] = anchor_tail[positive] * (
        anchor_tail[positive].log() - other_tail_log[positive]
    )
    return head + tail


@torch.no_grad()
def freeze_cache(anchor, teacher, prompt, generated, top_k, device="cpu", temperature=1.0):
    if not prompt or not generated:
        raise ValueError("nonempty prompt and generated token ids required")
    ids = torch.tensor(prompt + generated[:-1], dtype=torch.long, device=device)
    actions = torch.tensor(generated, dtype=torch.long, device=device)
    a = log_probs(anchor, ids, len(prompt), temperature)
    t = log_probs(teacher, ids, len(prompt), temperature)
    if a.shape != t.shape:
        raise ValueError("student and teacher vocabulary shapes must agree")
    if not 0 <= top_k <= a.shape[-1]:
        raise ValueError("invalid top_k")
    support = a.topk(top_k, dim=-1).indices
    head = a.gather(-1, support)
    mask = torch.ones_like(a, dtype=torch.bool)
    mask.scatter_(-1, support, False)
    tail = torch.logsumexp(a.masked_fill(~mask, float("-inf")), -1).exp()
    return PrefixCache(
        ids.cpu(),
        actions.cpu(),
        len(prompt),
        support.cpu(),
        head.cpu(),
        tail.cpu(),
        a.gather(-1, actions[:, None]).squeeze(-1).cpu(),
        t.gather(-1, actions[:, None]).squeeze(-1).cpu(),
        aggregate_kl(head, tail, t, support).cpu(),
    )


def train_trajectory(student, cache, method, eta, temperature=1.0):
    device = next(student.parameters()).device
    lp = log_probs(student, cache.input_ids.to(device), cache.prompt_length, temperature)
    actions = cache.actions.to(device)
    student_action = lp.gather(-1, actions[:, None]).squeeze(-1)
    with torch.no_grad():
        center = aggregate_kl(
            cache.anchor_head.to(device),
            cache.anchor_tail.to(device),
            lp.detach(),
            cache.support.to(device),
        )
    return sampled_pg_loss(
        method,
        student_action,
        cache.anchor_action.to(device),
        cache.teacher_action.to(device),
        cache.teacher_center.to(device),
        center,
        eta,
    )
