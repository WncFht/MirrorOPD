"""Exact sampled centers from frozen head inputs; no retained N-by-V tensor.

Projection keeps the active linear's arithmetic (including low precision
rounding before temperature division); probability reductions use FP32/FP64.
All statistics are stopped. Only normalized sampled action LP has a backward.
"""

import torch
import torch.nn.functional as F

from mirroropd.core.distributions import probability_dtype, support_overlap
from mirroropd.core.signal import power_reward


class ProjectedDistribution:
    """A token tile whose vocabulary is projected and released one block at a time."""

    def __init__(self, hidden, weight, bias, temperature, chunk):
        self.hidden, self.weight, self.bias = hidden, weight, bias
        self.temperature, self.chunk = temperature, chunk

    def blocks(self):
        for start in range(0, len(self.weight), self.chunk):
            stop = start + self.chunk
            z = (
                F.linear(
                    self.hidden,
                    self.weight[start:stop],
                    self.bias[start:stop] if self.bias is not None else None,
                )
                / self.temperature
            )
            yield start, z.to(probability_dtype(z))

    @torch.no_grad()
    def statistics(self, actions, topj, support=None):
        n, dtype = len(self.hidden), probability_dtype(self.hidden)
        logz = torch.full((n,), -torch.inf, dtype=dtype, device=self.hidden.device)
        action = torch.zeros_like(logz)
        top = logz.new_empty((n, 0))
        ids = actions.new_empty((n, 0))
        selected = logz.new_zeros(support.shape) if support is not None else None
        tail = torch.full_like(logz, -torch.inf)
        for start, z in self.blocks():
            width = z.shape[-1]
            logz = torch.logaddexp(logz, z.logsumexp(-1))
            inside = (actions >= start) & (actions < start + width)
            action += (
                z.gather(-1, (actions - start).clamp(0, width - 1)[:, None]).squeeze(-1) * inside
            )
            if topj:
                candidates = torch.cat((top, z), -1)
                candidate_ids = torch.cat(
                    (ids, torch.arange(start, start + width, device=z.device).expand(n, -1)), -1
                )
                top, where = candidates.topk(min(topj, candidates.shape[-1]), -1)
                ids = candidate_ids.gather(-1, where)
            if support is not None:
                inside = (support >= start) & (support < start + width)
                selected += z.gather(-1, (support - start).clamp(0, width - 1)) * inside
                masked = z.clone()
                # Mask only members of this block, without clobbering edge rows.
                rows, columns = inside.nonzero(as_tuple=True)
                masked[rows, support[rows, columns] - start] = -torch.inf
                tail = torch.logaddexp(tail, masked.logsumexp(-1))
        return {
            "logz": logz,
            "action": action - logz,
            "ids": ids,
            "head": top - logz[:, None],
            "selected": selected - logz[:, None] if support is not None else None,
            "tail": tail - logz if support is not None else None,
        }

    @torch.no_grad()
    def entropy(self, logz):
        result = torch.zeros_like(logz)
        for _, z in self.blocks():
            lp = z - logz[:, None]
            result -= (lp.exp() * lp).sum(-1)
        return result


@torch.no_grad()
def center(left, right, qlogz, plogz, signal=None):
    """KL(q||p), or the negative q expectation of the existing power reward."""
    result = torch.zeros_like(qlogz)
    for (start, qz), (other, pz) in zip(left.blocks(), right.blocks(), strict=True):
        if start != other or qz.shape != pz.shape:
            raise ValueError("Projected distributions require the same vocabulary tiles")
        q, p = qz - qlogz[:, None], pz - plogz[:, None]
        contribution = q - p if signal is None else -power_reward(p - q, signal)
        result += (q.exp() * contribution).sum(-1)
    return result


class _ActionLogProbability(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, actions, logz, action_lp, temperature, chunk):
        ctx.save_for_backward(hidden, weight, bias, actions, logz)
        ctx.temperature, ctx.chunk = temperature, chunk
        ctx.autocast = torch.is_autocast_enabled(hidden.device.type)
        ctx.projection_dtype = (
            torch.get_autocast_dtype(hidden.device.type) if ctx.autocast else hidden.dtype
        )
        return action_lp.clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        h, w, b, actions, logz = ctx.saved_tensors
        dtype = probability_dtype(h)
        dh = torch.zeros_like(h, dtype=dtype) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(w, dtype=probability_dtype(w)) if ctx.needs_input_grad[1] else None
        db = (
            torch.zeros_like(b, dtype=probability_dtype(b))
            if b is not None and ctx.needs_input_grad[2]
            else None
        )
        with torch.autocast(h.device.type, enabled=ctx.autocast, dtype=ctx.projection_dtype):
            distribution = ProjectedDistribution(h, w, b, ctx.temperature, ctx.chunk)
            for start, z in distribution.blocks():
                stop = start + z.shape[-1]
                g = -(z - logz[:, None]).exp()
                inside = (actions >= start) & (actions < stop)
                g.scatter_add_(
                    -1,
                    (actions - start).clamp(0, z.shape[-1] - 1)[:, None],
                    inside.to(dtype)[:, None],
                )
                # Match linear -> temperature division -> probability upcast.
                g = ((g * upstream[:, None]).to(ctx.projection_dtype) / ctx.temperature).to(dtype)
                with torch.autocast(h.device.type, enabled=False):
                    if dh is not None:
                        dh += g @ w[start:stop].to(ctx.projection_dtype).to(dtype)
                    if dw is not None:
                        dw[start:stop] += g.T @ h.to(ctx.projection_dtype).to(dtype)
                    if db is not None:
                        db[start:stop] += g.sum(0)
        # AMP casts inputs before linear: its backward crosses the same cast
        # edge. Sum vocabulary contributions before rounding the hidden grad.
        return (
            dh.to(ctx.projection_dtype).to(h.dtype) if dh is not None else None,
            dw.to(ctx.projection_dtype).to(w.dtype) if dw is not None else None,
            db.to(ctx.projection_dtype).to(b.dtype) if db is not None else None,
            None,
            None,
            None,
            None,
            None,
        )


def sampled_loss(
    hidden, weight, bias, cache, objective, frozen_weight, frozen_bias, *, token_chunk, vocab_chunk
):
    from mirroropd.core.cache import select_valid
    from mirroropd.core.diagnostics import diagnostic_names
    from mirroropd.core.objectives import LossResult, score_coefficient

    cache.context.validate(objective)
    packed = hidden.reshape(-1, hidden.shape[-1])[cache.valid.reshape(-1)]
    pieces, diagnostics = [], {}
    for start in range(0, len(packed), token_chunk):
        h = packed[start : start + token_chunk]
        frozen = select_valid(cache, start, start + token_chunk)
        p = ProjectedDistribution(h, weight, bias, cache.context.temperature, vocab_chunk)
        q = ProjectedDistribution(
            frozen.anchor_hidden, frozen_weight, frozen_bias, cache.context.temperature, vocab_chunk
        )
        with torch.no_grad():
            stats = p.statistics(
                frozen.actions,
                objective.estimator.topj if objective.diagnostics == "full" else 0,
                frozen.head_ids if objective.diagnostics == "full" else None,
            )
            ks = center(q, p, frozen.anchor_logz, stats["logz"])
            u = stats["action"] - frozen.anchor_action
            reward = power_reward(frozen.teacher_action - frozen.anchor_action, objective.signal)
            coefficient = score_coefficient(
                objective.algorithm, u, reward, frozen.reward_center, ks, objective.signal.eta
            )
            metrics = {}
            if objective.diagnostics != "none":
                metrics = {
                    "action_nll": -stats["action"],
                    "frozen_reward_center": frozen.reward_center,
                    "student_anchor_kl": ks,
                    "reward": reward,
                    "score_coefficient": coefficient,
                    "log_anchor_ratio": u,
                }
            if objective.diagnostics == "full":
                metrics.update(
                    student_entropy=p.entropy(stats["logz"]),
                    anchor_head_mass=frozen.anchor_head.exp().sum(-1),
                    anchor_tail_mass=frozen.anchor_tail.exp(),
                    student_head_mass=stats["selected"].exp().sum(-1),
                    student_tail_mass=stats["tail"].exp(),
                    student_anchor_support_jaccard=support_overlap(stats["ids"], frozen.head_ids),
                    student_teacher_support_jaccard=support_overlap(
                        stats["ids"], frozen.teacher_ids
                    ),
                    anchor_teacher_support_jaccard=support_overlap(
                        frozen.head_ids, frozen.teacher_ids
                    ),
                )
        action_lp = _ActionLogProbability.apply(
            h,
            weight,
            bias,
            frozen.actions,
            stats["logz"],
            stats["action"],
            cache.context.temperature,
            vocab_chunk,
        )
        loss = coefficient * action_lp
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite hidden-anchor sampled BPO score")
        pieces.append(loss)
        for name, value in metrics.items():
            diagnostics.setdefault(name, []).append(value)

    def restore(parts):
        result = (parts[0] if parts else hidden).new_zeros(
            cache.valid.shape, dtype=parts[0].dtype if parts else probability_dtype(hidden)
        )
        if parts:
            result[cache.valid] = torch.cat(parts)
        return result

    loss = restore(pieces)
    if not pieces:
        connected = hidden.nan_to_num().sum() + weight.nan_to_num().sum()
        if bias is not None:
            connected = connected + bias.nan_to_num().sum()
        loss = loss + connected * 0
        diagnostics = {name: [] for name in diagnostic_names(objective)}
    return LossResult(
        loss, {key: restore(value) for key, value in diagnostics.items()}, objective.loss_kind
    )


@torch.no_grad()
def build_hidden_anchor(hidden, weight, bias, actions, context, *, vocab_chunk):
    from mirroropd.core.cache import AnchorCache

    if context.anchor_storage != "hidden":
        raise ValueError("Hidden anchor requires its explicit storage context")
    if (
        actions.dtype != torch.long
        or actions.shape != (len(hidden),)
        or ((actions < 0) | (actions >= context.vocab_size)).any()
    ):
        raise ValueError("Hidden anchor actions must align with legal response predictors")
    if len(weight) != context.vocab_size or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("Hidden anchor head differs from the declared vocabulary/width")
    p = ProjectedDistribution(hidden, weight, bias, context.temperature, vocab_chunk)
    stats = p.statistics(actions, context.estimator.topj)
    support = p.statistics(actions, 0, stats["ids"])
    dtype = (
        torch.get_autocast_dtype(hidden.device.type)
        if torch.is_autocast_enabled(hidden.device.type)
        else hidden.dtype
    )
    return AnchorCache(
        context=context,
        valid=torch.ones(len(hidden), dtype=torch.bool, device=hidden.device),
        actions=actions.detach().clone(),
        head_ids=stats["ids"],
        anchor_head=stats["head"],
        anchor_tail=support["tail"],
        anchor_action=stats["action"],
        anchor_hidden=hidden.detach().to(dtype=dtype, copy=True),
        anchor_logz=stats["logz"],
    )


@torch.no_grad()
def score_hidden_teacher(
    hidden, weight, bias, anchor, frozen_weight, frozen_bias, *, diagnostics, vocab_chunk
):
    from mirroropd.core.cache import FrozenCache

    p = ProjectedDistribution(hidden, weight, bias, anchor.context.temperature, vocab_chunk)
    stats = p.statistics(
        anchor.actions, anchor.context.estimator.topj if diagnostics == "full" else 0
    )
    q = ProjectedDistribution(
        anchor.anchor_hidden, frozen_weight, frozen_bias, anchor.context.temperature, vocab_chunk
    )
    kt = center(q, p, anchor.anchor_logz, stats["logz"], anchor.context.signal)
    return FrozenCache(
        context=anchor.context,
        **anchor.tensors(),
        teacher_action=stats["action"],
        reward_center=kt,
        teacher_ids=stats["ids"] if diagnostics == "full" else None,
    )
