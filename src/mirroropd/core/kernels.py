"""Bounded-workspace objective reductions with explicit frozen evidence.

The eager objectives remain the numerical reference. These functions only
change execution: full BPO is the same anchor-weighted centered residual;
sampled coefficients stay stopped and use the current student center.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mirroropd.core.cache import select_valid
from mirroropd.core.distributions import probability_dtype


class _FullBPO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, anchor, target, eta, token_chunk, vocab_chunk):
        dtype = probability_dtype(logits)
        mean = torch.zeros(len(logits), dtype=dtype, device=logits.device)
        loss = torch.zeros_like(mean)
        # Subtract a common logit first to keep large-offset inputs stable.
        origin = logits[:, 0].to(dtype)
        for begin in range(0, len(logits), token_chunk):
            stop = begin + token_chunk
            for start in range(0, logits.shape[-1], vocab_chunk):
                end = start + vocab_chunk
                q = anchor[begin:stop, start:end].exp()
                x = (
                    logits[begin:stop, start:end].to(dtype)
                    - origin[begin:stop, None]
                    - target[begin:stop, start:end]
                )
                mean[begin:stop] += (q * x).sum(-1)
            for start in range(0, logits.shape[-1], vocab_chunk):
                end = start + vocab_chunk
                q = anchor[begin:stop, start:end].exp()
                x = (
                    logits[begin:stop, start:end].to(dtype)
                    - origin[begin:stop, None]
                    - target[begin:stop, start:end]
                )
                loss[begin:stop] += (q * (x - mean[begin:stop, None]).square()).sum(-1) / (2 * eta)
        ctx.save_for_backward(logits, anchor, target, origin, mean)
        ctx.eta, ctx.token_chunk, ctx.vocab_chunk = eta, token_chunk, vocab_chunk
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        logits, anchor, target, origin, mean = ctx.saved_tensors
        gradient = torch.empty_like(logits)
        dtype = probability_dtype(logits)
        for begin in range(0, len(logits), ctx.token_chunk):
            stop = begin + ctx.token_chunk
            for start in range(0, logits.shape[-1], ctx.vocab_chunk):
                end = start + ctx.vocab_chunk
                x = (
                    logits[begin:stop, start:end].to(dtype)
                    - origin[begin:stop, None]
                    - target[begin:stop, start:end]
                )
                g = anchor[begin:stop, start:end].exp() * (x - mean[begin:stop, None]) / ctx.eta
                gradient[begin:stop, start:end] = g * upstream[begin:stop, None]
        return gradient, None, None, None, None, None


def full_bpo(logits, anchor, target, eta, *, token_chunk=64, vocab_chunk=4096):
    if anchor.requires_grad or target.requires_grad:
        raise ValueError("BPO anchor and target must be frozen")
    return _FullBPO.apply(logits, anchor, target, eta, token_chunk, vocab_chunk)


def projected_loss(
    hidden,
    weight,
    bias,
    cache,
    objective,
    *,
    anchor_version,
    token_chunk=64,
    vocab_chunk=4096,
    temperature=1.0,
):
    """Project only valid predictors; checkpoint each token tile including its loss.

    Frozen dense fields stay compact. Checkpointing discards logits and the
    objective's vocabulary-sized intermediates after each tile, and recomputes
    them for backward. This bounds projection workspace by token_chunk*V;
    full BPO additionally tiles its reductions over V. Model weight gradients
    and the explicitly budgeted frozen cache remain resident.
    """
    from mirroropd.core.objectives import LossResult, compute_loss

    cache.context.validate(objective, anchor_version=anchor_version)
    values = hidden.reshape(-1, hidden.shape[-1])[cache.valid.reshape(-1)]
    if (
        objective.algorithm == "bpo"
        and objective.estimator.integration == "full"
        and objective.diagnostics == "none"
        and len(values)
    ):
        from mirroropd.core.linear_bpo import linear_bpo

        loss = linear_bpo(
            values,
            weight,
            bias,
            cache.anchor_full,
            cache.target_full,
            objective.signal.eta,
            token_chunk=token_chunk,
            vocab_chunk=vocab_chunk,
            temperature=temperature,
        )
        output = loss.new_zeros(cache.valid.shape)
        output[cache.valid] = loss
        if not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite projected conditional BPO objective")
        return LossResult(output, {}, objective.loss_kind)
    pieces = []
    diagnostics = {}
    for start in range(0, len(values), token_chunk):
        current = values[start : start + token_chunk]
        frozen = select_valid(cache, start, start + token_chunk)

        def execute(h, w, b, frozen=frozen):
            logits = F.linear(h, w, b) / temperature
            # Core dispatch owns the target and diagnostics semantics.
            result = compute_loss(
                logits,
                frozen,
                objective,
                anchor_version=anchor_version,
                vocab_chunk_size=vocab_chunk,
            )
            names = tuple(sorted(result.diagnostics))
            return (result.loss, *[result.diagnostics[name] for name in names])

        # Names are declared by the objective; output tuple layout is fixed even
        # for an empty diagnostic policy and during checkpoint recomputation.
        from mirroropd.core.diagnostics import diagnostic_names

        names = tuple(sorted(diagnostic_names(objective)))
        output = (
            checkpoint(execute, current, weight, bias, use_reentrant=False)
            if torch.is_grad_enabled()
            else execute(current, weight, bias)
        )
        pieces.append(output[0])
        for name, value in zip(names, output[1:], strict=True):
            diagnostics.setdefault(name, []).append(value.detach())

    def restore(parts):
        result = (parts[0] if parts else hidden).new_zeros(cache.valid.shape)
        if parts:
            result[cache.valid] = torch.cat(parts)
        return result

    loss = restore(pieces)
    if not pieces:
        loss = loss + (hidden.sum() + weight.sum()) * 0
    return LossResult(
        loss, {name: restore(parts) for name, parts in diagnostics.items()}, objective.loss_kind
    )
