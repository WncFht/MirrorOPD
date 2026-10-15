"""Full conditional BPO projected in token/vocabulary tiles.

No student softmax or N-by-V logits is materialized. The frozen anchor and
normalized target remain the exact dense evidence; changing their storage or
center approximation is not part of this kernel. Weight gradients have their
ordinary V-by-H parameter size and are not counted as bounded workspace.
"""

import torch
import torch.nn.functional as F

from mirroropd.core.distributions import probability_dtype


class _LinearBPO(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, hidden, weight, bias, anchor, target, eta, token_chunk, vocab_chunk, temperature
    ):
        dtype = probability_dtype(hidden)
        mean = torch.zeros(len(hidden), device=hidden.device, dtype=dtype)
        origin = torch.zeros_like(mean)
        loss = torch.zeros_like(mean)
        for begin in range(0, len(hidden), token_chunk):
            stop = begin + token_chunk
            h = hidden[begin:stop]
            origin[begin:stop] = (
                (F.linear(h, weight[:1], bias[:1] if bias is not None else None) / temperature)
                .squeeze(-1)
                .to(dtype)
            )
            for start in range(0, len(weight), vocab_chunk):
                end = start + vocab_chunk
                z = (
                    F.linear(h, weight[start:end], bias[start:end] if bias is not None else None)
                    / temperature
                )
                x = z.to(dtype) - origin[begin:stop, None] - target[begin:stop, start:end]
                mean[begin:stop] += (anchor[begin:stop, start:end].exp() * x).sum(-1)
            for start in range(0, len(weight), vocab_chunk):
                end = start + vocab_chunk
                z = (
                    F.linear(h, weight[start:end], bias[start:end] if bias is not None else None)
                    / temperature
                )
                residual = (
                    z.to(dtype)
                    - origin[begin:stop, None]
                    - target[begin:stop, start:end]
                    - mean[begin:stop, None]
                )
                loss[begin:stop] += (anchor[begin:stop, start:end].exp() * residual.square()).sum(
                    -1
                ) / (2 * eta)
        ctx.save_for_backward(hidden, weight, bias, anchor, target, origin, mean)
        ctx.eta, ctx.token_chunk, ctx.vocab_chunk, ctx.temperature = (
            eta,
            token_chunk,
            vocab_chunk,
            temperature,
        )
        ctx.autocast = torch.is_autocast_enabled(hidden.device.type)
        ctx.projection_dtype = (
            torch.get_autocast_dtype(hidden.device.type) if ctx.autocast else hidden.dtype
        )
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        h, w, b, q, target, origin, mean = ctx.saved_tensors
        dtype = probability_dtype(h)
        dh = torch.zeros_like(h, dtype=dtype) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(w, dtype=probability_dtype(w)) if ctx.needs_input_grad[1] else None
        db = (
            torch.zeros_like(b, dtype=probability_dtype(b))
            if b is not None and ctx.needs_input_grad[2]
            else None
        )
        for begin in range(0, len(h), ctx.token_chunk):
            stop = begin + ctx.token_chunk
            for start in range(0, len(w), ctx.vocab_chunk):
                end = start + ctx.vocab_chunk
                with torch.autocast(
                    h.device.type, enabled=ctx.autocast, dtype=ctx.projection_dtype
                ):
                    z = (
                        F.linear(
                            h[begin:stop], w[start:end], b[start:end] if b is not None else None
                        )
                        / ctx.temperature
                    )
                residual = (
                    z.to(dtype)
                    - origin[begin:stop, None]
                    - target[begin:stop, start:end]
                    - mean[begin:stop, None]
                )
                grad = (
                    q[begin:stop, start:end].exp()
                    * residual
                    * upstream[begin:stop, None]
                    / (ctx.eta * ctx.temperature)
                )
                # Match the projection's gradient cast, then accumulate tile
                # products in probability precision rather than rounding each
                # partial hidden gradient back into BF16.
                g = grad.to(ctx.projection_dtype).to(dtype)
                with torch.autocast(h.device.type, enabled=False):
                    if dh is not None:
                        dh[begin:stop] += g @ w[start:end].to(ctx.projection_dtype).to(dtype)
                    if dw is not None:
                        dw[start:end] += g.T @ h[begin:stop].to(ctx.projection_dtype).to(dtype)
                    if db is not None:
                        db[start:end] += g.sum(0)
        return (
            dh.to(h.dtype) if dh is not None else None,
            dw.to(w.dtype) if dw is not None else None,
            db.to(b.dtype) if db is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def linear_bpo(hidden, weight, bias, anchor, target, eta, *, token_chunk, vocab_chunk, temperature):
    if anchor.requires_grad or target.requires_grad:
        raise ValueError("BPO evidence must be frozen")
    return _LinearBPO.apply(
        hidden, weight, bias, anchor, target, eta, token_chunk, vocab_chunk, temperature
    )
