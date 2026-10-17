"""Selected probabilities and stable tail reductions without full log-softmax copies."""

from dataclasses import dataclass

import torch


def probability_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


@torch.no_grad()
def log_normalizer(logits, token_chunk=64, vocab_chunk=4096):
    """Stream FP32/FP64 reductions without an N-by-V upcast or saved softmax."""
    result = torch.empty(logits.shape[:-1], device=logits.device, dtype=probability_dtype(logits))
    for begin in range(0, len(logits), token_chunk):
        rows = logits[begin : begin + token_chunk]
        value = result.new_full(rows.shape[:-1], -torch.inf)
        for start in range(0, logits.shape[-1], vocab_chunk):
            part = rows[..., start : start + vocab_chunk].to(result.dtype).logsumexp(-1)
            value = torch.logaddexp(value, part)
        result[begin : begin + token_chunk] = value
    return result


class _SelectedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, ids, normalizer):
        ctx.save_for_backward(logits, ids, normalizer)
        return logits.gather(-1, ids).to(normalizer.dtype) - normalizer.unsqueeze(-1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, ids, normalizer = ctx.saved_tensors
        gradient = torch.empty_like(logits)
        for begin in range(0, len(logits), 64):
            stop = begin + 64
            index, upstream = ids[begin:stop], grad_output[begin:stop].to(normalizer.dtype)
            total = upstream.sum(-1, keepdim=True)
            for start in range(0, logits.shape[-1], 4096):
                end = min(start + 4096, logits.shape[-1])
                piece = logits[begin:stop, start:end].to(normalizer.dtype)
                piece = -(piece - normalizer[begin:stop, None]).exp() * total
                belongs = (index >= start) & (index < end)
                piece.scatter_add_(
                    -1, (index - start).clamp(0, end - start - 1), upstream * belongs
                )
                gradient[begin:stop, start:end] = piece
        return gradient, None, None


@dataclass(slots=True)
class Distribution:
    logits: torch.Tensor
    log_normalizer: torch.Tensor | None
    log_probabilities: torch.Tensor | None = None

    @classmethod
    def from_logits(cls, logits, *, materialize=False):
        if materialize:
            values = logits.to(probability_dtype(logits))
            return cls(values, None, values.log_softmax(-1))
        return cls(logits, log_normalizer(logits))

    def selected(self, ids):
        if self.log_probabilities is not None:
            return self.log_probabilities.gather(-1, ids)
        if torch.is_grad_enabled() and self.logits.requires_grad:
            return _SelectedLogProbs.apply(self.logits, ids, self.log_normalizer)
        return self.logits.gather(-1, ids).to(
            self.log_normalizer.dtype
        ) - self.log_normalizer.unsqueeze(-1)

    def action(self, actions):
        return self.selected(actions.unsqueeze(-1)).squeeze(-1)

    def full(self):
        if self.log_probabilities is not None:
            return self.log_probabilities
        if torch.is_grad_enabled() and self.logits.requires_grad:
            # A caller explicitly requesting all LPs needs the normalizer
            # derivative too. The streamed normalizer itself is detached.
            return self.logits.to(probability_dtype(self.logits)).log_softmax(-1)
        return self.logits.to(probability_dtype(self.logits)) - self.log_normalizer.unsqueeze(-1)

    @torch.no_grad()
    def tail(self, ids, chunk_size):
        result = torch.empty(
            self.logits.shape[:-1], device=self.logits.device, dtype=probability_dtype(self.logits)
        )
        for start in range(0, len(self.logits), chunk_size):
            stop = start + chunk_size
            values = self.logits if self.log_probabilities is None else self.log_probabilities
            block = values[start:stop].detach().to(result.dtype, copy=True)
            block.scatter_(-1, ids[start:stop], float("-inf"))
            reduced = block.logsumexp(-1)
            result[start:stop] = (
                reduced - self.log_normalizer[start:stop]
                if self.log_probabilities is None
                else reduced
            )
        return result

    @torch.no_grad()
    def entropy(self, chunk_size):
        result = torch.empty(
            self.logits.shape[:-1], device=self.logits.device, dtype=probability_dtype(self.logits)
        )
        for start in range(0, len(self.logits), chunk_size):
            stop = start + chunk_size
            block = (
                self.log_probabilities[start:stop]
                if self.log_probabilities is not None
                else self.logits[start:stop].to(result.dtype)
                - self.log_normalizer[start:stop, None]
            )
            result[start:stop] = -(block.exp() * block).sum(-1)
        return result


def grouped_kl(qhead, qtail, phead, ptail):
    result = (qhead.exp() * (qhead - phead)).sum(-1)
    nonempty = torch.isfinite(qtail)
    result[nonempty] += qtail[nonempty].exp() * (qtail[nonempty] - ptail[nonempty])
    return result


def support_overlap(left, right):
    j = left.shape[-1]
    if j == 0:
        return torch.zeros(left.shape[:-1], device=left.device)
    if j > 64:
        ordered = right.sort(-1).values.contiguous()
        locations = torch.searchsorted(ordered, left.contiguous()).clamp_max(j - 1)
        intersection = ordered.gather(-1, locations).eq(left).sum(-1)
    else:
        intersection = left.unsqueeze(-1).eq(right.unsqueeze(-2)).any(-1).sum(-1)
    return intersection / (2 * j - intersection)
