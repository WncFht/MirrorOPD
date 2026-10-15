"""Selected probabilities and stable tail reductions without full log-softmax copies."""

from dataclasses import dataclass

import torch


def probability_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


@dataclass(slots=True)
class Distribution:
    logits: torch.Tensor
    log_normalizer: torch.Tensor | None
    log_probabilities: torch.Tensor | None = None

    @classmethod
    def from_logits(cls, logits, *, materialize=False):
        values = logits.to(probability_dtype(logits))
        if materialize:
            return cls(values, None, values.log_softmax(-1))
        return cls(values, values.logsumexp(-1))

    def selected(self, ids):
        if self.log_probabilities is not None:
            return self.log_probabilities.gather(-1, ids)
        return self.logits.gather(-1, ids) - self.log_normalizer.unsqueeze(-1)

    def action(self, actions):
        return self.selected(actions.unsqueeze(-1)).squeeze(-1)

    def full(self):
        if self.log_probabilities is not None:
            return self.log_probabilities
        return self.logits - self.log_normalizer.unsqueeze(-1)

    @torch.no_grad()
    def tail(self, ids, chunk_size):
        result = self.logits.new_empty(self.logits.shape[:-1])
        for start in range(0, len(self.logits), chunk_size):
            stop = start + chunk_size
            values = self.logits if self.log_probabilities is None else self.log_probabilities
            block = values[start:stop].detach().clone()
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
        result = self.logits.new_empty(self.logits.shape[:-1])
        for start in range(0, len(self.logits), chunk_size):
            stop = start + chunk_size
            block = (
                self.log_probabilities[start:stop]
                if self.log_probabilities is not None
                else self.logits[start:stop] - self.log_normalizer[start:stop, None]
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
