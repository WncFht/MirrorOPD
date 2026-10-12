"""Native Top-J set comparisons, independent of the optimization support."""

import torch


def topk_pair_overlap(
    left: torch.Tensor, right: torch.Tensor, pair: str
) -> dict[str, torch.Tensor]:
    """Compare equal-size unique Top-J sets, ignoring their ranking order.

    J=0 returns zero intersection fraction and Jaccard by convention. Ties
    retain torch.topk's native membership; ordering inside a set is irrelevant.
    Inputs are token IDs; the returned diagnostics never carry gradients.
    """
    if left.shape != right.shape:
        raise ValueError("Top-J set shapes must match")
    j = left.shape[-1]
    intersection = left.unsqueeze(-1).eq(right.unsqueeze(-2)).any(-1).float().sum(-1)
    union = 2 * j - intersection
    return {
        f"mirror_overlap_{pair}_intersection_count": intersection,
        f"mirror_overlap_{pair}_union_count": union,
        f"mirror_overlap_{pair}_intersection_fraction": intersection / j if j else intersection,
        f"mirror_overlap_{pair}_jaccard": intersection / union if j else intersection,
    }
