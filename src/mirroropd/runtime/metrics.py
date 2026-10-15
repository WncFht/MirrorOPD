"""Token-weighted sufficient statistics with one packed device-to-host copy."""

import math

import torch

from mirroropd.runtime.events import environment_writer

METRIC_DEFINITIONS = {
    "reward": "Normalized power reward A(T/q) on the recorded rollout action",
    "frozen_reward_center": "Frozen -E_q A(T/q), exact or partition-estimated; not ordinary KL",
    "on_policy_baseline": "Action-independent frozen reward baseline used by fresh on-policy",
    "student_anchor_kl": "KL(q||p), exact or partition-estimated",
    "action_nll": "-log p(a) on the recorded action",
    "score_coefficient": "Stopped coefficient multiplying log p(a) for gradient descent",
    "log_anchor_ratio": "log p(a)-log q(a)",
    "importance_ratio": "p(a)/q(a), used by the sampled MDPO gradient",
    "conditional_objective": "Full conditional BPO objective or target KL/eta; algorithm in event context",
    "student_entropy": "Complete conditional entropy of p",
    "anchor_head_mass": "Frozen anchor probability of its selected support",
    "anchor_tail_mass": "Frozen anchor probability outside its selected support",
    "student_head_mass": "Current student probability on the frozen support",
    "student_tail_mass": "Current student probability outside the frozen support",
    "student_anchor_support_jaccard": "Native student/anchor Top-J set Jaccard",
    "student_teacher_support_jaccard": "Native student/teacher Top-J set Jaccard",
    "anchor_teacher_support_jaccard": "Native anchor/teacher Top-J set Jaccard",
}


def token_moments(diagnostics, mask):
    mask = mask.bool()
    names = sorted(diagnostics)
    unknown = set(names) - METRIC_DEFINITIONS.keys()
    if unknown:
        raise ValueError(f"Unregistered diagnostic names: {sorted(unknown)}")
    for name in names:
        if diagnostics[name].shape != mask.shape:
            raise ValueError(f"Diagnostic {name} does not align with valid tokens")
    values = (
        torch.stack([diagnostics[name].detach().reshape(-1) for name in names]).double()[
            :, mask.reshape(-1)
        ]
        if names
        else None
    )
    count = mask.sum().double()
    if values is not None and values.shape[-1]:
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite distillation diagnostics")
        stats = torch.stack(
            (values.sum(-1), values.square().sum(-1), values.amin(-1), values.amax(-1)), -1
        )
        packed = torch.cat((stats.reshape(-1), count.reshape(1))).cpu().tolist()
        result = {"distill/raw/count": ("sum", packed[-1])}
        for index, name in enumerate(names):
            for column, stat, reduction in (
                (0, "sum", "sum"),
                (1, "sum_sq", "sum"),
                (2, "min", "min"),
                (3, "max", "max"),
            ):
                result[f"distill/raw/{name}/{stat}"] = (reduction, packed[index * 4 + column])
        return result
    result = {"distill/raw/count": ("sum", count.cpu().item())}
    for name in names:
        for stat, reduction, value in (
            ("sum", "sum", 0.0),
            ("sum_sq", "sum", 0.0),
            ("min", "min", float("inf")),
            ("max", "max", float("-inf")),
        ):
            result[f"distill/raw/{name}/{stat}"] = (reduction, value)
    return result


def finalize_moments(metrics, dp_size):
    """verl DP-averages SUM statistics; undo this factor only for counts and ESS."""
    raw = {key: value for key, value in metrics.items() if key.startswith("distill/raw/")}
    if not raw:
        return dict(metrics)
    if dp_size < 1:
        raise ValueError("dp_size must be positive")
    result = {key: value for key, value in metrics.items() if key not in raw}
    count = raw["distill/raw/count"]
    result["distill/valid_tokens"] = count * dp_size
    if count == 0:
        return result
    names = {
        key.removeprefix("distill/raw/").removesuffix("/sum") for key in raw if key.endswith("/sum")
    }
    for name in names:
        prefix = "distill/raw/" + name
        mean = raw[prefix + "/sum"] / count
        second = raw[prefix + "/sum_sq"] / count
        variance = max(0.0, second - mean * mean)
        result.update(
            {
                f"distill/{name}/mean": mean,
                f"distill/{name}/std": math.sqrt(variance),
                f"distill/{name}/rms": math.sqrt(second),
                f"distill/{name}/min": raw[prefix + "/min"],
                f"distill/{name}/max": raw[prefix + "/max"],
            }
        )
    if "importance_ratio" in names:
        total, squares = (
            raw["distill/raw/importance_ratio/sum"],
            raw["distill/raw/importance_ratio/sum_sq"],
        )
        result["distill/importance_ess"] = dp_size * total * total / squares if squares else 0.0
        result["distill/importance_ess_fraction"] = (
            total * total / (count * squares) if squares else 0.0
        )
    return result


def record_update(
    metrics, *, outer, inner, epochs, algorithm, loss_kind, clip_grad, global_rank=0, writer=None
):
    if global_rank != 0:
        return None
    writer = writer or environment_writer("worker-0")
    if writer is None:
        return None
    values = {key: float(value) for key, value in metrics.items() if key.startswith("distill/")}
    if "grad_norm" in metrics:
        norm = float(metrics["grad_norm"])
        values.update(
            {
                "optimization/gradient_norm": norm,
                "optimization/clip_scale": min(1.0, clip_grad / max(norm, 1e-12))
                if clip_grad > 0
                else 1.0,
                "optimization/was_clipped": float(clip_grad > 0 and norm > clip_grad),
            }
        )
    return writer.emit(
        "optimization.step",
        outer=outer,
        inner=inner,
        update=outer * epochs + inner,
        updates_completed=outer * epochs + inner + 1,
        measurement="pre_update",
        algorithm=algorithm,
        loss_kind=loss_kind,
        metrics=values,
    )
