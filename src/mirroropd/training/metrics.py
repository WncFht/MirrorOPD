"""Merge token moments before deriving inner diagnostics, never micro means."""

import json
import math
from pathlib import Path

import torch


def token_moments(fields, mask):
    """Return (reduction, value) pairs usable without importing verl.

    All measurements are detached and describe the pre-update policy. SUM
    sufficient statistics permit unequal micro-batch lengths and DP ranks.
    Quantiles are deliberately absent: averages of quantiles are not quantiles.
    """
    mask = mask.bool()
    result = {"mirror/raw/count": ("sum", float(mask.sum()))}
    if not mask.any():
        return result
    for key, field in fields.items():
        if not key.startswith("mirror_"):
            continue
        if field.shape != mask.shape:
            raise ValueError(f"unaligned diagnostic: {key}")
        values = field.detach()[mask].double()
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"nonfinite diagnostic: {key}")
        name = key.removeprefix("mirror_")
        for suffix, reduction, value in (
            ("sum", "sum", values.sum()),
            ("sum_sq", "sum", values.square().sum()),
            ("min", "min", values.min()),
            ("max", "max", values.max()),
        ):
            result[f"mirror/raw/{name}/{suffix}"] = (reduction, float(value))
    return result


def finalize_token_moments(metrics, dp_size=1):
    """Consume verl's DP-averaged sums after micro/DP aggregation.

    verl Metric.aggregate_dp averages SUM across DP ranks. Ratios cancel
    that common factor; counts and ESS must multiply it back explicitly.
    """
    raw = {k: v for k, v in metrics.items() if k.startswith("mirror/raw/")}
    if not raw:
        return dict(metrics)
    result = {k: v for k, v in metrics.items() if k not in raw}
    count = raw["mirror/raw/count"]
    result["mirror/valid_tokens"] = count * dp_size
    if count == 0:
        return result
    names = {k.split("/")[2] for k in raw if k.endswith("/sum")}
    for name in names:
        prefix = f"mirror/raw/{name}"
        mean = raw[prefix + "/sum"] / count
        second = raw[prefix + "/sum_sq"] / count
        variance = max(0.0, second - mean * mean)
        result.update(
            {
                f"mirror/{name}/mean": mean,
                f"mirror/{name}/std": math.sqrt(variance),
                f"mirror/{name}/rms": math.sqrt(second),
                f"mirror/{name}/min": raw[prefix + "/min"],
                f"mirror/{name}/max": raw[prefix + "/max"],
            }
        )
    if "importance_ratio" in names:
        s = raw["mirror/raw/importance_ratio/sum"]
        ss = raw["mirror/raw/importance_ratio/sum_sq"]
        result["mirror/importance_ess"] = dp_size * s * s / ss if ss else 0.0
        result["mirror/importance_ess_fraction"] = s * s / (count * ss) if ss else 0.0
    return result


def append_inner_record(path, metrics, *, outer_call, inner_step, metadata=None):
    """One source worker writes one already globally reduced update record."""
    record = dict(metadata or {})
    record.update(outer_call=outer_call, inner_step=inner_step, measurement="pre-update")
    record["metrics"] = {
        k: float(v) for k, v in metrics.items() if k.startswith("mirror/") or k == "grad_norm"
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
