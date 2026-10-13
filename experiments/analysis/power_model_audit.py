"""Exact conditional full-vocabulary diagnostics from the real 1.5B model cache."""

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

from mirroropd.proximal import power_log_target, power_signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHAPES = (0.0, 0.25, 0.5, 0.75, 1.0)
PRECISIONS = ("bf16", "fp32")


def normalized(logp):
    logp = logp.double()
    return logp - logp.logsumexp(-1, keepdim=True)


def radius(qlog, tlog, kappa, strength):
    target = power_log_target(qlog, tlog, kappa, strength)
    affinity = torch.exp((qlog + target) / 2).sum(-1).clamp(0, 1)
    return 4 * affinity.acos().square()


def calibrate(cache):
    """Match one scalar interpolation strength across shapes on an independent panel."""
    manifest = json.loads((cache / "manifest.json").read_text())
    chosen = set(
        torch.randperm(
            manifest["diagnostic_tokens"], generator=torch.Generator().manual_seed(4242)
        )[:128].tolist()
    )
    panels = {p: [] for p in PRECISIONS}
    teachers = []
    position = 0
    for rank in range(4):
        artifact = torch.load(cache / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
        for sequence in artifact["sequences"]:
            indices = [i for i in range(len(sequence["selected"])) if position + i in chosen]
            position += len(sequence["selected"])
            if indices:
                for precision in PRECISIONS:
                    panels[precision].append(sequence["q_logp_" + precision][indices].clone())
                teachers.append(sequence["teacher_logp"][indices].clone())
        del artifact
        gc.collect()
    tlog = normalized(torch.cat(teachers))
    result = {}
    for precision in PRECISIONS:
        qlog = normalized(torch.cat(panels[precision]))
        reference = float(radius(qlog, tlog, 0, 0.1).mean())
        strengths = {"0.0": 0.1}
        for kappa in SHAPES[1:]:
            left, right = 0.0, 1.0
            for _ in range(22):
                middle = (left + right) / 2
                if float(radius(qlog, tlog, kappa, middle).mean()) < reference:
                    left = middle
                else:
                    right = middle
            strengths[str(kappa)] = (left + right) / 2
        result[precision] = {
            "calibration_prefixes": len(qlog),
            "mean_squared_fisher_rao_radius": reference,
            "matched_strengths": strengths,
        }
    return result


def prefix_metrics(qlog, tlog, kappa, strength):
    """Separate action-conditional covariance from variability across prefixes."""
    q, teacher = qlog.exp(), tlog.exp()
    r = tlog - qlog
    f = power_signal(r, kappa, strength) / strength
    mean = (q * f).sum(-1)
    centered = f - mean.unsqueeze(-1)
    sum_q2 = q.square().sum(-1)
    score_norm2 = (1 - 2 * q + sum_q2.unsqueeze(-1)).clamp_min(0)
    mean_gradient_norm2 = (q * centered).square().sum(-1)
    mean_second = (q * centered.square() * score_norm2).sum(-1)
    norm_mean = (q * score_norm2).sum(-1)
    optimum = (q * f * score_norm2).sum(-1) / norm_mean.clamp_min(1e-300)
    optimal_second = (q * (f - optimum.unsqueeze(-1)).square() * score_norm2).sum(-1)
    target = power_log_target(qlog, tlog, kappa, strength)
    m = target.exp()
    w = r if kappa == 0 else torch.expm1(kappa * r) / kappa
    rmean, wmean = (q * r).sum(-1), (q * w).sum(-1)
    rc, wc = r - rmean.unsqueeze(-1), w - wmean.unsqueeze(-1)
    speed = (q * wc.square()).sum(-1)
    cosine = (q * rc * wc).sum(-1) / ((q * rc.square()).sum(-1) * speed).sqrt().clamp_min(1e-300)
    variance = mean_second - mean_gradient_norm2
    optimal_variance = optimal_second - mean_gradient_norm2
    if torch.any(variance < -1e-9) or torch.any(optimal_variance < -1e-9):
        raise RuntimeError("A conditional covariance trace is numerically negative")
    return {
        "signal_second_moment_unit_slope": (q * f.square()).sum(-1),
        "conditional_logit_variance_mean_baseline": variance.clamp_min(0),
        "conditional_logit_variance_optimal_baseline": optimal_variance.clamp_min(0),
        "squared_mean_logit_gradient": mean_gradient_norm2,
        "native_teacher_reverse_kl": (q * (qlog - tlog)).sum(-1),
        "target_teacher_reverse_kl": (m * (target - tlog)).sum(-1),
        "target_teacher_forward_kl": (teacher * (tlog - target)).sum(-1),
        "fisher_radius_squared": radius(qlog, tlog, kappa, strength),
        "natural_gradient_fisher_speed_squared": speed,
        "natural_direction_cosine_to_reverse_kl": cosine,
        "optimal_baseline": optimum,
    }


def cluster_ci(clusters, field, numerator, denominator):
    """Bootstrap whole responses; weighting retains the prefix sampling measure."""
    keys = sorted({key[0] for key in clusters if key[1] == numerator})
    n = np.array([clusters[(key, numerator)][field] for key in keys])
    d = np.array([clusters[(key, denominator)][field] for key in keys])
    rng = np.random.default_rng(20261013)
    resampled = rng.integers(0, len(keys), size=(2000, len(keys)))
    ratio = n[resampled].sum(-1) / d[resampled].sum(-1)
    return {
        "ratio": float(n.sum() / d.sum()),
        "response_cluster_ci95": np.quantile(ratio, [0.025, 0.975]).tolist(),
        "responses": len(keys),
    }


def figures(output, aggregate):
    colors = {0.0: "#335cba", 0.25: "#8d5bbb", 0.5: "#279681", 0.75: "#b54472", 1.0: "#db7434"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for precision, ax in zip(PRECISIONS, axes, strict=True):
        for strength, style in ((0.1, "-"), (0.3, "--")):
            cells = [aggregate[(precision, "fixed", k, strength)] for k in SHAPES]
            reference = cells[0]["conditional_logit_variance_mean_baseline"]
            ax.plot(
                SHAPES,
                [c["conditional_logit_variance_mean_baseline"] / reference for c in cells],
                marker="o",
                ls=style,
                label=f"strength={strength:g}",
            )
        ax.axhline(1, color="gray", ls=":")
        ax.set(
            xlabel="Power shape kappa",
            ylabel="Conditional covariance trace / geometric",
            title=f"{precision.upper()} anchor: unit local signal slope",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"cached-variance.{extension}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None,
        )
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for precision, ax in zip(PRECISIONS, axes, strict=True):
        for kappa in SHAPES:
            cells = [
                v
                for (p, mode, k, _), v in aggregate.items()
                if p == precision and mode == "matched" and k == kappa
            ]
            ax.scatter(
                cells[0]["fisher_radius_squared"],
                cells[0]["conditional_logit_variance_mean_baseline"],
                color=colors[kappa],
                s=55,
                label=f"kappa={kappa:g}",
            )
        ax.set(
            xlabel="Mean squared Fisher-Rao target displacement",
            ylabel="Unit-slope conditional covariance trace",
            title=f"{precision.upper()}: matched-radius controls",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"cached-matched-radius.{extension}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(args.cache)
    (args.output / "radius-calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps({"stage": "calibrated", "calibration": calibration}), flush=True)
    sums, counts, clusters = {}, {}, {}
    total_prefixes = 0
    header = None
    with gzip.open(args.raw, "wt", newline="") as stream:
        writer = None
        for rank in range(4):
            artifact = torch.load(
                args.cache / f"rank-{rank}.pt", map_location="cpu", weights_only=False
            )
            for sequence_index, sequence in enumerate(artifact["sequences"]):
                response = rank * 16 + sequence_index
                n = len(sequence["selected"])
                total_prefixes += n
                for precision in PRECISIONS:
                    cases = [("fixed", k, s) for k in SHAPES for s in (0.1, 0.3)] + [
                        ("matched", k, calibration[precision]["matched_strengths"][str(k)])
                        for k in SHAPES
                    ]
                    for start in range(0, n, 16):
                        stop = min(start + 16, n)
                        qlog = normalized(sequence["q_logp_" + precision][start:stop])
                        tlog = normalized(sequence["teacher_logp"][start:stop])
                        for mode, kappa, strength in cases:
                            key = (precision, mode, kappa, strength)
                            values = prefix_metrics(qlog, tlog, kappa, strength)
                            if kappa > 0:
                                exponent = min(2, 1 / kappa)
                                assert torch.isfinite(
                                    values["signal_second_moment_unit_slope"]
                                ).all()
                                if kappa <= 0.5:
                                    bound = (math.log1p(-strength) ** 2 + strength**2) / (
                                        kappa**2 * strength**2
                                    )
                                    assert torch.all(
                                        values["signal_second_moment_unit_slope"] <= bound + 1e-8
                                    )
                                assert exponent > 0
                            if writer is None:
                                header = [
                                    "response",
                                    "position",
                                    "precision",
                                    "mode",
                                    "kappa",
                                    "strength",
                                    *values,
                                ]
                                writer = csv.DictWriter(stream, fieldnames=header)
                                writer.writeheader()
                            columns = {name: tensor.tolist() for name, tensor in values.items()}
                            for i in range(stop - start):
                                writer.writerow(
                                    {
                                        "response": response,
                                        "position": int(sequence["selected"][start + i]),
                                        "precision": precision,
                                        "mode": mode,
                                        "kappa": kappa,
                                        "strength": strength,
                                        **{name: entries[i] for name, entries in columns.items()},
                                    }
                                )
                            cell = sums.setdefault(key, dict.fromkeys(values, 0.0))
                            cluster = clusters.setdefault(
                                (response, key), dict.fromkeys(values, 0.0)
                            )
                            for name, tensor in values.items():
                                value = float(tensor.sum())
                                cell[name] += value
                                cluster[name] += value
                            counts[key] = counts.get(key, 0) + stop - start
                print(
                    json.dumps({"rank": rank, "response": response, "prefixes": total_prefixes}),
                    flush=True,
                )
            del artifact
            gc.collect()
    assert total_prefixes == 4096
    aggregate = {
        key: {name: value / counts[key] for name, value in cell.items()}
        for key, cell in sums.items()
    }
    comparisons = {}
    for precision in PRECISIONS:
        for strength in (0.1, 0.3):
            for kappa in (0.5, 1.0):
                comparisons[f"{precision}-lambda{strength}-kappa{kappa}"] = cluster_ci(
                    clusters,
                    "conditional_logit_variance_mean_baseline",
                    (precision, "fixed", kappa, strength),
                    (precision, "fixed", 0.0, strength),
                )
    records = [
        {
            "precision": key[0],
            "mode": key[1],
            "kappa": key[2],
            "strength": key[3],
            "prefixes": counts[key],
            **cell,
        }
        for key, cell in aggregate.items()
    ]
    summary = {
        "scope": "Actual 1.5B frozen model distributions; exact conditional logit covariance, not full parameter-gradient variance",
        "prefixes": total_prefixes,
        "responses": 64,
        "prompts": 1,
        "calibration": calibration,
        "cells": records,
        "comparisons": comparisons,
        "cache_manifest_sha256": hashlib.sha256(
            (args.cache / "manifest.json").read_bytes()
        ).hexdigest(),
        "raw_gzip_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "cached-audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures(args.output, aggregate)
    print(json.dumps({"stage": "completed", "comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
