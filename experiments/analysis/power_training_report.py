"""Report cached-model comparisons now and full-parameter pilot outcomes when available."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

from mirroropd.proximal import minimax_velocity_shape, worst_fisher_speed_squared

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def history(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]


def relative_noise(repo):
    clusters = {}
    with gzip.open(repo / "data/power-mean-20261013/prefix-audit.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] != "fixed":
                continue
            key = (
                row["precision"],
                float(row["strength"]),
                float(row["kappa"]),
                int(row["response"]),
            )
            value = clusters.setdefault(key, np.zeros(2))
            value += np.array(
                [
                    float(row["conditional_logit_variance_mean_baseline"]),
                    float(row["squared_mean_logit_gradient"]),
                ]
            )
    result = {}
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0, 64, size=(2000, 64))
    for precision in ("bf16", "fp32"):
        for strength in (0.1, 0.3):
            baseline = np.stack([clusters[(precision, strength, 0.0, r)] for r in range(64)])
            for kappa in (0.5, 1.0):
                current = np.stack([clusters[(precision, strength, kappa, r)] for r in range(64)])
                c, b = current[draws].sum(1), baseline[draws].sum(1)
                bootstrap = (c[:, 0] / c[:, 1]) / (b[:, 0] / b[:, 1])
                ratio = (current[:, 0].sum() / current[:, 1].sum()) / (
                    baseline[:, 0].sum() / baseline[:, 1].sum()
                )
                result[f"{precision}-lambda{strength}-kappa{kappa}"] = {
                    "relative_logit_noise_ratio": float(ratio),
                    "paired_response_cluster_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                    "scope": "Ratio of averaged per-prefix covariance and squared mean logit gradient; not full-parameter batch SNR",
                }
    return result


def figures(output, cells):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    grid = np.linspace(0.08, 0.5, 200)
    optimum = minimax_velocity_shape()
    axes[0].plot(grid, [worst_fisher_speed_squared(k) for k in grid])
    axes[0].scatter([optimum], [worst_fisher_speed_squared(optimum)], color="#279681", zorder=3)
    axes[0].set(
        xlabel="Target shape",
        ylabel="Sharp worst Fisher speed squared",
        title=f"Target velocity criterion: optimum {optimum:.6f}",
    )
    axes[0].set_yscale("log")
    axes[1].plot(grid, 1 / grid, color="#279681")
    axes[1].scatter([0.5], [2], color="#279681", zorder=3)
    axes[1].set(
        xlabel="Inner fitting shape",
        ylabel="Sharp coefficient-noise / fitting-error constant",
        title="Different criterion: inner optimum 0.5",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    save(fig, output, "two-minimax-criteria")
    if cells:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for cell in cells:
            for group, ax in zip(("train", "heldout"), axes, strict=True):
                rows = cell[group]
                if rows:
                    ax.plot(
                        [r["update"] for r in rows],
                        [r["teacher_reverse_kl"] for r in rows],
                        label=cell["id"],
                    )
        for group, ax in zip(("train", "heldout"), axes, strict=True):
            ax.set(
                xlabel="Actual optimizer updates", ylabel="Native teacher reverse KL", title=group
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=5)
        save(fig, output, "training-teacher-kl")


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"{name}.{extension}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    plan = (
        json.loads((args.suite / "plan.json").read_text())
        if (args.suite / "plan.json").exists()
        else []
    )
    cells = []
    for config in plan:
        path = args.suite / config["id"]
        if not (path / "train-native.jsonl").exists():
            continue
        cells.append(
            {
                "id": config["id"],
                "config": config,
                "train": history(path / "train-native.jsonl"),
                "heldout": history(path / "heldout-native.jsonl"),
                "completed": (path / "completion.json").exists(),
            }
        )
    probes = {}
    for path in args.suite.glob("power-canary-*/gradient-probe.json"):
        probes[path.parent.name] = json.loads(path.read_text())
    k = minimax_velocity_shape()
    summary = {
        "cached_relative_noise": relative_noise(repo),
        "training_cells": cells,
        "parameter_gradient_probes": probes,
        "minimax_velocity_shape": k,
        "sharp_worst_fisher_speed_squared": worst_fisher_speed_squared(k),
        "completed_training_cells": sum(c["completed"] for c in cells),
        "planned_training_cells": len(plan),
        "scope": "Cached model audit and training pilot are separate evidence levels; no completed GPU training is inferred from cached distributions",
    }
    (args.output / "training-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures(args.output, cells)
    print(
        json.dumps(
            {
                "completed": summary["completed_training_cells"],
                "planned": len(plan),
                "relative_logit_noise": summary["cached_relative_noise"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
