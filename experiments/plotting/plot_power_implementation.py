"""Plot saved power-mirror correctness and fitting measurements.

Semantic ledger: error curves are prefix-weighted means on the same 256
prefixes; training curves share one estimator color across both algorithms.
Geometry ledger: J is a logarithmic head count; update is an optimizer count;
normalized loss is dimensionless. Memory is the reported GPU-0 allocation.
Every plotted value comes from archived JSON/CSV; no fitted synthetic curves.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

from experiments.plotting.vendor import fvstyle as fv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, output, name, caption):
    fig.text(0.5, 0.015, caption, ha="center", fontsize=7, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    problems = fv.audit(fig)
    (output / f"{name}-audit.json").write_text(json.dumps(problems, indent=2, default=str) + "\n")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/power-implementation-20261015"
    summary = json.loads((output / "summary.json").read_text())
    fv.apply_rc("paper")
    colors = fv.series_colors("paper", 3)
    if summary.get("cpu_gradients"):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9), sharey=True)
        fig.subplots_adjust(bottom=0.25, top=0.88, left=0.12, right=0.97, wspace=0.35)
        for ax, row in zip(axes, summary["cpu_gradients"], strict=True):
            ax.plot(range(3), row["production"], marker="o", color=colors[0], label="Production")
            ax.plot(
                range(3),
                row["reference"],
                linestyle="none",
                marker="x",
                color=colors[1],
                label="Independent target",
            )
            ax.set(
                xlabel="Action id",
                ylabel="Gradient component",
                title=row["method"].upper(),
                xticks=[0, 1, 2],
            )
            ax.legend(fontsize=7)
        save(fig, output, "cpu-gradients", "FP64 independent formula check; kappa=0.5, eta=0.3")
    rows = [row for row in summary["audit"] if row["method"] == "bpo" and row["topj"] > 0]
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        fig.subplots_adjust(bottom=0.22, top=0.91, left=0.15, right=0.97)
        for color, field, label in zip(
            colors,
            ("teacher_center_mae", "student_center_mae", "score_coefficient_mae"),
            ("Teacher center", "Student center", "BPO coefficient"),
            strict=True,
        ):
            ax.loglog(
                [row["topj"] for row in rows],
                [row[field] for row in rows],
                marker="o",
                color=color,
                label=label,
            )
        ax.set(
            xlabel="Retained head size J",
            ylabel="Mean absolute error",
            title="Complete-head consistency on actual model caches",
        )
        ax.set_xticks([16, 64, 256, 151665], ["16", "64", "256", "V"])
        ax.set_yticks([1e-2, 1e-5, 1e-8, 1e-11, 1e-14, 1e-17])
        ax.legend(fontsize=8)
        save(
            fig,
            output,
            "center-errors",
            "64 responses, 256 fixed prefixes; FP64; kappa=0.5, eta=0.3",
        )
    all_arms = {arm["id"]: arm for arm in summary["training"]}
    if "bpo-full-lifecycle" in all_arms:
        directory = root / "data/power-implementation-20261015/bpo-full-lifecycle"
        measurements = [
            json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()
        ]
        refresh = all_arms["bpo-full-lifecycle"]["anchor_refresh"]
        points = sorted(
            [(r["update"], 0, r) for r in measurements] + [(r["update"], 1, r) for r in refresh],
            key=lambda row: row[:2],
        )
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        fig.subplots_adjust(bottom=0.24, top=0.89, left=0.15, right=0.97)
        for color, group in zip(colors, ("train", "heldout"), strict=False):
            ax.plot(
                [p[0] for p in points],
                [p[2][group + "/anchor_student_kl"] for p in points],
                marker="o",
                color=color,
                label=group,
            )
        ax.axvline(2, color=fv.NEUTRALS["baseline"], linestyle="--", label="Anchor refresh")
        ax.set(
            xlabel="Optimizer updates",
            ylabel="Anchor-student KL",
            title="Frozen-prefix anchor lifecycle",
        )
        ax.legend(fontsize=7)
        save(
            fig,
            output,
            "anchor-lifecycle",
            "Refresh after update 2; checkpoint and next-update parameter errors are zero",
        )
    if any("parameter-gradient" in name for name in all_arms):
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        fig.subplots_adjust(bottom=0.24, top=0.89, left=0.15, right=0.97)
        for color, precision in zip(colors, ("bf16", "fp32"), strict=False):
            values = []
            for method in ("bpo", "mdpo"):
                name = f"{method}-parameter-gradient" + ("-fp32" if precision == "fp32" else "")
                values.extend(
                    [
                        r["relative_l2_error"]
                        for r in all_arms.get(name, {}).get("gradient_checks", [])
                    ]
                )
            if len(values) == 4:
                ax.semilogy(range(4), values, marker="o", color=color, label=precision.upper())
        ax.set_xticks(
            range(4), ["BPO initial", "BPO updated", "MDPO initial", "MDPO updated"], fontsize=7
        )
        ax.set(
            ylabel="Relative parameter gradient error",
            xlabel="Fixed-parameter comparison",
            title="Independent-target gradient checks",
        )
        ax.legend(fontsize=7)
        save(
            fig,
            output,
            "parameter-gradients",
            "Each precision uses its native anchor; the same fixed response prefixes",
        )
    variants = ("topj16", "full-center", "full-actions")
    arms = {arm["id"]: arm for arm in summary["training"] if arm["id"].startswith("train-")}
    if not arms:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.18, top=0.92, left=0.12, right=0.98, hspace=0.65, wspace=0.35)
    for row_index, method in enumerate(("bpo", "mdpo")):
        for column, group in enumerate(("train", "heldout")):
            ax = axes[row_index, column]
            for color, variant in zip(colors, variants, strict=True):
                name = f"train-{method}-{variant}"
                if name not in arms:
                    continue
                path = root / "data/power-implementation-20261015" / name / "measurements.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines()]
                field = group + ("/loss" if method == "bpo" else "/target_reverse_kl")
                ax.plot(
                    [r["update"] for r in records],
                    [r[field] / records[0][field] for r in records],
                    marker="o",
                    color=color,
                    label=variant,
                )
            ax.axhline(1, linestyle="--", color=fv.NEUTRALS["baseline"], linewidth=0.8)
            ax.set(
                xlabel="Optimizer updates",
                ylabel="Loss / initial loss",
                title=f"{method.upper()}: {group}",
            )
    axes[0, 0].legend(fontsize=7)
    save(
        fig,
        output,
        "training-comparison",
        "Same initialization and fixed prefixes; single seed; no hyperparameter selection",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    fig.subplots_adjust(bottom=0.22, top=0.92, left=0.22, right=0.98, wspace=0.4)
    names = sorted(arms)
    labels = [name.removeprefix("train-").replace("full-", "F-") for name in names]
    for ax, field in zip(axes, ("time", "memory"), strict=True):
        values = [
            arms[name]["seconds"] / arms[name]["updates"]
            if field == "time"
            else arms[name]["peak_allocated_gib"]
            for name in names
        ]
        bars = ax.barh(
            np.arange(len(names)),
            values,
            color=fv.NEUTRALS["baseline"],
        )
        for patch in bars:
            fv.mark_neutral(patch)
        ax.set_yticks(np.arange(len(names)), labels, fontsize=7)
        ax.set(
            xlabel="Seconds / update" if field == "time" else "GPU-0 allocated GiB",
            ylabel="Estimator",
        )
    save(
        fig,
        output,
        "training-cost",
        "Time includes periodic readouts; model loading excluded; memory is GPU-0 peak",
    )


if __name__ == "__main__":
    main()
