"""Refresh OneShot evidence attachments without rewriting the maintained report."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.analysis.oneshot_effective_time import (
    audit_completed_arm,
    audit_evaluation,
    collapse_statistics,
    history,
)
from training.oneshot_suite import selected_experiment_plan

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "runs/oneshot-effective-time-20261012"
OUT = ROOT / "results/oneshot-effective-time/consolidated"


def figure(fig, name):
    """Save shareable raster and vector figures from the same observations."""
    for ext in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, metadata=metadata)
    plt.close(fig)


def observations(config):
    """Read only committed, deduplicated updates from one arm."""
    return history(SUITE / config["id"] / "updates.jsonl", "update_before")


def full_rows(identifier):
    """Read full-vocabulary native readouts on the fixed 4096-prefix reservoir."""
    return history(SUITE / identifier / "fixed-native.jsonl", "update")


def distance(arm, step):
    """Use the post-final readout when the requested step is the terminal step."""
    return arm["distance"][arm["steps"].index(step)]


def draw(bf, fp_config, evaluations):
    """Plot complete learning-rate arms, stopped precision data and benchmarks."""
    colors = ("#2166ac", "#4d9221", "#b2182b")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm, color in zip(bf, colors, strict=True):
        for ax, field in zip(axes, ("steps", "effective_time"), strict=True):
            ax.plot(arm[field], arm["distance"], color=color, label=f"lr={arm['config']['lr']:g}")
            ax.set_yscale("log")
            ax.set_ylabel("Mean absolute action log-prob gap (395862 fixed actions)")
            ax.grid(alpha=0.2)
            ax.set_xlim(left=0)
    axes[0].set_xlabel("Completed optimizer updates")
    axes[1].set_xlabel("Effective time u = updates * lr / 1e-6")
    axes[0].legend()
    fig.suptitle("Fixed responses, seed 42: three completed BF16 runs")
    figure(fig, "frozen-effective-time")
    fp = observations(fp_config)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    stop = fp[-1]["update_before"]
    for identifier, label, color in (
        (bf[0]["config"]["id"], "BF16 training", "#2166ac"),
        (fp_config["id"], "FP32 training (stopped)", "#b2182b"),
    ):
        rows = history(SUITE / identifier / "updates.jsonl", "update_before")
        selected = [r for r in rows if r["update_before"] <= stop]
        axes[0].plot(
            [r["update_before"] for r in selected],
            [r["metrics"]["current_advantage_abs/mean"] for r in selected],
            label=label,
            color=color,
        )
        rows = [r for r in full_rows(identifier) if r["update"] <= stop]
        axes[1].plot(
            [r["update"] for r in rows],
            [r["metrics"]["student_teacher_full_kl"] for r in rows],
            label=label,
            color=color,
        )
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Completed optimizer updates")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Mean absolute gap (all fixed actions)")
    axes[1].set_ylabel("Student-to-teacher KL (4096 fixed prefixes)")
    fig.suptitle("Matched lr=5e-7: native training precision, common observed interval")
    figure(fig, "precision-comparison")
    tasks = ("MATH-500", "AMC23", "AIME25")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for arm, color in zip(bf, colors, strict=True):
        config = arm["config"]
        ids = ["initial-s42"] + [config["id"] + f"-update{t}" for t in config["checkpoint_steps"]]
        dose = [0] + [config["lr"] / 1e-6 * t for t in config["checkpoint_steps"]]
        for col, task in enumerate(tasks):
            for row, key in enumerate(("accuracy", "budget_exhausted_fraction")):
                axes[row, col].plot(
                    dose,
                    [100 * evaluations[i][task][key] for i in ids],
                    marker="o",
                    color=color,
                    label=f"lr={config['lr']:g}",
                )
    for col, task in enumerate(tasks):
        axes[0, col].axhline(
            100 * evaluations["teacher"][task]["accuracy"],
            color="gray",
            linestyle="--",
            label="Teacher",
        )
        axes[0, col].set_title(task)
        for row in range(2):
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_xlabel("Effective time u")
            axes[row, col].set_ylim(0, 100)
    axes[0, 0].set_ylabel("Greedy accuracy (%)")
    axes[1, 0].set_ylabel("7168-token budget exhausted (%)")
    axes[0, 0].legend(fontsize=7)
    figure(fig, "frozen-benchmarks")
    with (ROOT / "results/oneshot-inner-audit/inner-history.csv").open() as handle:
        inner = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for row, method in enumerate(("mdpo", "bpo")):
        for outer, color in zip((0, 4, 49), colors, strict=True):
            rows = [r for r in inner if r["method"] == method and int(r["outer_call"]) == outer]
            x = [int(r["inner_step"]) for r in rows]
            for col, field in enumerate(("anchor_student_tail_kl", "grad_norm")):
                axes[row, col].plot(
                    x,
                    [float(r[field]) for r in rows],
                    color=color,
                    marker=".",
                    label=f"outer={outer}",
                )
                axes[row, col].grid(alpha=0.2)
                axes[row, col].set_xlabel("Completed inner updates at measurement (pre-update)")
                axes[row, col].set_title(method.upper())
                axes[row, col].legend(fontsize=8)
        axes[row, 0].set_ylabel("Tail-aware anchor-to-student KL")
        axes[row, 1].set_ylabel("Gradient norm")
    figure(fig, "formal-inner-windows")


def main():
    """Refresh numerical attachments; editorial prose has one maintained source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-legacy-figures", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = selected_experiment_plan(SUITE)
    bf = [
        audit_completed_arm(SUITE, c)
        for c in plan
        if c["method"] == "opd" and c["rollout"] == "frozen" and c["lr"] > 0
    ]
    assert len(bf) == 3
    assert all(arm is not None for arm in bf)
    fp_config = next(
        c
        for c in json.loads((SUITE / "plan.json").read_text())
        if c["id"] == "frozen-opd-fp32-lr5em7-s42"
    )
    evaluations = {
        p.parent.name: audit_evaluation(p.parent)
        for p in (SUITE / "evaluation").glob("*/completion.json")
    }
    if args.refresh_legacy_figures:
        draw(bf, fp_config, evaluations)
    snapshot = {
        "scope": json.loads((SUITE / "execution-selection.json").read_text()),
        "bf16_time_collapse": collapse_statistics(bf),
        "formal_benchmarks": {
            k: v for k, v in evaluations.items() if k != "canary-bf16-s42-update4"
        },
        "fp32_partial_logged_updates": len(observations(fp_config)),
        "online_progress": json.loads(
            (SUITE / "online-opd-bf16-lr5em7-s42/progress.json").read_text()
        ),
        "controller_state": json.loads((SUITE / "status.json").read_text()),
        "completed_bf16_arms": [
            {
                "id": a["config"]["id"],
                "updates": a["config"]["updates"],
                "final_distance": a["distance"][-1],
            }
            for a in bf
        ],
    }
    (OUT / "report-snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(
        json.dumps(
            {"snapshot": str(OUT / "report-snapshot.json"), "editorial_report_modified": False}
        )
    )


if __name__ == "__main__":
    main()
