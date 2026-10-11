"""Compare verified math runs using existing training summaries and raw logs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from experiments.analysis.gsm8k_training_report import read_rows, report

BENCHMARKS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}


def summarize(run, output, verification_root):
    verification = json.loads((verification_root / f"{run.name}-verification.json").read_text())
    if not verification.get("passed") or verification["evaluation_examples"] != 570:
        raise ValueError(f"Run lacks verified full-suite evidence: {run.name}")
    summary = report(
        run,
        output / run.name,
        expected_test_examples=570,
        benchmark_label="MATH-500/AMC23/AIME25",
    )
    summary["method"] = verification["method"]
    summary["wandb"] = verification["wandb"]
    inner = read_rows(run / "inner.jsonl")
    frozen = [row["metrics"] for row in inner if row["inner_step"] == 0]
    training = [
        row["data"] for row in read_rows(run / "outer.jsonl") if "timing_s/step" in row["data"]
    ]
    window = min(10, len(frozen))
    if len(training) != len(frozen):
        raise ValueError("Training outer records must match frozen rollout snapshots")
    # J>1 repeats frozen teacher/reward values within an outer. Compare outer
    # windows, so the first ten inner rows cannot stand in for ten rollouts.
    summary["outer_window_comparison"] = {"window_outer_count": window}
    for label, rows, key in (
        ("teacher_center_mean", frozen, "mirror/teacher_center/mean"),
        ("sampled_reward_mean", frozen, "mirror/sampled_reward/mean"),
        ("verifier_score_mean", training, "critic/score/mean"),
        ("training_budget_exhausted_mean", training, "response_length/clip_ratio"),
    ):
        summary["outer_window_comparison"][label] = {
            "first": fmean(row[key] for row in rows[:window]),
            "last": fmean(row[key] for row in rows[-window:]),
        }
    summary["benchmarks"] = {}
    for benchmark, examples in BENCHMARKS.items():
        accuracy = f"val-core/{benchmark}/acc/mean@1"
        exhausted = f"val-aux/{benchmark}/budget_exhausted/mean@1"
        evaluations = summary["evaluation"]
        initial, final = evaluations[0], evaluations[-1]
        summary["benchmarks"][benchmark] = {
            "examples": examples,
            "initial_outer": initial["outer_step"],
            "final_outer": final["outer_step"],
            "initial_accuracy": initial[accuracy],
            "final_accuracy": final[accuracy],
            "change_percentage_points": 100 * (final[accuracy] - initial[accuracy]),
            "initial_budget_exhausted": initial[exhausted],
            "final_budget_exhausted": final[exhausted],
        }
    return summary, inner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Verified math runs: fixed greedy tests, one seed per method")
    for run in args.run:
        summary, inner = summarize(run, output, args.verification_root)
        summaries.append(summary)
        label = (
            summary["method"].removeprefix("mirror_")
            + f" (O={summary['completed_outer_calls']}, J={inner[0]['epochs']})"
        )
        first = [row for row in inner if row["inner_step"] == 0]
        outer = [row["outer_call"] for row in first]
        axes[0, 0].plot(
            outer,
            [row["metrics"]["mirror/teacher_center/mean"] for row in first],
            label=label,
        )
        axes[0, 1].plot(
            outer,
            [-row["metrics"]["mirror/sampled_reward/mean"] for row in first],
            label=label,
        )
        axes[0, 2].plot(
            [row["outer_call"] * row["epochs"] + row["inner_step"] for row in inner],
            [row["metrics"]["mirror/importance_ess_fraction"] for row in inner],
            label=label,
        )
        for column, benchmark in enumerate(BENCHMARKS):
            evaluations = summary["evaluation"]
            axes[1, column].plot(
                [row["outer_step"] for row in evaluations],
                [100 * row[f"val-core/{benchmark}/acc/mean@1"] for row in evaluations],
                "o-",
                label=label,
            )
    labels = (
        "Tail-aware teacher KL lower bound (nats)",
        "Sampled reverse KL (nats)",
        "ESS fraction",
    )
    for column, label in enumerate(labels):
        axes[0, column].set_ylabel(label)
        axes[0, column].set_xlabel("Inner update" if column == 2 else "Frozen rollout outer")
    for column, (benchmark, examples) in enumerate(BENCHMARKS.items()):
        axes[1, column].set_title(f"{benchmark}, n={examples}")
        axes[1, column].set_ylabel("Accuracy (%)")
        axes[1, column].set_xlabel("Completed outer updates")
        axes[1, column].set_ylim(0, 100)
    axes[0, 2].set_ylim(0, 1.02)
    for axis in axes.flat:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=200)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    (output / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (output / "benchmark-deltas.csv").open("w") as stream:
        fields = ["method", "benchmark", *next(iter(summaries[0]["benchmarks"].values()))]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for benchmark, values in summary["benchmarks"].items():
                writer.writerow({"method": summary["method"], "benchmark": benchmark, **values})
    print(json.dumps({"verified_runs": len(summaries), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
