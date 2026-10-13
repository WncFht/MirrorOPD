"""Same-checkpoint LR responses: one-step gain, forty-step loss, and gradient norms."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.analysis.scalar_large_report import rows_at


def plot(root, output):
    groups = {}
    sources = {}
    for run in rows_at(root):
        directory = root / run["row"]["id"]
        fork = json.loads((directory / "fork.jsonl").read_text().splitlines()[0])
        updates = [
            json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()
        ]
        groups.setdefault(fork["source_update"], []).append((run, updates))
        for name in ("summary.json", "audits.jsonl", "updates.jsonl", "fork.jsonl"):
            path = directory / name
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not groups:
        raise ValueError("no finished same-state branches")
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        3, len(groups), figsize=(5.4 * len(groups), 11), squeeze=False, constrained_layout=True
    )
    statistics = []
    for column, (stage, branches) in enumerate(sorted(groups.items())):
        branches.sort(key=lambda item: item[0]["row"]["lr"])
        initial = [run["points"][0]["metrics"]["train"]["bpo"] for run, _ in branches]
        norms = [updates[0]["gradient_norm_before_clip"] for _, updates in branches]
        response = []
        for run, updates in branches:
            points = run["points"]
            q0 = points[0]["metrics"]["train"]["bpo"]
            q1 = next(point["metrics"]["train"]["bpo"] for point in points if point["update"] == 1)
            lr = run["row"]["lr"]
            response.append((lr / 1e-6, 100 * (1 - q1 / q0)))
            label = f"{lr:.3g}"
            axes[1, column].semilogy(
                [p["update"] for p in points],
                [p["metrics"]["train"]["bpo"] / q0 for p in points],
                label=label,
            )
            axes[2, column].semilogy(
                [p["update_after"] for p in updates],
                [p["gradient_norm_before_clip"] for p in updates],
                label=label,
            )
        axes[0, column].plot(*zip(*response, strict=True), marker="o")
        axes[0, column].set_xscale("log", base=2)
        axes[0, column].axhline(0, color="gray", linestyle=":")
        axes[0, column].set_title(f"Parent update {stage}")
        axes[0, column].set_xlabel("LR / 1e-6")
        axes[1, column].set_xlabel("Branch optimizer updates")
        axes[2, column].set_xlabel("Branch optimizer updates")
        statistics.append(
            {
                "parent_update": stage,
                "initial_bpo_range": [min(initial), max(initial)],
                "initial_grad_norm_range": [min(norms), max(norms)],
                "one_step_gain_percent": response,
            }
        )
        for row in range(3):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("One-step full-BPO reduction (%)")
    axes[1, 0].set_ylabel("Full BPO / branch initial")
    axes[2, 0].set_ylabel("Pre-clip full-pool gradient norm")
    axes[1, 0].legend(title="LR", fontsize=8, ncol=2)
    figure.suptitle(
        "Same parameters, Adam moments and frozen supervision within each column\n64 complete trajectories per update; FP32, eta=0.3, Top-16 production gradient"
    )
    for extension in ("png", "pdf"):
        figure.savefig(output / f"same-state-lr-response.{extension}", dpi=180)
    plt.close(figure)
    (output / "same-state-lr-response.json").write_text(
        json.dumps({"stages": statistics, "source_sha256": sources}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.output)


if __name__ == "__main__":
    main()
