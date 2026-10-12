"""Fixed-pool scalar scheduling curves on update, backward, and wall-time axes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.analysis.scalar_large_report import rows_at


def label(run):
    config = run["summary"]["config"]
    if run["row"]["id"].startswith("ablate-"):
        return run["row"]["id"].removeprefix("ablate-")
    method = config["method"]
    if method == "constant":
        return f"constant {config['lr']:.3g}"
    if method == "search":
        return f"search every {config['search_every']}"
    if method == "feedback":
        return f"feedback {config['target_rate']:.3g}"
    return method


def plot_phase(root, output):
    runs = rows_at(root)
    if not runs:
        return []
    output.mkdir(parents=True, exist_ok=True)
    seeds = sorted({run["row"]["seed"] for run in runs})
    generated = []
    for seed in seeds:
        subset = [run for run in runs if run["row"]["seed"] == seed]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for run in subset:
            points = run["points"]
            axes_values = (
                [point["update"] for point in points],
                [point["counts"]["backwards"] for point in points],
                [point["seconds"] for point in points],
            )
            for row, (measure, split) in enumerate(
                (("bpo", "train"), ("teacher_reverse_kl", "validation"))
            ):
                initial = points[0]["metrics"][split][measure]
                values = [point["metrics"][split][measure] / initial for point in points]
                for column, x in enumerate(axes_values):
                    axes[row, column].semilogy(
                        x, values, marker=".", markersize=3, linewidth=1.3, label=label(run)
                    )
        for row in range(2):
            for column, xlabel in enumerate(
                ("Optimizer updates", "Physical backward calls", "Elapsed seconds")
            ):
                axes[row, column].set_xlabel(xlabel)
                axes[row, column].grid(alpha=0.2)
                axes[row, column].set_ylabel(
                    "Train full BPO / initial"
                    if row == 0
                    else "Held-out teacher reverse KL / initial"
                )
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Frozen DeepSeek 1.5B pool, seed {seed}; independent FP32 audits")
        stem = output / f"{root.name}-seed-{seed}"
        for extension in ("png", "pdf"):
            path = stem.with_suffix("." + extension)
            fig.savefig(path, dpi=180)
            generated.append(str(path))
        plt.close(fig)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = plot_phase(args.root, args.output)
    (args.output / "figure-index.json").write_text(json.dumps(paths, indent=2) + "\n")
    print(json.dumps({"figures": len(paths)}))


if __name__ == "__main__":
    main()
