"""Summarize frozen scalar scheduling experiments and render numeric plots."""

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def row_summary(run):
    points = run["trajectory"]
    initial, final = points[0]["exact"], points[-1]["exact"]
    row = {
        "method": run.get("method", run.get("configuration", {}).get("method")),
        "lr": run.get("lr", run.get("configuration", {}).get("lr")),
        "backwards": run["backwards"],
        "forwards": run["forwards"],
        "seconds": run["seconds"],
        "initial_exact": initial,
        "final_exact": final,
        "final_ratio": final / initial,
        "rejected": run["rejected"],
    }
    for ratio in ("0.1", "0.01", "0.001"):
        index = run["thresholds"][ratio]["sustained5"]
        last_above = max(
            (i for i, point in enumerate(points) if point["exact"] > float(ratio) * initial),
            default=-1,
        )
        persistent = last_above + 1
        row[f"persistent_{ratio}"] = persistent if persistent < len(points) else None
        row[f"steps_{ratio}"] = index
        row[f"seconds_{ratio}"] = points[index].get("seconds") if index is not None else None
        row[f"forwards_{ratio}"] = points[index]["forwards"] if index is not None else None
    return row


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_toy(root, name):
    path = root / f"{name}-expanded.json"
    if not path.exists():
        path = path.with_suffix(".json.gz")
    if not path.exists():
        return
    source = read(path)
    rows = []
    groups = defaultdict(list)
    for run in source["results"]:
        row = row_summary(run)
        row.update(seed=run["seed"], optimizer=run["optimizer"], case=json.dumps(run["case"]))
        rows.append(row)
        groups[(run["optimizer"], json.dumps(run["case"]), run["method"], run["lr"])].append(row)
    save_csv(root / f"{name}-runs.csv", rows)
    aggregates = []
    for (optimizer, case, method, lr), group in groups.items():
        hits = [r["steps_0.01"] for r in group if r["steps_0.01"] is not None]
        aggregates.append(
            {
                "optimizer": optimizer,
                "case": case,
                "method": method,
                "lr": lr,
                "seeds": len(group),
                "median_final_ratio": statistics.median(r["final_ratio"] for r in group),
                "hits_0.01": len(hits),
                "median_steps_0.01_successful": statistics.median(hits) if hits else None,
                "median_seconds": statistics.median(r["seconds"] for r in group),
                "median_forwards": statistics.median(r["forwards"] for r in group),
            }
        )
    save_csv(root / f"{name}-aggregate.csv", aggregates)


def plot_lm(root):
    rows = []
    for path in sorted(root.glob("lm-*.json")):
        run = read(path)
        row = row_summary(run)
        row["file"] = path.name
        row["objective"] = run["configuration"]["objective"]
        row["monitor"] = run["configuration"]["monitor"]
        row["precision"] = run["configuration"].get("precision", "bf16")
        row["final_sampled_metric"] = run["trajectory"][-1]["metric"]
        row["final_teacher_kl"] = run["trajectory"][-1]["teacher_kl"]
        rows.append(row)
    save_csv(root / "lm-runs.csv", rows)
    paths = sorted(root.glob("lm-fp32-exact-*long.json"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for path in paths:
        run = read(path)
        points = run["trajectory"]
        label = (
            "search"
            if run["configuration"]["method"] == "search"
            else f"constant {run['configuration']['lr']:.3g}"
        )
        normalized = [p["exact"] / points[0]["exact"] for p in points]
        axes[0].semilogy([p["step"] for p in points], normalized, label=label)
        axes[1].semilogy([p["seconds"] for p in points], normalized, label=label)
        cumulative = [0]
        for p in points[:-1]:
            cumulative.append(cumulative[-1] + p.get("selected_lr", p["lr"]))
        axes[2].semilogy(cumulative, normalized, label=label)
    for ax, xlabel in zip(
        axes, ("Backward calls", "Elapsed seconds", "Cumulative learning rate"), strict=True
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Full BPO / initial Full BPO")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("DeepSeek 1.5B, fixed 416 prefixes, full BPO objective, FP32")
    fig.tight_layout()
    fig.savefig(root / "lm-exact.png", dpi=180)
    fig.savefig(root / "lm-exact.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for path in sorted(root.glob("lm-fp32-sampled-*.json")):
        run = read(path)
        points = run["trajectory"]
        label = path.stem.replace("lm-fp32-sampled-", "")
        for ax, field in zip(axes, ("metric", "exact"), strict=True):
            ax.semilogy(
                [p["step"] for p in points],
                [p[field] / points[0][field] for p in points],
                label=label,
            )
            ax.set_xlabel("Backward calls")
            ax.grid(alpha=0.25)
    axes[0].set_ylabel("Sampled residual square / initial")
    axes[1].set_ylabel("Full BPO / initial")
    axes[0].legend(fontsize=7)
    fig.suptitle("Production sampled score with Top-16 + tail center, fixed pool")
    fig.tight_layout()
    fig.savefig(root / "lm-sampled-mismatch.png", dpi=180)
    fig.savefig(root / "lm-sampled-mismatch.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/scalar-schedule"))
    args = parser.parse_args()
    for name in ("quadratic", "categorical"):
        summarize_toy(args.root, name)
    plot_lm(args.root)
    print(json.dumps({"root": str(args.root), "status": "summarized"}))


if __name__ == "__main__":
    main()
