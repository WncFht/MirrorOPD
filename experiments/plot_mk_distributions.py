#!/usr/bin/env python3
"""Plot full-GSM8K Top-K mass distributions for two Qwen3 rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KS = [1, 2, 4, 8, 16, 32]
COLORS = {"Qwen3-1.7B": "#2166ac", "Qwen3-4B": "#b2182b"}


def quantile_table(df: pd.DataFrame, label: str, level: str) -> pd.DataFrame:
    rows = []
    for k in KS:
        x = df[f"m{k}"] if level == "prefix" else df[f"mean_m{k}"]
        rows.append(
            {
                "model": label,
                "level": level,
                "K": k,
                "mean": x.mean(),
                "median": x.median(),
                "p01": x.quantile(0.01),
                "p10": x.quantile(0.10),
                "p90": x.quantile(0.90),
                "p99": x.quantile(0.99),
            }
        )
    if level == "prefix":
        x = df["tail32"]
        rows.append(
            {
                "model": label,
                "level": level,
                "K": "tail32",
                "mean": x.mean(),
                "median": x.median(),
                "p01": x.quantile(0.01),
                "p10": x.quantile(0.10),
                "p90": x.quantile(0.90),
                "p99": x.quantile(0.99),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", required=True, help="1.7B result directory")
    ap.add_argument("--four", required=True, help="4B result directory")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    specs = [("Qwen3-1.7B", Path(args.one)), ("Qwen3-4B", Path(args.four))]
    data = {}
    stat = []
    for label, root in specs:
        prefix = pd.read_csv(root / "prefix_mk.csv")
        prompt = pd.read_csv(root / "prompt_mk.csv")
        for frame in (prefix, prompt):
            for k in KS:
                col = f"m{k}" if frame is prefix else f"mean_m{k}"
                frame[col] = frame[col].clip(lower=0.0, upper=1.0)
        data[label] = (prefix, prompt)
        stat += [
            quantile_table(prefix, label, "prefix"),
            quantile_table(prompt, label, "prompt"),
        ]
    pd.concat(stat, ignore_index=True).to_csv(
        out / "mk_distribution_stats.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, (prefix, _) in data.items():
        ax.plot(
            KS,
            [prefix[f"m{k}"].mean() for k in KS],
            marker="o",
            label=label,
            color=COLORS[label],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(map(str, KS))
    ax.set_ylim(0.9, 1.001)
    ax.set_xlabel("Top-K")
    ax.set_ylabel(r"mean $M(K)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "mean_mk_vs_k.png", dpi=220)
    fig.savefig(out / "mean_mk_vs_k.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, (prefix, _) in data.items():
        tail = np.maximum(prefix["tail32"].to_numpy(), 1e-30)
        ax.hist(
            tail,
            bins=np.logspace(np.log10(tail.min()), 0, 55),
            histtype="step",
            density=True,
            linewidth=1.8,
            label=label,
            color=COLORS[label],
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"tail mass $1-M(32)$")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tail32_histogram.png", dpi=220)
    fig.savefig(out / "tail32_histogram.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, (prefix, _) in data.items():
        tail = np.sort(np.maximum(prefix["tail32"].to_numpy(), 1e-30))
        y = np.arange(1, len(tail) + 1) / len(tail)
        ax.step(tail, y, where="post", label=label, color=COLORS[label])
    ax.set_xscale("log")
    ax.set_xlabel(r"tail mass $1-M(32)$")
    ax.set_ylabel("empirical CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tail32_ecdf.png", dpi=220)
    fig.savefig(out / "tail32_ecdf.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (label, (prefix, prompt)) in zip(axes, data.items()):
        vals = prompt["mean_m32"].to_numpy()
        ax.hist(vals, bins=40, color=COLORS[label], alpha=0.78)
        ax.axvline(
            np.median(vals),
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"median={np.median(vals):.6f}",
        )
        ax.set_title(label)
        ax.set_xlabel(r"per-prompt mean $M(32)$")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("number of prompts")
    fig.tight_layout()
    fig.savefig(out / "prompt_mean_m32_histogram.png", dpi=220)
    fig.savefig(out / "prompt_mean_m32_histogram.pdf")
    plt.close(fig)

    # A compact machine-readable summary for the report.
    summary = []
    for label, (prefix, prompt) in data.items():
        for name, frame in [("prefix", prefix), ("prompt", prompt)]:
            x = frame["tail32"] if name == "prefix" else 1.0 - frame["mean_m32"]
            summary.append(
                {
                    "model": label,
                    "level": name,
                    "n": len(x),
                    "mean_tail32": x.mean(),
                    "median_tail32": x.median(),
                    "p90_tail32": x.quantile(0.90),
                    "p99_tail32": x.quantile(0.99),
                    "fraction_tail_lt_1e-3": (x < 1e-3).mean(),
                    "fraction_tail_lt_1e-4": (x < 1e-4).mean(),
                    "fraction_tail_lt_1e-5": (x < 1e-5).mean(),
                }
            )
    pd.DataFrame(summary).to_csv(out / "tail32_summary.csv", index=False)


if __name__ == "__main__":
    main()
