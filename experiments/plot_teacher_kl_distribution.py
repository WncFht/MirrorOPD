#!/usr/bin/env python3
"""Plot K_k^T distribution and its relation to entropy/rollout position."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prefix = pd.read_csv(root / "kt_prefix.csv")
    prompt = pd.read_csv(root / "kt_prompt.csv")
    x = prefix["kt_kl"].clip(lower=0)
    q = x.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
    pd.DataFrame({"quantile": q.index, "kt_kl": q.values}).to_csv(
        out / "kt_quantiles.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "n_prefixes": len(prefix),
                "n_prompts": len(prompt),
                "mean_kt_kl": x.mean(),
                "median_kt_kl": x.median(),
                "p90_kt_kl": x.quantile(0.9),
                "p99_kt_kl": x.quantile(0.99),
                "max_kt_kl": x.max(),
                "fraction_lt_0p01": (x < 0.01).mean(),
                "fraction_lt_0p1": (x < 0.1).mean(),
                "fraction_lt_1": (x < 1).mean(),
                "prompt_mean_mean": prompt.mean_kt_kl.mean(),
                "prompt_mean_median": prompt.mean_kt_kl.median(),
                "prompt_mean_p90": prompt.mean_kt_kl.quantile(0.9),
                "prompt_mean_p99": prompt.mean_kt_kl.quantile(0.99),
            }
        ]
    ).to_csv(out / "kt_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = np.logspace(np.log10(max(x.min(), 1e-8)), np.log10(max(x.max(), 1e-8)), 70)
    ax.hist(x, bins=bins, histtype="step", linewidth=1.8, density=True, color="#b2182b")
    ax.set_xscale("log")
    ax.set_xlabel(r"$K_k^T(s)=D_{KL}(\pi_k\Vert\pi_T)$")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "kt_prefix_histogram.png", dpi=220)
    fig.savefig(out / "kt_prefix_histogram.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.sort(x.to_numpy())
    ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color="#b2182b")
    ax.set_xscale("log")
    ax.set_xlabel(r"$K_k^T(s)$")
    ax.set_ylabel("empirical CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "kt_prefix_ecdf.png", dpi=220)
    fig.savefig(out / "kt_prefix_ecdf.pdf")
    plt.close(fig)

    grouped = (
        prefix.groupby("step")
        .agg(
            mean_kt_kl=("kt_kl", "mean"),
            median_kt_kl=("kt_kl", "median"),
            p90_kt_kl=("kt_kl", lambda y: y.quantile(0.9)),
            count=("kt_kl", "size"),
        )
        .reset_index()
    )
    grouped.to_csv(out / "kt_by_position.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(grouped.step, grouped.mean_kt_kl, label="mean", color="#b2182b")
    ax.plot(grouped.step, grouped.median_kt_kl, label="median", color="#2166ac")
    ax.set_xlabel("generated position")
    ax.set_ylabel(r"$K_k^T$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "kt_by_position.png", dpi=220)
    fig.savefig(out / "kt_by_position.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.hexbin(
        prefix.student_entropy_nats,
        prefix.kt_kl,
        gridsize=55,
        bins="log",
        cmap="magma",
        mincnt=1,
    )
    ax.set_xlabel("student entropy (nats)")
    ax.set_ylabel(r"$K_k^T(s)$")
    ax.set_yscale("log")
    fig.colorbar(ax.collections[0], ax=ax, label="log10 count")
    fig.tight_layout()
    fig.savefig(out / "kt_vs_entropy.png", dpi=220)
    fig.savefig(out / "kt_vs_entropy.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
