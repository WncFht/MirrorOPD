#!/usr/bin/env python3
"""Analyze scalar reverse-KL Top-K estimators only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def calibrate(z: torch.Tensor, target: float) -> float:
    lo, hi = 0.05, 20.0
    for _ in range(42):
        mid = (lo + hi) / 2
        p = torch.softmax(mid * z, -1)
        if torch.topk(p, 32).values.sum() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@torch.no_grad()
def make_rows(vocab: int, masses: list[float], ks: list[int], seed: int, device: str):
    torch.manual_seed(seed)
    z = torch.randn(vocab, device=device)
    z_ref = torch.randn(vocab, device=device)
    rows = []
    for mass in masses:
        p = torch.softmax(calibrate(z, mass) * z, -1)
        q = torch.softmax(z_ref, -1)
        log_ratio = torch.log(p) - torch.log(q)
        kl = (p * log_ratio).sum()
        order = torch.argsort(p, descending=True)
        for k in ks:
            mask = torch.zeros_like(p, dtype=torch.bool)
            if k:
                mask[order[:k]] = True
            tail = ~mask
            head_value = (p[mask] * log_ratio[mask]).sum()
            tail_value = (p[tail] * log_ratio[tail]).sum()
            trunc_bias = head_value - kl

            # One Y~p corrects the missing tail, hence this scalar estimator is
            # unbiased on-policy. Only the tail contributes to its variance.
            corrected_second = (p[tail] * log_ratio[tail].square()).sum()
            corrected_variance = corrected_second - tail_value.square()
            rows.append(
                {
                    "mass_target": mass,
                    "top_k": k,
                    "kl": float(kl),
                    "head_value": float(head_value),
                    "tail_value": float(tail_value),
                    "trunc_bias": float(trunc_bias),
                    "trunc_abs_relative_bias": float(trunc_bias.abs() / kl.abs()),
                    "trunc_variance": 0.0,
                    "corrected_bias": 0.0,
                    "corrected_abs_relative_bias": 0.0,
                    "corrected_variance": float(corrected_variance.clamp_min(0)),
                    "corrected_std": float(corrected_variance.clamp_min(0).sqrt()),
                    "corrected_relative_std": float(
                        corrected_variance.clamp_min(0).sqrt() / kl.abs()
                    ),
                }
            )
    return rows


def write_csv(rows, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_axis(axis, xlabel, ylabel):
    axis.set_xscale("symlog", linthresh=1)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", alpha=0.25)


def plot_bias(rows, out: Path):
    masses = sorted({row["mass_target"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, mass in zip(axes.flat, masses):
        cur = [row for row in rows if row["mass_target"] == mass]
        x = [row["top_k"] for row in cur]
        axis.plot(x, [max(row["trunc_abs_relative_bias"], 1e-12) for row in cur], "o-", label="Trunc-Top-K")
        axis.plot(x, [1e-12 for _ in cur], "--", label="Top-K corrected")
        axis.set_title(f"M32(pi)={mass:g}")
        configure_axis(axis, "Top-K (K=0 is sampled-token K1)", "absolute relative bias")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Scalar KL bias: deterministic truncation versus sampled-token correction")
    fig.tight_layout()
    fig.savefig(out / "topk_value_bias.png", dpi=220)
    fig.savefig(out / "topk_value_bias.pdf")
    plt.close(fig)


def plot_variance(rows, out: Path):
    masses = sorted({row["mass_target"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, mass in zip(axes.flat, masses):
        cur = [row for row in rows if row["mass_target"] == mass]
        x = [row["top_k"] for row in cur]
        axis.plot(x, [1e-12 for _ in cur], "--", color="tab:gray", label="Trunc-Top-K (zero variance)")
        axis.plot(x, [max(row["corrected_relative_std"] ** 2, 1e-12) for row in cur], "o-", color="tab:blue", label="Top-K corrected")
        axis.set_title(f"M32(pi)={mass:g}")
        configure_axis(axis, "Top-K (K=0 is sampled-token K1)", "relative variance")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Scalar KL variance: only the sampled tail contributes")
    fig.tight_layout()
    fig.savefig(out / "topk_value_variance.png", dpi=220)
    fig.savefig(out / "topk_value_variance.pdf")
    plt.close(fig)


def plot_rmse(rows, out: Path):
    masses = sorted({row["mass_target"] for row in rows})
    batch_sizes = [1, 32, 512, 32768]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(batch_sizes)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, mass in zip(axes.flat, masses):
        cur = [row for row in rows if row["mass_target"] == mass]
        x = [row["top_k"] for row in cur]
        axis.plot(x, [max(row["trunc_abs_relative_bias"], 1e-12) for row in cur], "k--", label="Trunc-Top-K (all B)")
        for color, batch_size in zip(colors, batch_sizes):
            corrected_rmse = [max(row["corrected_relative_std"] / np.sqrt(batch_size), 1e-12) for row in cur]
            axis.plot(x, corrected_rmse, "o-", color=color, label=f"Top-K corrected, B={batch_size}")
        axis.set_title(f"M32(pi)={mass:g}")
        configure_axis(axis, "Top-K (K=0 is sampled-token K1)", "relative scalar-KL RMSE")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Scalar KL RMSE: truncation bias versus sampled-tail variance")
    fig.tight_layout()
    fig.savefig(out / "topk_value_rmse_vs_k.png", dpi=220)
    fig.savefig(out / "topk_value_rmse_vs_k.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    rows = make_rows(args.vocab, [0.2, 0.5, 0.8, 0.9], [0, 1, 4, 8, 16, 32, 64, 128, 256, 512], args.seed, device)
    write_csv(rows, output / "value_metrics.csv")
    plot_bias(rows, output)
    plot_variance(rows, output)
    plot_rmse(rows, output)
    (output / "value_analysis_metadata.json").write_text(json.dumps({"device": device, "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if device == "cuda" else None, "estimators": ["Trunc-Top-K", "Top-K corrected with one sampled token"]}, indent=2))


if __name__ == "__main__":
    main()
