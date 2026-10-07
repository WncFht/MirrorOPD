#!/usr/bin/env python3
"""Analyze scalar reverse-KL estimators separately from their gradients."""

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
        lr = torch.log(p) - torch.log(q)
        kl = (p * lr).sum()
        order = torch.argsort(p, descending=True)
        k1_var = (p * lr.square()).sum() - kl.square()
        k3pp = lr + torch.exp(-lr) - 1.0
        k3pp_var = (p * k3pp.square()).sum() - kl.square()
        for k in ks:
            mask = torch.zeros_like(p, dtype=torch.bool)
            if k:
                mask[order[:k]] = True
            tail = ~mask
            head_value = (p[mask] * lr[mask]).sum()
            tail_value = (p[tail] * lr[tail]).sum()
            topk_second = (p[tail] * lr[tail].square()).sum()
            topk_var = topk_second - tail_value.square()
            trunc_bias = head_value - kl
            rows.append({
                "mass_target": mass,
                "top_k": k,
                "kl": float(kl),
                "head_value": float(head_value),
                "tail_value": float(tail_value),
                "trunc_bias": float(trunc_bias),
                "trunc_abs_relative_bias": float(trunc_bias.abs() / kl.abs()),
                "sampled_k1_std": float(k1_var.clamp_min(0).sqrt()),
                "topk_value_std": float(topk_var.clamp_min(0).sqrt()),
                "k3pp_std": float(k3pp_var.clamp_min(0).sqrt()),
                "sampled_k1_relative_std": float(k1_var.clamp_min(0).sqrt() / kl.abs()),
                "topk_value_relative_std": float(topk_var.clamp_min(0).sqrt() / kl.abs()),
                "k3pp_relative_std": float(k3pp_var.clamp_min(0).sqrt() / kl.abs()),
            })
    return rows


def write_csv(rows, path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def plot_bias_variance(rows, out: Path):
    masses = sorted({r["mass_target"] for r in rows})
    colors = plt.cm.viridis(np.linspace(.1, .9, len(masses)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for c, mass in zip(colors, masses):
        cur = [r for r in rows if r["mass_target"] == mass]
        x = [r["top_k"] for r in cur]
        ax[0].plot(x, [max(r["trunc_abs_relative_bias"], 1e-12) for r in cur], "o-", color=c, label=f"M32={mass:g}")
        ax[1].plot(x, [max(r["sampled_k1_relative_std"], 1e-12) for r in cur], "--", color=c, alpha=.7)
        ax[1].plot(x, [max(r["topk_value_relative_std"], 1e-12) for r in cur], "o-", color=c, label=f"Top-K, M32={mass:g}")
        ax[1].plot(x, [max(r["k3pp_relative_std"], 1e-12) for r in cur], ":", color=c, alpha=.8)
    for a in ax:
        a.set_xscale("symlog", linthresh=1); a.set_yscale("log"); a.grid(True, which="both", alpha=.25)
        a.set_xlabel("Top-K (K=0 is sampled K1/K4)")
    ax[0].set_title("Scalar KL bias")
    ax[0].set_ylabel("absolute relative bias")
    ax[0].legend(fontsize=8)
    ax[1].set_title("Scalar KL one-token standard deviation")
    ax[1].set_ylabel("relative standard deviation")
    ax[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Scalar reverse-KL estimator: bias and variance\nsolid Top-K; dashed K1; dotted K3++")
    fig.tight_layout(); fig.savefig(out / "topk_value_bias_variance.png", dpi=220); fig.savefig(out / "topk_value_bias_variance.pdf"); plt.close(fig)


def plot_rmse(rows, out: Path):
    masses = sorted({r["mass_target"] for r in rows})
    bs = [1, 32, 512, 32768]
    colors = plt.cm.plasma(np.linspace(.1, .9, len(bs)))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for a, mass in zip(ax.flat, masses):
        cur = [r for r in rows if r["mass_target"] == mass]
        for c, b in zip(colors, bs):
            x = [r["top_k"] for r in cur]
            # Value RMSE of a B-token average: sqrt(bias^2 + variance/B).
            top = [np.sqrt(r["trunc_abs_relative_bias"] ** 2 + r["topk_value_relative_std"] ** 2 / b) for r in cur]
            k1 = [r["sampled_k1_relative_std"] / np.sqrt(b) for r in cur]
            a.plot(x, top, "o-", color=c, label=f"Top-K, B={b}")
            if b in (1, 32768):
                a.plot(x, k1, "--", color=c, alpha=.75, label=f"K1, B={b}")
        a.set_title(f"M32(pi)={mass:g}"); a.set_xscale("symlog", linthresh=1); a.set_yscale("log"); a.grid(True, which="both", alpha=.25)
    ax[0, 0].set_ylabel("relative scalar-KL RMSE"); ax[1, 0].set_ylabel("relative scalar-KL RMSE")
    ax[1, 0].set_xlabel("Top-K (K=0: sampled K1/K4)"); ax[1, 1].set_xlabel("Top-K (K=0: sampled K1/K4)")
    ax[0, 1].legend(fontsize=7, ncol=2)
    fig.suptitle("Scalar KL RMSE versus Top-K and effective averaging size B")
    fig.tight_layout(); fig.savefig(out / "topk_value_rmse_vs_k.png", dpi=220); fig.savefig(out / "topk_value_rmse_vs_k.pdf"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--vocab", type=int, default=32000); p.add_argument("--seed", type=int, default=20261008); p.add_argument("--cpu", action="store_true"); a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    rows = make_rows(a.vocab, [.2, .5, .8, .9], [0, 1, 4, 8, 16, 32, 64, 128, 256, 512], a.seed, device)
    write_csv(rows, out / "value_metrics.csv"); plot_bias_variance(rows, out); plot_rmse(rows, out)
    (out / "value_analysis_metadata.json").write_text(json.dumps({"device": device, "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if device == "cuda" else None}, indent=2))


if __name__ == "__main__": main()
