#!/usr/bin/env python3
"""Plot how Top-K changes single-token statistics and estimator error."""

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
def structural_rows(
    vocab: int, masses: list[float], ks: list[int], seed: int, device: str
):
    torch.manual_seed(seed)
    z = torch.randn(vocab, device=device)
    zr = torch.randn(vocab, device=device)
    rows = []
    for mass in masses:
        p = torch.softmax(calibrate(z, mass) * z, -1)
        q = torch.softmax(zr, -1)
        lr = torch.log(p) - torch.log(q)
        kl = (p * lr).sum()
        true = p * (lr - kl)
        true_norm = true.square().sum().sqrt()
        order = torch.argsort(p, descending=True)
        for k in ks:
            if k == 0:
                h = torch.zeros_like(p)
                mask = torch.zeros_like(p, dtype=torch.bool)
            else:
                idx = order[:k]
                mask = torch.zeros_like(p, dtype=torch.bool)
                mask[idx] = True
                w = torch.where(mask, p * (lr + 1), torch.zeros_like(p))
                h = w - p * w.sum()
            tail = ~mask
            # Literal paper estimator's expected tail gradient.
            tail_w = torch.where(tail, p * lr, torch.zeros_like(p))
            expected = h + tail_w - p * tail_w.sum()
            bias = (expected - true).square().sum().sqrt() / true_norm
            trunc = (h - true).square().sum().sqrt() / true_norm
            tail_mass = p[tail].sum()
            tail_kl = (p[tail] * lr[tail]).sum()
            # Exact single-sample RMSE of head + masked K4 tail.
            d = h - true
            dp = (d * p).sum()
            cross = (p[tail] * lr[tail] * (d[tail] - dp)).sum()
            vnorm2 = (
                p[tail] * lr[tail].square() * (1 - 2 * p[tail] + (p.square().sum()))
            ).sum()
            single_mse = d.square().sum() + 2 * cross + vnorm2
            rows.append(
                {
                    "mass_target": mass,
                    "top_k": k,
                    "top_mass": float(p[mask].sum()),
                    "tail_mass": float(tail_mass),
                    "tail_kl": float(tail_kl),
                    "kl": float(kl),
                    "tail_kl_fraction_abs": float(tail_kl.abs() / kl.abs()),
                    "trunc_bias_rmse": float(trunc),
                    "paper_asymptotic_bias": float(bias),
                    "single_token_rmse": float(single_mse.sqrt() / true_norm),
                }
            )
    return rows


def load_metrics(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(rows, path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_structural(rows, out: Path):
    masses = sorted({r["mass_target"] for r in rows})
    ks = sorted({r["top_k"] for r in rows})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flat
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(masses)))
    for c, mass in zip(colors, masses):
        cur = [r for r in rows if r["mass_target"] == mass]
        x = [r["top_k"] for r in cur]
        ax[0].plot(
            x, [r["top_mass"] for r in cur], "o-", color=c, label=f"M32={mass:g}"
        )
        ax[1].plot(x, [r["tail_mass"] for r in cur], "o-", color=c)
        ax[2].plot(
            x,
            [r["single_token_rmse"] for r in cur],
            "o-",
            color=c,
            label=f"M32={mass:g}",
        )
    for a in ax:
        a.set_xscale("symlog", linthresh=1)
        a.grid(True, which="both", alpha=0.25)
        a.set_xlabel("Top-K (K=0 is sampled K4)")
    ax[0].set_ylabel("probability mass in head")
    ax[0].set_title("Head mass")
    ax[1].set_ylabel("probability mass in tail")
    ax[1].set_title("Tail mass")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("single-token relative RMSE")
    ax[2].set_title("One prefix, one sampled token")
    for c, mass in zip(colors, masses):
        cur = [r for r in rows if r["mass_target"] == mass]
        x = [r["top_k"] for r in cur]
        ax[3].plot(x, [r["trunc_bias_rmse"] for r in cur], "o:", color=c, alpha=0.8)
        ax[3].plot(
            x,
            [r["paper_asymptotic_bias"] for r in cur],
            "o-",
            color=c,
            label=f"M32={mass:g}",
        )
    ax[3].set_yscale("log")
    ax[3].set_ylabel("relative gradient bias")
    ax[3].set_title("Bias floor: truncation vs literal masked tail")
    ax[0].legend(fontsize=8)
    ax[2].legend(fontsize=8)
    fig.suptitle("How K changes the Top-K KL estimator")
    fig.tight_layout()
    fig.savefig(out / "topk_structural_sweep.png", dpi=220)
    fig.savefig(out / "topk_structural_sweep.pdf")
    plt.close(fig)


def plot_error_sweep(metrics, out: Path):
    masses = sorted({float(r["mass_target"]) for r in metrics})
    bs = [1, 32, 512, 32768]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for a, mass in zip(ax.flat, masses):
        cur = [r for r in metrics if float(r["mass_target"]) == mass]
        for b in bs:
            rs = [r for r in cur if int(r["batch_size"]) == b]
            rs.sort(key=lambda r: int(r["top_k"]))
            a.plot(
                [0] + [int(r["top_k"]) for r in rs],
                [float(rs[0]["sampled_rmse"])] + [float(r["topk_rmse"]) for r in rs],
                "o-",
                label=f"B={b}",
            )
        a.set_title(f"M32(pi)={mass:g}")
        a.set_xscale("symlog", linthresh=1)
        a.grid(True, which="both", alpha=0.25)
        a.set_yscale("log")
    ax[0, 0].set_ylabel("relative gradient RMSE")
    ax[1, 0].set_ylabel("relative gradient RMSE")
    ax[1, 0].set_xlabel("Top-K (K=0: sampled K4)")
    ax[1, 1].set_xlabel("Top-K (K=0: sampled K4)")
    ax[0, 1].legend(fontsize=8)
    fig.suptitle("Top-K error versus effective Monte Carlo batch size")
    fig.tight_layout()
    fig.savefig(out / "topk_error_vs_k.png", dpi=220)
    fig.savefig(out / "topk_error_vs_k.pdf")
    plt.close(fig)


def plot_bias_variance(rows, out: Path):
    masses = sorted({r["mass_target"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(masses)))
    for c, mass in zip(colors, masses):
        cur = [r for r in rows if r["mass_target"] == mass]
        x = [r["top_k"] for r in cur]
        bias_trunc = np.asarray([r["trunc_bias_rmse"] for r in cur])
        bias_masked = np.asarray([r["paper_asymptotic_bias"] for r in cur])
        rmse = np.asarray([r["single_token_rmse"] for r in cur])
        std = np.sqrt(np.maximum(rmse * rmse - bias_masked * bias_masked, 0.0))
        ax[0].plot(x, bias_trunc, ":", color=c, alpha=0.8)
        ax[0].plot(x, bias_masked, "o-", color=c, label=f"M32={mass:g}")
        ax[1].plot(x, std, "o-", color=c, label=f"M32={mass:g}")
    for a in ax:
        a.set_xscale("symlog", linthresh=1)
        a.set_yscale("log")
        a.grid(True, which="both", alpha=0.25)
        a.set_xlabel("Top-K (K=0 is sampled K4)")
    ax[0].set_title("Bias component")
    ax[0].set_ylabel("relative error")
    ax[0].legend(fontsize=8)
    ax[1].set_title("Standard-deviation component")
    ax[1].legend(fontsize=8)
    fig.suptitle(
        "Bias--variance decomposition for one prefix and one sampled token\nsolid: literal masked-tail bias; dotted: truncated-head bias"
    )
    fig.tight_layout()
    fig.savefig(out / "topk_bias_variance.png", dpi=220)
    fig.savefig(out / "topk_bias_variance.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    rows = structural_rows(
        a.vocab,
        [0.2, 0.5, 0.8, 0.9],
        [0, 1, 4, 8, 16, 32, 64, 128, 256, 512],
        a.seed,
        device,
    )
    write_csv(rows, out / "structural_metrics.csv")
    metrics = load_metrics(Path(a.metrics))
    plot_structural(rows, out)
    plot_error_sweep(metrics, out)
    plot_bias_variance(rows, out)
    (out / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "device": device,
                "torch": torch.__version__,
                "cuda": torch.cuda.get_device_name(0) if device == "cuda" else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
