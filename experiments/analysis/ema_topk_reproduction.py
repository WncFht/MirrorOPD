#!/usr/bin/env python3
"""Reproduce the synthetic Top-K KL bias/variance experiment from 2602.04417.

The computation is intentionally token-level: the vocabulary is sampled once,
then independent tokens are drawn from the same categorical policy.  This
isolates estimator error from autoregressive rollout noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def calibrate_scale(z: torch.Tensor, target: float, k: int = 32) -> float:
    """Find s such that mass of softmax(s*z) on its top-k is target."""
    lo, hi = 0.05, 20.0
    for _ in range(42):
        mid = (lo + hi) / 2
        p = torch.softmax(mid * z, dim=-1)
        mass = torch.topk(p, k=k, dim=-1).values.sum().item()
        if mass < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def head_gradient(p: torch.Tensor, log_ratio: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
    """Gradient wrt logits of sum_{j in top_idx} p_j log(p_j/q_j)."""
    a = torch.zeros_like(p)
    a[top_idx] = log_ratio[top_idx] + 1.0
    return p * (
        a * (torch.zeros_like(p).scatter(0, top_idx, p[top_idx]))
        - (p[top_idx] * (log_ratio[top_idx] + 1.0)).sum()
    )


def head_gradient_fast(
    p: torch.Tensor, log_ratio: torch.Tensor, top_idx: torch.Tensor
) -> torch.Tensor:
    """Same expression written without a temporary scatter product."""
    mask = torch.zeros_like(p, dtype=torch.bool)
    mask[top_idx] = True
    weighted = torch.where(mask, p * (log_ratio + 1.0), torch.zeros_like(p))
    return weighted - p * weighted.sum()


@torch.no_grad()
def estimate_rmse(
    p: torch.Tensor,
    log_ratio: torch.Tensor,
    top_idx: torch.Tensor,
    true_grad: torch.Tensor,
    top_k: int,
    batch_size: int,
    repeats: int,
    generator: torch.Generator,
    chunk_repeats: int = 32,
) -> dict[str, float]:
    """Estimate relative RMSE for sampled, truncated, and Top-K estimators."""
    v = p.numel()
    mask = torch.zeros(v, dtype=torch.bool, device=p.device)
    mask[top_idx[:top_k]] = True
    hgrad = head_gradient_fast(p, log_ratio, top_idx[:top_k])
    true_norm2 = torch.sum(true_grad.square()).item()
    sq_sample = 0.0
    sq_trunc = float(torch.sum((hgrad - true_grad).square()).item())
    sq_topk = 0.0
    p_for_sample = p.unsqueeze(0)
    for start in range(0, repeats, chunk_repeats):
        r = min(chunk_repeats, repeats - start)
        samples = torch.multinomial(
            p_for_sample.expand(r, -1),
            batch_size,
            replacement=True,
            generator=generator,
        )
        lr = log_ratio[samples]
        # For each replicate, g_sample = E[lr * (onehot - p)].
        sums = torch.zeros((r, v), device=p.device, dtype=p.dtype)
        sums.scatter_add_(1, samples, lr)
        mean_lr = lr.mean(dim=1, keepdim=True)
        sample_grad = sums / batch_size - mean_lr * p
        sq_sample += torch.sum((sample_grad - true_grad).square(), dim=1).sum().item()
        tail = ~mask[samples]
        tail_lr = lr * tail
        tail_sums = torch.zeros((r, v), device=p.device, dtype=p.dtype)
        tail_sums.scatter_add_(1, samples, tail_lr)
        tail_mean = tail_lr.mean(dim=1, keepdim=True)
        top_grad = hgrad + tail_sums / batch_size - tail_mean * p
        sq_topk += torch.sum((top_grad - true_grad).square(), dim=1).sum().item()
    return {
        "sampled_rmse": math.sqrt(sq_sample / repeats / true_norm2),
        "trunc_rmse": math.sqrt(sq_trunc / true_norm2),
        "topk_rmse": math.sqrt(sq_topk / repeats / true_norm2),
    }


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    z = torch.randn(args.vocab, device=device)
    z_ref = torch.randn(args.vocab, device=device)
    p_rows: list[dict[str, float]] = []
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for mass_target in args.masses:
        scale = calibrate_scale(z, mass_target)
        p = torch.softmax(scale * z, dim=-1)
        q = torch.softmax(z_ref, dim=-1)
        log_ratio = torch.log(p) - torch.log(q)
        kl = torch.sum(p * log_ratio)
        true_grad = p * (log_ratio - kl)
        top_idx = torch.topk(p, k=max(args.topks), sorted=True).indices
        realized_mass = torch.sum(p[top_idx[:32]]).item()
        for top_k in args.topks:
            for batch_size in args.batch_sizes:
                metrics = estimate_rmse(
                    p,
                    log_ratio,
                    top_idx,
                    true_grad,
                    top_k,
                    batch_size,
                    args.repeats,
                    gen,
                )
                p_rows.append(
                    {
                        "mass_target": mass_target,
                        "mass_realized": realized_mass,
                        "scale": scale,
                        "top_k": top_k,
                        "batch_size": batch_size,
                        "vocab": args.vocab,
                        "repeats": args.repeats,
                        **metrics,
                    }
                )
                print(json.dumps(p_rows[-1]), flush=True)
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(p_rows[0]))
        writer.writeheader()
        writer.writerows(p_rows)
    meta = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "args": vars(args),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    make_plot(p_rows, out)


def make_plot(rows: list[dict[str, float]], out: Path) -> None:
    masses = sorted({float(r["mass_target"]) for r in rows})
    ks = sorted({int(r["top_k"]) for r in rows})
    batches = sorted({int(r["batch_size"]) for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ks)))
    for ax, mass in zip(axes.flat, masses, strict=False):
        subset = [r for r in rows if float(r["mass_target"]) == mass]
        # Sampled K4 is common to all top-k choices; plot it once.
        sample = [r for r in subset if int(r["top_k"]) == ks[0]]
        ax.loglog(
            batches,
            [r["sampled_rmse"] for r in sample],
            "k--",
            lw=2,
            label="sampled K4",
        )
        for color, k in zip(colors, ks, strict=True):
            cur = [r for r in subset if int(r["top_k"]) == k]
            ax.loglog(batches, [r["trunc_rmse"] for r in cur], color=color, ls=":", alpha=0.65)
            ax.loglog(
                batches,
                [r["topk_rmse"] for r in cur],
                color=color,
                lw=2,
                label=f"Top-{k} + tail",
            )
        ax.set_title(rf"$M_{{32}}(\pi)\approx {mass:g}$")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylim(bottom=1e-3)
    axes[0, 0].set_ylabel("relative gradient RMSE")
    axes[1, 0].set_ylabel("relative gradient RMSE")
    axes[1, 0].set_xlabel("independent sampled tokens B")
    axes[1, 1].set_xlabel("independent sampled tokens B")
    axes[0, 1].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Top-K reverse-KL estimator: bias--variance tradeoff\nsynthetic iid token setting from 2602.04417"
    )
    fig.tight_layout()
    fig.savefig(out / "topk_kl_bias_variance.png", dpi=220)
    fig.savefig(out / "topk_kl_bias_variance.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/ema_topk_reproduction")
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--repeats", type=int, default=128)
    p.add_argument("--masses", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.9])
    p.add_argument("--topks", type=int, nargs="+", default=[4, 16, 32, 64, 128])
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
        ],
    )
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
