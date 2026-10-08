#!/usr/bin/env python3
"""Compare four scalar KL objects on the same empirical prefix measure.

Each point aggregates all prefixes at one retained support size K.
Bias is mean conditional error; variance is mean conditional sampling
variance, not cross-prefix dispersion. EMA observed errors use bf16 tokens
against NF4 scoring distributions. Axes retain identical integer K geometry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path.home() / ".agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv

fv.apply_rc("paper")

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()
inp = Path(a.input)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(inp)
positive_k = df.top_k > 0
fig, ax = plt.subplots(2, 2, figsize=(7, 5.8))
ax = ax.ravel()
colors = {
    "trunc": fv.HUES["vteal"],
    "normalized": fv.HUES["vcoral"],
    "ema": fv.HUES["vblue"],
    "ta": fv.HUES["vgreen"],
    "observed": fv.HUES["vgray"],
}
ax[0].plot(
    df.top_k,
    df.mean_topk_mass,
    "o-",
    color=fv.HUES["vgray"],
    label="mean student Top-K mass",
)
ax[0].set(xlabel="K", ylabel="mass", title="Head mass")
ax[0].set_ylim(-0.03, 1.03)
ax[0].grid(alpha=0.25)
ax[1].plot(df.top_k, df.bias_trunc, "o-", color=colors["trunc"], label="Truncated")
ax[1].plot(df.top_k, df.bias_tail_aware, "^-", color=colors["ta"], label="Tail-aware")
ax[1].plot(
    df.top_k[positive_k],
    df.bias_normalized[positive_k],
    "s-",
    color=colors["normalized"],
    label="Normalized",
)
ax[1].plot(
    df.top_k, df.bias_corrected_theory, "-", color=colors["ema"], label="EMA theory"
)
ax[1].set(
    xlabel="K", ylabel="signed bias (nats)", title="Bias versus full-vocabulary KL"
)
ax[1].set_yscale("symlog", linthresh=1e-5)
ax[1].grid(alpha=0.25)
ax[2].plot(df.top_k, df.rmse_trunc, "o-", color=colors["trunc"], label="Truncated")
ax[2].plot(df.top_k, df.rmse_tail_aware, "^-", color=colors["ta"], label="Tail-aware")
ax[2].plot(
    df.top_k, df.rmse_corrected_theory, "-", color=colors["ema"], label="EMA theory"
)
ax[2].plot(
    df.top_k,
    df.rmse_corrected_observed,
    "x--",
    color=colors["observed"],
    label="EMA observed (BF16 tokens)",
)
ax[2].plot(
    df.top_k[positive_k],
    df.rmse_normalized[positive_k],
    "s-",
    color=colors["normalized"],
    label="Normalized",
)
ax[2].set(xlabel="K", ylabel="RMSE", title="RMSE versus full-vocabulary KL")
ax[2].set_yscale("log")
ax[2].grid(alpha=0.25)
ax[3].plot(
    df.top_k,
    df.variance_corrected,
    "o-",
    color=colors["ema"],
    label="EMA one-token variance",
)
ax[3].plot(
    df.top_k,
    df.variance_tail_aware,
    "--",
    color=colors["observed"],
    label="Truncated / Normalized / Tail-aware = 0",
)
ax[3].set(xlabel="K", ylabel="variance", title="Conditional estimator variance")
ax[3].grid(alpha=0.25)
ax[3].legend(fontsize=8)
fig.suptitle("Qwen3-1.7B student || Qwen3-4B teacher on GSM8K rollouts")
handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.canvas.draw()
for problem in fv.audit(fig):
    print(problem)
fig.savefig(out / "pair_kl_topk_metrics.png", dpi=220)
fig.savefig(out / "pair_kl_topk_metrics.pdf")
plt.close(fig)
