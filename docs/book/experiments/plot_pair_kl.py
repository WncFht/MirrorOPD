#!/usr/bin/env python3
"""Plot aggregate Top-K scalar reverse-KL errors."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--output', required=True)
a = ap.parse_args()
inp = Path(a.input)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(inp)
positive_k = df.top_k > 0
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.ravel()
ax[0].plot(df.top_k, df.mean_topk_mass, 'o-', label='mean student Top-K mass')
ax[0].set(xlabel='K', ylabel='mass', title='Head mass')
ax[0].set_ylim(0, 1.02)
ax[0].grid(alpha=.25)
ax[1].plot(df.top_k, abs(df.bias_trunc), 'o-', label='Trunc |bias|')
ax[1].plot(df.top_k[positive_k], abs(df.bias_normalized[positive_k]), 's-', label='Normalized Top-K |bias|')
ax[1].plot(df.top_k, abs(df.bias_corrected_observed), 'x--', label='Corrected observed |bias|')
ax[1].set(xlabel='K', ylabel='absolute bias', title='Bias versus full-vocabulary KL')
ax[1].set_yscale('log')
ax[1].grid(alpha=.25)
ax[1].legend(fontsize=8)
ax[2].plot(df.top_k, df.rmse_trunc, '--', label='Trunc RMSE')
ax[2].plot(df.top_k, df.rmse_corrected_theory, 'o-', label='Corrected RMSE (theory)')
ax[2].plot(df.top_k, df.rmse_corrected_observed, 'x-', label='Corrected RMSE (observed)')
ax[2].plot(df.top_k[positive_k], df.rmse_normalized[positive_k], 's-', label='Normalized Top-K RMSE')
ax[2].set(xlabel='K', ylabel='RMSE', title='RMSE versus full-vocabulary KL')
ax[2].set_yscale('log')
ax[2].grid(alpha=.25)
ax[2].legend(fontsize=8)
ax[3].plot(df.top_k, df.variance_corrected, 'o-', label='Corrected one-token variance')
ax[3].plot(df.top_k[positive_k], df.variance_normalized[positive_k], 's-', label='Normalized Top-K variance')
ax[3].set(xlabel='K', ylabel='variance', title='Conditional estimator variance')
ax[3].grid(alpha=.25)
ax[3].legend(fontsize=8)
fig.suptitle('Qwen3-1.7B student || Qwen3-4B teacher on GSM8K rollouts')
fig.tight_layout()
fig.savefig(out / 'pair_kl_topk_metrics.png', dpi=220)
fig.savefig(out / 'pair_kl_topk_metrics.pdf')
plt.close(fig)
