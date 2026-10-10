#!/usr/bin/env python3
"""Plot the enumerated loss landscape, using vendored figure-viz tokens."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vendor import fvstyle as fv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/book-reconstruction"


def main():
    theme = fv.apply_rc("paper")
    data = np.genfromtxt(OUT / "landscape.csv", delimiter=",", names=True)
    colors = fv.series_colors("paper", 3)
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for method, color in zip(["part", "mean", "bpo"], colors, strict=True):
        ax.plot(data["logit"], data[method], color=color, label=method.upper())
    reference = ax.axvline(4, color=fv.NEUTRALS["vink"], ls=":", lw=0.8)
    fv.mark_neutral(reference)
    ax.set(xlabel="First logit (second logit = 0)", ylabel="Population loss")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3, frameon=False)
    fv.apply_frame(ax, theme.frame)
    fv.apply_grid(ax, theme.grid)
    fig.tight_layout()
    problems = fv.audit(fig)
    (OUT / "figure-audit.txt").write_text(str(problems) + "\n")
    fig.savefig(OUT / "identities.pdf")
    fig.savefig(OUT / "identities.png", dpi=200)
    print(problems)


if __name__ == "__main__":
    main()
