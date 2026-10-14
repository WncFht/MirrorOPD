"""Plot the power-divergence generators and their binary-distribution distances."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/fanghaotian/.agents/skills/figure-viz/assets/mpl")
import fvstyle as fv

OUT = Path(__file__).resolve().parent
KAPPAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def generator(x: np.ndarray, kappa: float) -> np.ndarray:
    if kappa == 0:
        return x - 1.0 - np.log(x)
    if kappa == 1:
        return x * np.log(x) - x + 1.0
    return (kappa * x + 1.0 - kappa - x**kappa) / (kappa * (1.0 - kappa))


def divergence_binary(u: np.ndarray, kappa: float) -> np.ndarray:
    q = np.array([0.7, 0.3])
    p = np.stack([u, 1.0 - u], axis=-1)
    if kappa == 0:
        return np.sum(q * np.log(q / p), axis=-1)
    if kappa == 1:
        return np.sum(p * np.log(p / q), axis=-1)
    affinity = np.sum(p**kappa * q ** (1.0 - kappa), axis=-1)
    return (1.0 - affinity) / (kappa * (1.0 - kappa))


def main() -> None:
    theme = fv.apply_rc("paper")
    colors = fv.series_colors("paper", len(KAPPAS))
    x = np.geomspace(0.03, 12.0, 700)
    u = np.linspace(0.005, 0.995, 700)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    ax = axes[0]
    for color, kappa in zip(colors, KAPPAS, strict=True):
        ax.plot(x, generator(x, kappa), color=color, lw=2.0, label=rf"$\kappa={kappa:g}$")
    ax.axvline(1.0, color=fv.NEUTRALS["baseline"], ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"ratio $x=P_a/Q_a$")
    ax.set_ylabel(r"generator $g_\kappa(x)$")
    ax.set_title("Generator shape")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fv.apply_frame(ax, theme.frame)
    fv.apply_grid(ax, theme.grid)

    ax = axes[1]
    for color, kappa in zip(colors, KAPPAS, strict=True):
        ax.plot(u, divergence_binary(u, kappa), color=color, lw=2.0, label=rf"$\kappa={kappa:g}$")
    ax.set_xlabel(r"first component $P_1$ (with $Q=(0.7,0.3)$)")
    ax.set_ylabel(r"$D_\kappa(P\Vert Q)$")
    ax.set_title("Binary-distribution divergence")
    fv.apply_frame(ax, theme.frame)
    fv.apply_grid(ax, theme.grid)

    fig.text(
        0.5,
        0.01,
        "kappa=0: reverse KL; kappa=1/2: 2 x squared Hellinger; kappa=1: forward KL",
        ha="center",
        color=fv.HUES["vgray"],
        fontsize=8,
    )
    fig.savefig(OUT / "power-divergence-family.pdf", bbox_inches="tight")
    fig.savefig(OUT / "power-divergence-family.png", dpi=220, bbox_inches="tight")
    problems = fv.audit(fig)
    for problem in problems:
        print(f"audit: {problem}")


if __name__ == "__main__":
    main()
