"""Historical OPD effective-time fit, held-out curves, and LR failure endpoints."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.analysis.oneshot_lr_stability import RUN_IDS, curve


def plot(inputs, output):
    results = json.loads((output / "analysis.json").read_text())
    fit = results["reference_fit"]
    parameters = np.log([fit["floor"], *fit["amplitudes"], *fit["rates_per_u"]])
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, run_id in enumerate(RUN_IDS):
        frame = pd.read_csv(inputs / f"{run_id}.csv")
        lr = frame["lr"].iloc[0]
        step = frame["step"].to_numpy() - 1
        u = step * lr / 1e-6
        distance = frame["distance"].to_numpy()
        name = f"lr {lr:.2g}" + (" (fit)" if run_id in fit["fit_run_ids"] else " (held out)")
        color = colors[index]
        axes[0, 0].semilogy(step, distance, color=color, linewidth=1, alpha=0.85, label=name)
        visible = u <= 600
        axes[0, 1].semilogy(u[visible], distance[visible], color=color, linewidth=1, alpha=0.85)
        norms = frame["grad_norm"]
        axes[1, 0].semilogy(step, norms.rolling(20, min_periods=1).median(), color=color)
        axes[1, 1].semilogy(u[visible], norms.to_numpy()[visible], color=color, alpha=0.65)
        valid = (u >= 30) & visible
        axes[2, 0].semilogy(u[valid], distance[valid] / curve(parameters, u[valid]), color=color)
    fit_u = np.linspace(30, fit["fit_max_u"], 200)
    extended_u = np.linspace(fit["fit_max_u"], 600, 200)
    axes[0, 1].semilogy(
        fit_u, curve(parameters, fit_u), color="black", linewidth=2, label="low-LR fitted model"
    )
    axes[0, 1].semilogy(
        extended_u,
        curve(parameters, extended_u),
        color="black",
        linewidth=2,
        linestyle="--",
        label="model extrapolation",
    )
    axes[0, 1].legend(fontsize=8)
    for row in (0, 1, 2):
        column = 0 if row == 2 else 1
        axes[row, column].axvspan(fit["fit_max_u"], 600, color="gray", alpha=0.08)
        axes[row, column].set_xlim(0, 600)
    for axis in axes[1]:
        axis.axhline(1, color="black", linestyle=":", linewidth=1)
    axes[2, 0].axhline(1, color="black", linestyle=":", linewidth=1)
    rows = results["runs"]
    axes[2, 1].loglog(
        [r["lr"] for r in rows],
        [r["distance_tail_median30"] for r in rows],
        marker="o",
        color="black",
    )
    axes[2, 1].axvspan(1e-5, 1e-4, color="gray", alpha=0.15, label="untested rates")
    axes[2, 1].axvline(
        fit["hypothetical_two_mode_GD_upper_lr"],
        color="purple",
        linestyle="--",
        label="two-mode GD limit; fails as AdamW guarantee",
    )
    axes[2, 1].legend(fontsize=8)
    axes[0, 0].legend(fontsize=8, ncol=2)
    titles = (
        (
            "Sampled reverse-KL proxy vs update count",
            "Effective time: low-LR fit, held-out LR tests",
        ),
        (
            "Pre-clip grad norm: trailing-20-step median",
            "Pre-clip grad norm vs effective time (raw)",
        ),
        ("Observed proxy / low-LR reference", "Late distance vs LR; untested interval retained"),
    )
    labels = (
        ("Completed updates before rollout", "u = lr / 1e-6 x completed updates"),
        ("Completed updates before rollout", "Effective time u"),
        ("Effective time u", "Constant learning rate"),
    )
    for row in range(3):
        for column in range(2):
            axes[row, column].set_title(titles[row][column])
            axes[row, column].set_xlabel(labels[row][column])
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("d = -sampled token reward mean")
    axes[1, 0].set_ylabel("Gradient norm")
    axes[2, 0].set_ylabel("Observed / reference")
    axes[2, 1].set_ylabel("Last-30-step median distance")
    fig.suptitle(
        "Historical One-Shot-OPD project M4: online multi-prompt DAPO, 64 rollouts per update\nBF16, AdamW, response cap 7168; 279 logged steps per run; all analysis on CPU"
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"effective-time-and-lr-boundary.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    models = results["stage_velocity_models"]
    for column, band in enumerate(([0.05, 0.02], [0.02, 0.01])):
        chosen = [m for m in models if m["effective_window"] == 30 and m["distance_band"] == band]
        x = np.linspace(0, 30, 300)
        for model in chosen:
            first_fit = model["phase"] == "held_out_1e-5_and_1e-4"
            axes[column].plot(
                x,
                model["a"] * x - model["b"] * x * x,
                linestyle="--" if first_fit else "-",
                color="blue",
                label="fit up to 5e-6" if first_fit else "refit up to 1e-5",
            )
        points = chosen[-1]["measurements"]
        axes[column].scatter(
            [p["lr"] / 1e-6 for p in points if p["lr"] < 1e-5],
            [p["rate"] for p in points if p["lr"] < 1e-5],
            color="black",
            label="calibration rates",
        )
        point = next(p for p in points if p["lr"] >= 1e-5)
        axes[column].scatter(
            [point["lr"] / 1e-6],
            [point["rate"]],
            color="orange",
            marker="D",
            label="initially held-out 1e-5",
        )
        zero = chosen[-1]["candidate_zero_rate_lr"] / 1e-6
        axes[column].axvline(zero, color="blue", alpha=0.4)
        axes[column].axhline(0, color="black", linewidth=0.8)
        axes[column].set_ylim(
            -max(p["rate"] for p in points) * 0.5, max(p["rate"] for p in points) * 1.4
        )
        axes[column].set_title(f"Distance {band[0]} to {band[1]}; refit zero at {zero:.1f}e-6")
        axes[column].set_xlabel("Learning rate / 1e-6")
        axes[column].set_ylabel("Stage-average absolute proxy decrease / update")
        axes[column].grid(alpha=0.2)
        axes[column].legend(fontsize=8)
    fig.suptitle(
        "Stage-wise empirical LR response; trailing effective-time window 30\nLR=1e-4 never reaches the lower band endpoints; candidate zero points remain untested"
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"stage-velocity-lr-candidates.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.inputs, args.output)


if __name__ == "__main__":
    main()
