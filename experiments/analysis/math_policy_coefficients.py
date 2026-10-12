"""Plot actual pre-update score coefficients, their formulas, and inner progress."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from experiments.analysis.math_policy_audit import COLORS, RUNS, finish, load_run, metrics

FORMULAS = {
    "OPD": [
        r"$c^{\mathrm{OPD}}=-(r^{\mathrm{cur}}_{k,n}+b_{k,n}^{pT})$",
        r"$r^{\mathrm{cur}}_{k,n}=\log\pi_T(a|s)-\log\pi_{k,n}(a|s)$",
        r"$b_{k,n}^{pT}=\mathrm{KL}_{\mathrm{TA}}(p\Vert\pi_T)$",
        r"Fresh rollout: $N=1$, $p=q$ at measurement",
    ],
    "MDPO": [
        r"$c^{\mathrm{MDPO}}=\omega_{k,n}(U_{k,n}/\eta-r_k)$",
        r"$U_{k,n}=\log\pi_{k,n}(a|s)-\log\pi_k(a|s)$",
        r"$r_k=\log\pi_T(a|s)-\log\pi_k(a|s)$",
        r"$\omega_{k,n}=\exp U_{k,n}$, $\eta=0.3$",
    ],
    "BPO": [
        r"$c^{\mathrm{BPO}}=(U_{k,n}+\widehat{\mathcal{K}}_{k,n}^{S})/\eta$",
        r"$\quad-r_k-\widehat{\mathcal{K}}_k^T$",
        r"$\widehat{\mathcal{K}}^S=\mathrm{KL}_{\mathrm{TA}}(q\Vert p)$",
        r"$\widehat{\mathcal{K}}^T=\mathrm{KL}_{\mathrm{TA}}(q\Vert\pi_T)$; no IS",
    ],
}


def run_plots(raw, bpo_raw, output):
    """Compare fixed completed outers; coefficient means are token weighted."""
    output.mkdir(parents=True, exist_ok=True)
    runs = {m: load_run(bpo_raw if m == "BPO" else raw, name) for m, name in RUNS.items()}
    common = min(r["completed"] for r in runs.values())
    selected = sorted({0, min(4, common - 1), min(9, common - 1), min(11, common - 1), common - 1})
    summary = {"common_outer_count": common, "selected_k": selected, "methods": {}}

    fig, axes = plt.subplots(3, 4, figsize=(22, 10))
    for row, (method, run) in enumerate(runs.items()):
        axes[row, 0].axis("off")
        axes[row, 0].set_title(method + ": descent coefficient", loc="left")
        for i, formula in enumerate(FORMULAS[method]):
            axes[row, 0].text(
                0, 0.88 - 0.19 * i, formula, fontsize=11, transform=axes[row, 0].transAxes
            )
        group = run["groups"][0]
        x = np.arange(run["N"])
        mean = metrics(group, "score_coefficient/mean")
        std = metrics(group, "score_coefficient/std")
        axes[row, 1].plot(x, mean, "o-", label=r"$\bar{c}$", color=COLORS[method])
        axes[row, 1].fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[method],
            alpha=0.14,
            label=r"$\bar{c}\pm\mathrm{std}(c)$ over tokens",
        )
        axes[row, 1].axhline(0, color="gray", linewidth=0.7)
        axes[row, 1].set(ylabel="Score coefficient, k=0", xlabel="Pre-update n")
        axes[row, 2].plot(
            x,
            metrics(group, "score_coefficient/rms"),
            "o-",
            color=COLORS[method],
            label=r"$\sqrt{\overline{c^2}}$",
        )
        axes[row, 2].set(ylabel="Coefficient RMS, k=0", xlabel="Pre-update n")
        axes[row, 3].plot(
            x,
            [r["metrics"]["grad_norm"] for r in group],
            "o-",
            color=COLORS[method],
            label=r"$\Vert g\Vert_2$",
        )
        axes[row, 3].set(ylabel="Pre-clip gradient norm, k=0", xlabel="Pre-update n")
        if run["N"] == 1:
            axes[row, 1].errorbar(x, mean, yerr=std, fmt="none", color=COLORS[method], capsize=4)
            for ax in axes[row, 1:]:
                ax.set_xticks([0])
    fig.suptitle(
        r"Actual loss: $\ell=\mathrm{sg}(c)\log p$; ascent advantage $A=-c$. Last n=15 is before update 16."
    )
    finish(fig, axes, output, "17-score-formulas-first-outer")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col, (method, run) in enumerate(runs.items()):
        for k in sorted({*selected, run["completed"] - 1}):
            group = run["groups"][k]
            axes[0, col].plot(
                np.arange(run["N"]),
                metrics(group, "score_coefficient/rms"),
                "o-",
                markersize=3,
                label=f"k={k}",
            )
            axes[1, col].plot(
                np.arange(run["N"]),
                [r["metrics"]["grad_norm"] for r in group],
                "o-",
                markersize=3,
                label=f"k={k}",
            )
        axes[0, col].set(title=method, xlabel="Pre-update n", ylabel=r"$\sqrt{\overline{c^2}}$")
        axes[1, col].set(title=method, xlabel="Pre-update n", ylabel=r"$\Vert g\Vert_2$")
        if run["N"] == 1:
            axes[0, col].set_xticks([0])
            axes[1, col].set_xticks([0])
    finish(fig, axes, output, "18-selected-outers-coefficients")

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for col, (method, run) in enumerate(runs.items()):
        groups = run["groups"][:common]
        c = np.asarray([metrics(g, "score_coefficient/rms") for g in groups])
        grad = np.asarray([[r["metrics"]["grad_norm"] for r in g] for g in groups])
        cmean = np.asarray([metrics(g, "score_coefficient/mean") for g in groups])
        summary["methods"][method] = {
            "available_outer_count": run["completed"],
            "N": run["N"],
            "common_window_c_rms_first_mean": float(c[:, 0].mean()),
            "common_window_c_rms_last_mean": float(c[:, -1].mean()),
            "common_window_grad_first_mean": float(grad[:, 0].mean()),
            "common_window_grad_last_mean": float(grad[:, -1].mean()),
            "c_rms_last_over_first_per_outer": (c[:, -1] / c[:, 0]).tolist(),
            "grad_last_over_first_per_outer": (grad[:, -1] / grad[:, 0]).tolist(),
            "grad_max": float(grad.max()),
            "clipped_updates": sum(r["metrics"]["grad_was_clipped"] for g in groups for r in g),
        }
        for row, matrix, label in [
            (0, c / c[:, :1], r"$\mathrm{RMS}(c_{k,n})/\mathrm{RMS}(c_{k,0})$"),
            (1, grad / grad[:, :1], r"$\Vert g_{k,n}\Vert/\Vert g_{k,0}\Vert$"),
            (2, cmean, r"$\bar{c}_{k,n}$"),
        ]:
            axes[row, col].plot(
                np.arange(run["N"]),
                matrix.mean(axis=0),
                "o-",
                color=COLORS[method],
                markersize=3,
                label=f"k=0..{common - 1} mean",
            )
            axes[row, col].fill_between(
                np.arange(run["N"]),
                matrix.min(axis=0),
                matrix.max(axis=0),
                color=COLORS[method],
                alpha=0.13,
                label="Across-outer range",
            )
            axes[row, col].set(title=method, xlabel="Pre-update n", ylabel=label)
    finish(fig, axes, output, "19-common-window-inner-progress")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for col, (method, run) in enumerate(runs.items()):
        groups = run["groups"]
        c = np.asarray([metrics(g, "score_coefficient/rms") for g in groups])
        g = np.asarray([[r["metrics"]["grad_norm"] for r in group] for group in groups])
        for row, matrix, label in [
            (0, c, "Coefficient RMS / own n=0"),
            (1, g, "Gradient norm / own n=0"),
        ]:
            ratio = matrix / matrix[:, :1]
            image = axes[row, col].imshow(
                ratio,
                aspect="auto",
                origin="lower",
                norm=LogNorm(vmin=min(0.1, float(ratio.min())), vmax=max(1.4, float(ratio.max()))),
                cmap="viridis",
            )
            axes[row, col].set(
                title=method + ": " + label, xlabel="Pre-update n", ylabel="Frozen rollout k"
            )
            fig.colorbar(image, ax=axes[row, col])
    finish(fig, axes, output, "20-all-outers-inner-progress")

    bpo = runs["BPO"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, k in zip(axes, [0, common - 1], strict=True):
        group = bpo["groups"][k]
        pieces = {
            r"$\bar{U}/\eta$": metrics(group, "sampled_log_ratio/mean") / 0.3,
            r"$\overline{\widehat{\mathcal{K}}^S}/\eta$": metrics(group, "student_center/mean")
            / 0.3,
            r"$-\bar{r}_k$": -metrics(group, "sampled_reward/mean"),
            r"$-\overline{\widehat{\mathcal{K}}^T}$": -metrics(group, "teacher_center/mean"),
        }
        expected = np.sum(list(pieces.values()), axis=0)
        np.testing.assert_allclose(expected, metrics(group, "score_coefficient/mean"), atol=1e-6)
        for name, values in pieces.items():
            ax.plot(np.arange(16), values, label=name)
        ax.plot(
            np.arange(16), expected, "k--", linewidth=2, label=r"Sum = $\bar{c}^{\mathrm{BPO}}$"
        )
        ax.set(
            title=f"BPO k={k}: actual logged components",
            xlabel="Pre-update n",
            ylabel="Mean coefficient component",
        )
    finish(fig, axes, output, "21-bpo-coefficient-decomposition")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for col, benchmark in enumerate(["MATH-500", "AMC23", "AIME25"]):
        for method, run in runs.items():
            rows = run["evaluation"]
            axes[col].plot(
                [r["step"] for r in rows],
                [100 * r["data"][f"val-core/{benchmark}/acc/mean@1"] for r in rows],
                "o-",
                color=COLORS[method],
                label=method,
                markersize=3,
            )
        axes[col].set(title=benchmark, xlabel="Completed outer", ylabel="Greedy accuracy (%)")
    fig.suptitle(
        f"BPO: {runs['BPO']['completed']} committed outers; evaluations only at recorded checkpoints"
    )
    finish(fig, axes, output, "22-latest-quality")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method, run in runs.items():
        values = np.asarray([metrics(g, "importance_ess_fraction") for g in run["groups"]])
        x = np.arange(run["completed"])
        axes[0].plot(x, 100 * (1 - values.min(axis=1)), color=COLORS[method], label=method)
        axes[1].plot(
            x,
            [metrics(g, "importance_ratio/max").max() for g in run["groups"]],
            color=COLORS[method],
            label=method,
        )
        axes[2].plot(
            x,
            [metrics(g, "sampled_log_ratio/rms")[-1] for g in run["groups"]],
            color=COLORS[method],
            label=method,
        )
    axes[0].set(xlabel="Frozen rollout k", ylabel=r"$100(1-\min_n f_{k,n})$ (%)")
    axes[1].set(xlabel="Frozen rollout k", ylabel=r"$\max_{n,j}\omega_j$")
    axes[2].set(xlabel="Frozen rollout k", ylabel=r"$\sqrt{\overline{U_{k,N-1}^2}}$")
    finish(fig, axes, output, "23-latest-is-ess")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method, ax in zip(["MDPO", "BPO"], axes, strict=True):
        run = runs[method]
        for k in sorted({*selected, run["completed"] - 1}):
            group = run["groups"][k]
            if method == "MDPO":
                y = metrics(group, "score_coefficient/mean")
            else:
                y = 0.3 / 2 * metrics(group, "score_coefficient/rms") ** 2
            ax.plot(np.arange(16), y, "o-", markersize=3, label=f"k={k}")
        ax.set(title=method, xlabel="Pre-update n")
    axes[0].set_ylabel(r"$\widehat{L}_{\mathrm{MDPO,IS}}=\bar{c}$")
    axes[1].set_ylabel(r"BPO TA residual proxy: $\eta\overline{c^2}/2$")
    finish(fig, axes, output, "24-inner-objective-estimates")
    (output / "coefficient-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    """Read completed-run baselines and an explicitly selected BPO snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--bpo-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_plots(args.raw, args.bpo_raw, args.output), indent=2))


if __name__ == "__main__":
    main()
