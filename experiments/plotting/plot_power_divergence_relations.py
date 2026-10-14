"""Draw analytic divergence generators and finite-step MD curvature for the book."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "results/power-divergence-relations"
    output.mkdir(parents=True, exist_ok=True)
    eta = 0.3
    kappa = 0.5
    log_ratio = np.linspace(-8, 8, 601)
    ratio = np.exp(log_ratio)
    power = ratio**kappa
    denominator = 1 - eta + eta * power
    md_curvature = power / denominator
    center_curvature = md_curvature * (1 - kappa * (1 - eta) / denominator)

    teacher_first = np.linspace(0.001, 0.999, 599)
    anchor = np.array([0.7, 0.3])
    teacher = np.column_stack((teacher_first, 1 - teacher_first))
    mixed = (1 - eta) * anchor + eta * teacher
    reverse = np.sum(anchor * np.log(anchor / teacher), axis=1)
    forward = np.sum(teacher * np.log(teacher / anchor), axis=1)
    center = np.sum(anchor * np.log(anchor / mixed), axis=1) / eta
    md = (
        (1 - eta) * np.sum(anchor * np.log(anchor / mixed), axis=1)
        + eta * np.sum(teacher * np.log(teacher / mixed), axis=1)
    ) / (eta * (1 - eta))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), layout="constrained")
    axes[0].plot(log_ratio, power, label=r"$x^2g_\kappa''(x)=x^\kappa$", color="#345995")
    axes[0].plot(log_ratio, md_curvature, label=r"$x^2\omega_{\kappa,\eta}''(x)$", color="#d97706")
    axes[0].plot(
        log_ratio, center_curvature, label=r"$x^2\phi_{\kappa,\eta}''(x)$", color="#228b65"
    )
    axes[0].axhline(1 / eta, color="0.45", linestyle=":", label=r"MD cap $1/\eta$")
    axes[0].set(
        yscale="log",
        xlabel=r"Log probability ratio $\log x$",
        ylabel=r"Rescaled curvature $x^2 f''(x)$",
        title=r"Three generators: $\kappa=1/2,\ \eta=0.3$",
    )
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(alpha=0.16)

    axes[1].plot(
        teacher_first,
        reverse,
        color="0.5",
        linestyle="--",
        label=r"$D_0(T\Vert q)=\mathrm{KL}(q\Vert T)$",
    )
    axes[1].plot(
        teacher_first, forward, color="#345995", label=r"$D_1(T\Vert q)=\mathrm{KL}(T\Vert q)$"
    )
    axes[1].plot(
        teacher_first,
        md,
        color="#d97706",
        label=r"$\mathcal{J}_{1,\eta}(q;T)=\mathrm{JS}_\eta/[\eta(1-\eta)]$",
    )
    axes[1].plot(
        teacher_first, center, color="#228b65", label=r"$K_{1,\eta}=\mathrm{KL}(q\Vert M)/\eta$"
    )
    axes[1].axvline(anchor[0], color="0.65", linestyle=":")
    axes[1].set(
        xlabel=r"Teacher probability $T_1$",
        ylabel="Divergence (nats)",
        title=r"Arithmetic shape: $q=(0.7,0.3),\ \eta=0.3$",
    )
    axes[1].legend(fontsize=8.5, loc="upper right")
    axes[1].grid(alpha=0.16)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"families-and-curvature.{extension}", dpi=190)
    plt.close(fig)
    metadata = {
        "kind": "deterministic_analytic_figure",
        "source": str(Path(__file__).relative_to(root)),
        "eta": eta,
        "curvature_kappa": kappa,
        "arithmetic_anchor": anchor.tolist(),
        "curvature_log_ratio_range": [-8, 8],
        "teacher_probability_range": [0.001, 0.999],
        "interpretation": "Exact scalar formulas; no model inference, training or Monte Carlo",
    }
    (output / "figure.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
