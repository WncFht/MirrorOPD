"""Plot analytic shape criteria and verify the reconstructed chapter's new identities."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from mirroropd.proximal import (
    minimax_velocity_shape,
    power_log_target,
    power_signal,
    worst_fisher_speed_squared,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def target(q, teacher, kappa, eta):
    return (
        power_log_target(
            torch.tensor(np.log(q), dtype=torch.float64),
            torch.tensor(np.log(teacher), dtype=torch.float64),
            kappa,
            eta,
        )
        .exp()
        .numpy()
    )


def affinity(q, teacher):
    return float(np.sqrt(q) @ np.sqrt(teacher))


def hellinger(q, teacher):
    return float(np.sum((np.sqrt(q) - np.sqrt(teacher)) ** 2))


def supervision_variance(q, teacher, kappa, eta):
    a = power_signal(torch.tensor(np.log(teacher / q)), kappa, eta).numpy() / eta
    return float(q @ (a - q @ a) ** 2)


def verify():
    rng = np.random.default_rng(20261014)
    maximum_affinity_error = 0.0
    maximum_local_remainder = 0.0
    maximum_local_rate_error = 0.0
    cases = 0
    for size in (3, 17):
        teacher = rng.dirichlet(np.full(size, 2.0))
        q = rng.dirichlet(np.full(size, 2.0))
        tilt = rng.uniform(-1, 1, size)
        h = teacher * (tilt - teacher @ tilt)
        quadratic = float(np.sum(h**2 / teacher))
        for eta in (0.1, 0.3, 0.7):
            root_target = target(q, teacher, 0.5, eta)
            overlap = affinity(q, teacher)
            z = (1 - eta) ** 2 + eta**2 + 2 * eta * (1 - eta) * overlap
            exact = ((1 - eta) * overlap + eta) / np.sqrt(z)
            maximum_affinity_error = max(
                maximum_affinity_error, abs(affinity(root_target, teacher) - exact)
            )
            assert hellinger(root_target, teacher) <= (1 - eta) * hellinger(q, teacher)
            for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
                probe = target(q, teacher, kappa, eta)
                if kappa > 0:
                    floor = (1 - eta) ** (1 / kappa) * q + eta ** (1 / kappa) * teacher
                    assert np.all(probe >= floor - 1e-14)
                if 0 < kappa <= 0.5:
                    variance = supervision_variance(q, teacher, kappa, eta)
                    overlap_power = float(q @ (teacher / q) ** kappa)
                    divergence = (1 - overlap_power) / (kappa * (1 - kappa))
                    assert variance <= divergence / (kappa * (1 - eta) ** 2) + 1e-12
                epsilon = 1e-4
                anchor = teacher + epsilon * h
                m = target(anchor, teacher, kappa, eta)
                approximation = (
                    teacher
                    + (1 - eta) * epsilon * h
                    + (kappa - 1)
                    * (1 - eta)
                    * eta
                    * epsilon**2
                    * (h**2 / teacher - teacher * quadratic)
                    / 2
                )
                remainder = float(np.max(np.abs(m - approximation)) / epsilon**3)
                maximum_local_remainder = max(maximum_local_remainder, remainder)
                assert remainder < 0.1
                rate = hellinger(m, teacher) / hellinger(anchor, teacher)
                maximum_local_rate_error = max(maximum_local_rate_error, abs(rate - (1 - eta) ** 2))
                assert abs(rate - (1 - eta) ** 2) < 1e-4
                cases += 1
    moment_checks = []
    for order in (2, 4):
        for kappa in (0.25, 0.5, 0.75, 1.0):
            moments = []
            for eta in (1e-5, 1e-6):
                if order * kappa < 1:
                    q, teacher = np.array([0.7, 0.3]), np.array([0.02, 0.98])
                else:
                    q = np.array([eta ** (1 / kappa) / 2, 1 - eta ** (1 / kappa) / 2])
                    teacher = np.array([0.5, 0.5])
                a = power_signal(torch.tensor(np.log(teacher / q)), kappa, eta).numpy() / eta
                moments.append(float(q @ np.abs(a - q @ a) ** order))
            slope = float(np.log(moments[1] / moments[0]) / np.log(0.1))
            expected = min(order, 1 / kappa) - order
            assert abs(slope - expected) < 0.003, (order, kappa, slope, expected)
            moment_checks.append(
                {"kappa": kappa, "order": order, "slope": slope, "expected": expected}
            )
    sharp_ratios = []
    for kappa in (0.25, minimax_velocity_shape(), 0.5):
        v = 1e-20
        complement = v * (1 - v**kappa) - (1 - v) * np.expm1(kappa * np.log1p(v))
        variance = v * (1 - v) * (v**kappa - (1 + v) ** kappa) ** 2 / kappa**2
        ratio_d = variance / (complement / (kappa * (1 - kappa)))
        ratio_k = variance / (complement / kappa)
        assert abs(ratio_d - 1 / kappa) < 1e-4
        assert abs(ratio_k - 1 / (kappa * (1 - kappa))) < 2e-4
        sharp_ratios.append(
            {
                "kappa": kappa,
                "witness_mass": v,
                "variance_over_divergence": ratio_d,
                "sharp_divergence_constant": 1 / kappa,
                "variance_over_center": ratio_k,
            }
        )
    return {
        "deterministic_local_cases": cases,
        "maximum_affinity_identity_error": maximum_affinity_error,
        "maximum_target_remainder_divided_by_epsilon_cubed": maximum_local_remainder,
        "maximum_local_rate_error_at_epsilon_1e_minus_4": maximum_local_rate_error,
        "moment_witness_checks": moment_checks,
        "sharp_relative_noise_witnesses": sharp_ratios,
        "random_vector_seed": 20261014,
        "arithmetic": "CPU float64; exact finite-alphabet sums, no Monte Carlo",
    }


def plot_relative_noise(out):
    eta = 0.3
    teacher = np.array([0.05, 0.95])
    q = np.array([0.999, 0.001])
    initial = hellinger(q, teacher)
    path = []
    for n in range(26):
        error = hellinger(q, teacher)
        variance = supervision_variance(q, teacher, 0.5, eta)
        error_bound = 4 * error / (1 - eta) ** 2
        geometric_bound = 4 * (1 - eta) ** n * initial / (1 - eta) ** 2
        assert variance <= error_bound + 1e-12
        assert error_bound <= geometric_bound + 1e-12
        path.append(
            {
                "outer": n,
                "hellinger": error,
                "variance": variance,
                "error_bound": error_bound,
                "geometric_bound": geometric_bound,
            }
        )
        q = target(q, teacher, 0.5, eta)
    assert sum(row["variance"] for row in path) <= 4 * initial / (eta * (1 - eta) ** 2)
    shape = np.linspace(0.08, 0.5, 300)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), layout="constrained")
    axes[0].plot(
        shape, 1 / shape, color="#345995", label=r"Sharp $V^0/D_\kappa$ ceiling: $1/\kappa$"
    )
    axes[0].plot(
        shape,
        1 / (shape * (1 - shape)),
        color="#d97706",
        label=r"Sharp $V^0/K_\kappa^0$ ceiling: $1/[\kappa(1-\kappa)]$",
    )
    axes[0].scatter([0.5, 0.5], [2, 4], color="#228b65", zorder=3)
    axes[0].set(
        xlabel=r"Shape $\kappa$",
        ylabel="Variance / teacher-error constant",
        title=r"Variance / error minimum: $\kappa=1/2$",
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.18)
    for key, label, color, style in [
        ("variance", "Actual supervision variance", "#345995", "-"),
        ("error_bound", r"Error bound $4H^2(q_n,T)/(1-\eta)^2$", "#228b65", "--"),
        ("geometric_bound", "Uniform geometric envelope", "#d97706", ":"),
    ]:
        axes[1].plot(
            [row["outer"] for row in path],
            [row[key] for row in path],
            label=label,
            color=color,
            linestyle=style,
        )
    axes[1].set(
        xlabel="Exact outer update n",
        ylabel="Supervision variance or its bound",
        yscale="log",
        title=r"Variance envelope: $\kappa=1/2,\ \eta=0.3$",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.18)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"relative-noise-and-convergence.{extension}", dpi=190)
    plt.close(fig)
    return {
        "initial_anchor": [0.999, 0.001],
        "teacher": teacher.tolist(),
        "eta": eta,
        "path": path,
        "variance_increased_after_first_step": path[1]["variance"] > path[0]["variance"],
        "cumulative_uniform_bound": 4 * initial / (eta * (1 - eta) ** 2),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "results/chapter9-restructure"
    out.mkdir(parents=True, exist_ok=True)
    checks = verify()
    optimum = minimax_velocity_shape()
    shape = np.linspace(0.015, 0.5, 450)
    ceiling = np.array([worst_fisher_speed_squared(float(k)) for k in shape])
    grid = np.linspace(0, 1, 401)
    pairs = [
        (np.array([0.9, 0.1]), np.array([0.01, 0.99])),
        (np.array([0.99, 0.01]), np.array([0.5, 0.5])),
    ]
    eta = 0.3
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), layout="constrained")
    axes[0].plot(shape, ceiling, color="#345995")
    axes[0].scatter(
        [optimum, 0.5],
        [worst_fisher_speed_squared(optimum), 4],
        color=["#d97706", "#228b65"],
        zorder=3,
    )
    axes[0].axvline(optimum, color="#d97706", linestyle=":")
    axes[0].annotate(
        "0.4419: 3.5273",
        (optimum, worst_fisher_speed_squared(optimum)),
        xytext=(0.19, 5.8),
        arrowprops={"arrowstyle": "->"},
    )
    axes[0].set(
        xlabel=r"Shape $\kappa$",
        ylabel="Sharp worst-case small-step variance",
        ylim=(3.2, 10),
        title="Minimum worst-case variance: 0.4419",
    )
    axes[0].grid(alpha=0.18)
    examples = []
    for axis, (q, teacher) in zip(axes[1:], pairs, strict=True):
        values = [hellinger(target(q, teacher, float(k), eta), teacher) for k in grid]
        axis.plot(grid, values, color="#345995")
        points = [hellinger(target(q, teacher, k, eta), teacher) for k in (0, 0.5, 1)]
        axis.scatter([0, 0.5, 1], points, color=["#7b4ab5", "#228b65", "#d97706"], zorder=3)
        axis.axvline(0.5, color="#228b65", linestyle=":")
        axis.set(
            xlabel=r"Shape $\kappa$",
            ylabel=r"Teacher error $H^2(m_{\kappa,\eta},T)$",
            title=rf"$q_1={q[0]:.2f},\ T_1={teacher[0]:.2f},\ \eta=0.3$",
        )
        axis.grid(alpha=0.18)
        examples.append(
            {
                "anchor": q.tolist(),
                "teacher": teacher.tolist(),
                "eta": eta,
                "hellinger_after_step_kappa_0_half_1": points,
            }
        )
    for extension in ("pdf", "png"):
        fig.savefig(out / f"kappa-criteria.{extension}", dpi=190)
    plt.close(fig)
    noise_path = plot_relative_noise(out)
    result = {
        "kind": "analytic_chapter_reconstruction",
        "source": str(Path(__file__).relative_to(root)),
        "minimax_shape": optimum,
        "minimax_variance": worst_fisher_speed_squared(optimum),
        "half_variance_ceiling": 4.0,
        "counterexamples": examples,
        "checks": checks,
        "square_root_noise_path": noise_path,
        "new_gpu_runs": 0,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
