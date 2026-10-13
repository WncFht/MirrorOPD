"""Verify the single-rollout BPO objective and distinguish its target from fixed-loss MD."""

import json
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log_normalize(z):
    top = np.max(z)
    return z - top - np.log(np.exp(z - top).sum())


def target(q, teacher, shape, strength):
    r = np.log(teacher / q)
    if strength == 1:
        return teacher.copy(), r
    f = (
        strength * r
        if shape == 0
        else np.logaddexp(np.log1p(-strength), np.log(strength) + shape * r) / shape
    )
    return np.exp(log_normalize(np.log(q) + f)), f


def objective(z, q, m, eta):
    u = z - np.log(m)
    residual = u - q @ u
    return (q @ residual**2) / (2 * eta)


def save(fig, directory, name):
    for ext in ("png", "pdf"):
        fig.savefig(
            directory / f"{name}.{ext}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None,
        )
    plt.close(fig)


def pipeline(directory):
    fig, ax = plt.subplots(figsize=(12, 3.5), layout="constrained")
    ax.axis("off")
    texts = [
        "Freeze anchor q\nOne rollout per prompt\nFreeze prefix weights",
        "Freeze teacher signal\nf = log(1-eta+eta exp(k r))/k\nk=0: f=eta*r\nTarget m proportional to q exp(f)",
        "BPO inner regression\nR = centered_q(log(p/q)-f)\nL = E_q[R squared]/(2 eta)",
        "Update student\nRefresh q after I steps\nNew rollout next outer step",
    ]
    centers = [0.11, 0.37, 0.64, 0.90]
    for x, text in zip(centers, texts, strict=True):
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=.8", "fc": "#edf4f8", "ec": "#315e7b"},
        )
    for a, b in pairwise(centers):
        ax.annotate(
            "",
            xy=(b - 0.105, 0.55),
            xytext=(a + 0.11, 0.55),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#315e7b"},
        )
    ax.text(
        0.5,
        0.12,
        "The rollout selects states. Exact action expectations require no additional response or group.",
        ha="center",
        transform=ax.transAxes,
        fontsize=11,
    )
    save(fig, directory, "pipeline")


def geometry(directory):
    q, teacher = np.array([0.75, 0.23, 0.02]), np.array([0.06, 0.24, 0.70])
    eta = 0.3
    covariance = np.diag(q) - np.outer(q, q)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    project = np.array([[0, 1, 0.5], [0, 0, np.sqrt(3) / 2]])
    triangle = project @ np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    axes[0].plot(*triangle, color="#aaaaaa")
    for kappa, color in ((0, "#3979ae"), (0.5, "#289b78"), (1, "#ce7944")):
        curve = np.array(
            [target(q, teacher, kappa, e)[0] for e in np.linspace(0.00001, 0.99999, 101)]
        )
        axes[0].plot(*(project @ curve.T), label=f"Power target k={kappa}", color=color)
        point = project @ target(q, teacher, kappa, eta)[0]
        axes[0].scatter(*point, color=color)
        m = target(q, teacher, kappa, eta)[0]
        zs = np.linspace(-2, 2, 101)
        direction = np.array([1.0, -1.0, 0.0])
        curvature = direction @ covariance @ direction / eta
        losses = [objective(np.log(m) + v * direction, q, m, eta) for v in zs]
        axes[1].plot(
            zs,
            losses,
            label=f"k={kappa}; curvature={curvature:.4f}",
            color=color,
            linestyle=("-", "--", ":")[(0, 0.5, 1).index(kappa)],
        )
    tangent = ((teacher / q) ** 0.5 - 1) / 0.5
    curve = np.array(
        [np.exp(log_normalize(np.log(q) + e * tangent)) for e in np.linspace(0, 1, 101)]
    )
    axes[0].plot(*(project @ curve.T), "--", color="#955aa0", label="Entropy MD of fixed D_0.5")
    axes[0].scatter(*(project @ np.stack([q, teacher]).T), color="black", s=25)
    axes[0].set(
        title="Finite-step targets differ; initial tangents agree",
        xlabel="Simplex coordinate",
        ylabel="Simplex coordinate",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        title="BPO curvature does not depend on target shape",
        xlabel="Logit displacement from each target",
        ylabel="Exact BPO loss",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, directory, "geometry-and-curvature")


def finite_inner_checks(directory):
    rng = np.random.default_rng(20261014)
    checks = 0
    worst = 0.0
    for dimension in (3, 17, 127):
        for kappa in (0.0, 0.25, 0.5, 1.0):
            for eta in (0.01, 0.3, 0.7, 1.0):
                q = np.exp(log_normalize(3 * rng.normal(size=dimension)))
                teacher = np.exp(log_normalize(3 * rng.normal(size=dimension)))
                m, f = target(q, teacher, kappa, eta)
                initial = objective(np.log(q), q, m, eta)
                history = []
                for beta in np.linspace(0, 1, 101):
                    logp = log_normalize(np.log(q) + beta * f)
                    p = np.exp(logp)
                    history.append(
                        [
                            p @ (logp - np.log(teacher)),
                            teacher @ (np.log(teacher) - logp),
                            0.5 * np.abs(p - teacher).sum(),
                            ((np.sqrt(p) - np.sqrt(teacher)) ** 2).sum(),
                        ]
                    )
                    predicted = (1 - beta) ** 2 * initial
                    np.testing.assert_allclose(
                        objective(logp, q, m, eta), predicted, atol=1e-12, rtol=1e-10
                    )
                increase = float(np.diff(np.asarray(history), axis=0).max())
                worst = max(worst, increase)
                assert increase < 1e-10
                checks += 1
    q, teacher = np.array([0.75, 0.23, 0.02]), np.array([0.06, 0.24, 0.70])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for kappa in (0.0, 0.5, 1.0):
        m, f = target(q, teacher, kappa, 0.3)
        beta = np.linspace(0, 1, 101)
        densities = np.array([np.exp(log_normalize(np.log(q) + b * f)) for b in beta])
        axes[0].plot(beta, [p @ np.log(p / teacher) for p in densities], label=f"k={kappa}")
        axes[1].plot(beta, [teacher @ np.log(teacher / p) for p in densities], label=f"k={kappa}")
        losses = [
            objective(np.log(p), q, m, 0.3) / objective(np.log(q), q, m, 0.3) for p in densities
        ]
        axes[2].plot(beta, losses, label=f"k={kappa}")
    for ax, title in zip(
        axes,
        ("Reverse KL to teacher", "Forward KL to teacher", "BPO loss / initial = (1-beta)^2"),
        strict=True,
    ):
        ax.set(xlabel="beta: progress toward frozen target", title=title)
        ax.legend()
        ax.grid(alpha=0.15)
    save(fig, directory, "finite-inner-progress")
    return {
        "distribution_shape_strength_cases": checks,
        "points_per_path": 101,
        "divergences": ["reverse KL", "forward KL", "total variation", "squared Hellinger"],
        "largest_numerical_increase": worst,
        "bpo_contraction": "(1-beta)^2 checked by exact summation",
    }


def main():
    directory = Path(__file__).resolve().parents[2] / "results/power-bpo-algorithm"
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    errors = []
    for dimension in (3, 17, 127):
        for kappa in (0, 0.1, 0.5, 1):
            q = np.exp(log_normalize(rng.normal(size=dimension)))
            teacher = np.exp(log_normalize(rng.normal(size=dimension)))
            m, _ = target(q, teacher, kappa, 0.3)
            z = rng.normal(size=dimension)
            covariance = np.diag(q) - np.outer(q, q)
            exact = covariance @ (z - np.log(m)) / 0.3
            step = 1e-5
            finite = np.array(
                [
                    (
                        objective(z + step * np.eye(dimension)[i], q, m, 0.3)
                        - objective(z - step * np.eye(dimension)[i], q, m, 0.3)
                    )
                    / (2 * step)
                    for i in range(dimension)
                ]
            )
            errors.append(float(np.max(np.abs(exact - finite))))
            assert errors[-1] < 1e-8
            eig = np.linalg.eigvalsh(covariance)[1:]
            assert eig[0] >= q.min() - 1e-12
            assert eig[-1] <= 0.5 + 1e-12
            residual = z - np.log(m)
            residual -= q @ residual
            p = np.exp(log_normalize(z))
            score = np.eye(dimension) - p[None, :]
            enumerated = (q * residual) @ score / 0.3
            np.testing.assert_allclose(enumerated, exact, atol=1e-12)
    q = np.array([0.4999995, 0.4999995, 1e-6])
    teacher = np.array([0.8, 0.199999, 1e-6])
    m, _ = target(q, teacher, 0.5, 0.3)
    p = np.exp(log_normalize(np.log(m) + np.array([0, 0, 12])))
    counter = {
        "q": q.tolist(),
        "teacher": teacher.tolist(),
        "target": m.tolist(),
        "candidate": p.tolist(),
        "bpo_initial": objective(np.log(q), q, m, 0.3),
        "bpo_candidate": objective(np.log(p), q, m, 0.3),
        "teacher_reverse_kl_initial": float(q @ np.log(q / teacher)),
        "teacher_reverse_kl_candidate": float(p @ np.log(p / teacher)),
    }
    assert counter["bpo_candidate"] < counter["bpo_initial"]
    assert counter["teacher_reverse_kl_candidate"] > counter["teacher_reverse_kl_initial"]
    q = np.array([0.75, 0.23, 0.02])
    teacher = np.array([0.06, 0.24, 0.70])
    eta = 0.3
    m, _ = target(q, teacher, 0.5, eta)
    true_md = np.exp(log_normalize(np.log(q) + eta * ((teacher / q) ** 0.5 - 1) / 0.5))
    summary = {
        "date": "2026-10-14",
        "finite_inner_teacher_progress": finite_inner_checks(directory),
        "gradient_max_abs_error": max(errors),
        "verified_cases": len(errors),
        "bpo_decrease_does_not_imply_teacher_kl_decrease": counter,
        "power_vs_fixed_loss_md": {
            "power_target": m.tolist(),
            "fixed_loss_entropy_md": true_md.tolist(),
            "l1_difference": float(np.abs(m - true_md).sum()),
        },
        "scope": "Exact finite-simplex checks, no new GPU training results",
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pipeline(directory)
    geometry(directory)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
