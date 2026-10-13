"""Teacher-preserving kernels, finite-inner certificates and sequence counterexamples."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from proximal_family_dynamics import divergence, natural_velocity
from proximal_family_geometry import COLORS, ROOT, log_target, logsumexp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "results/power-mean-20261013"


def teacher_kernel(q, teacher, kappa, strength):
    """Construct reversible probability flows from low- to high-ratio states."""
    m = np.exp(log_target(q, teacher, kappa, strength))
    s = q / teacher
    delta = m - q
    low, high = np.flatnonzero(delta > 1e-15), np.flatnonzero(delta < -1e-15)
    total = delta[low].sum()
    kernel = np.zeros((len(q), len(q)))
    if total < 1e-14:
        return np.eye(len(q)), m
    flows = delta[low, None] * (-delta[high])[None, :] / (total * (s[high][None, :] - s[low, None]))
    kernel[np.ix_(low, high)] = flows / teacher[low, None]
    kernel[np.ix_(high, low)] = flows.T / teacher[high, None]
    diagonal = 1 - kernel.sum(-1)
    assert diagonal.min() >= -1e-10
    np.fill_diagonal(kernel, diagonal)
    return kernel, m


def renyi(m, teacher, order):
    return float(logsumexp((1 + order) * np.log(m) - order * np.log(teacher)) / order)


def sequence_example(kappa, strength):
    root_q, root_t = np.array([0.99, 0.01]), np.array([0.01, 0.99])
    child_q = [np.array([0.5, 0.5]), np.array([1 - 1e-6, 1e-6])]
    child_t = [np.array([0.5, 0.5]), np.array([1e-30, 1 - 1e-30])]
    root_m = np.exp(log_target(root_q, root_t, kappa, strength))
    child_m = [
        np.exp(log_target(q, t, kappa, strength)) for q, t in zip(child_q, child_t, strict=True)
    ]
    qjoint = np.concatenate([root_q[i] * child_q[i] for i in range(2)])
    tjoint = np.concatenate([root_t[i] * child_t[i] for i in range(2)])
    mjoint = np.concatenate([root_m[i] * child_m[i] for i in range(2)])
    old = divergence(tjoint, qjoint, 0)
    new = divergence(tjoint, mjoint, 0)
    frozen = divergence(root_t, root_m, 0) + sum(
        root_q[i] * divergence(child_t[i], child_m[i], 0) for i in range(2)
    )
    assert divergence(root_t, root_m, 0) <= divergence(root_t, root_q, 0) + 1e-12
    assert all(
        divergence(t, m, 0) <= divergence(t, q, 0) + 1e-12
        for q, t, m in zip(child_q, child_t, child_m, strict=True)
    )
    return {
        "old_sequence_kl": old,
        "new_sequence_kl": new,
        "frozen_prefix_sum_kl": frozen,
        "new_hard_branch_probability": float(root_m[1]),
    }


def checks():
    rng = np.random.default_rng(20261013)
    kernel_error, certificate_error = 0.0, 0.0
    for dimension in (3, 16, 64):
        for _ in range(10):
            q, teacher, p = rng.dirichlet(np.ones(dimension), size=3)
            for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
                kernel, m = teacher_kernel(q, teacher, kappa, 0.2)
                kernel_error = max(
                    kernel_error,
                    float(np.max(np.abs(q @ kernel - m))),
                    float(np.max(np.abs(teacher @ kernel - teacher))),
                )
                assert np.max(np.abs(kernel.sum(-1) - 1)) < 1e-10
                assert kernel.min() >= -1e-10
                epsilon = divergence(m, p, 0)
                bound = min(
                    (1 + 1 / order) * epsilon + renyi(m, teacher, order)
                    for order in (0.01, 0.03, 0.1, 0.3, 1.0)
                )
                actual = divergence(teacher, p, 0)
                certificate_error = max(certificate_error, actual - bound)
                assert actual <= bound + 1e-10
                previous = 0.0
                for strength in np.linspace(0, 1, 21):
                    target = np.exp(log_target(q, teacher, kappa, strength))
                    distance = 4 * np.arccos(np.clip(np.sqrt(q * target).sum(), 0, 1)) ** 2
                    assert distance >= previous - 1e-10
                    previous = distance
    assert kernel_error < 1e-10
    return {
        "kernel_max_error": kernel_error,
        "renyi_certificate_max_violation": certificate_error,
        "sequence_counterexamples_lambda01": {
            str(k): sequence_example(k, 0.1) for k in (0.0, 0.5, 1.0)
        },
    }


def figures():
    q, teacher = np.array([0.65, 0.3, 0.05]), np.array([0.03, 0.17, 0.8])
    m = np.exp(log_target(q, teacher, 0.5, 0.3))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    grid = np.linspace(0, 1, 101)
    actual, span_bounds, renyi_bounds = [], [], []
    contrast = np.ptp(np.log(m / teacher))
    for fraction in grid:
        p = (1 - fraction) * q + fraction * m
        epsilon = divergence(m, p, 0)
        actual.append(divergence(teacher, p, 0))
        span_bounds.append(divergence(teacher, m, 0) + epsilon + contrast * np.sqrt(epsilon / 2))
        renyi_bounds.append(
            min(
                (1 + 1 / order) * epsilon + renyi(m, teacher, order)
                for order in np.logspace(-4, 1, 100)
            )
        )
    axes[0].plot(grid, actual, label="True teacher reverse KL", color=COLORS[0.0])
    axes[0].plot(grid, span_bounds, label="Range/Pinsker certificate", ls="--")
    axes[0].plot(grid, renyi_bounds, label="Renyi certificate", color=COLORS[0.5])
    axes[0].axhline(actual[0], color="gray", ls=":", label="Old teacher KL")
    axes[0].set(
        xlabel="Inner fitting fraction: 0=anchor, 1=target",
        ylabel="Teacher reverse KL / upper bound",
        title="A  Finite-inner error can certify teacher progress",
    )
    r = np.log(teacher / q)
    reverse = r - q @ r
    forward = teacher / q - 1
    shapes = np.linspace(0, 1, 101)
    reverse_cos, forward_cos = [], []
    for kappa in shapes:
        v = natural_velocity(q, teacher, kappa)
        w = v / q
        norm = np.sqrt(q @ w**2)
        reverse_cos.append(float(q @ (w * reverse) / (norm * np.sqrt(q @ reverse**2))))
        forward_cos.append(float(q @ (w * forward) / (norm * np.sqrt(q @ forward**2))))
    axes[1].plot(shapes, reverse_cos, label="Reverse-KL descent alignment", color=COLORS[0.0])
    axes[1].plot(shapes, forward_cos, label="Forward-KL descent alignment", color=COLORS[1.0])
    axes[1].set(
        xlabel="Power shape kappa",
        ylabel="Cosine in Fisher geometry",
        title="B  The best deterministic direction depends on the objective",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, "inner-certificate-and-direction")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    strengths = np.linspace(0, 1, 101)
    for kappa in (0.0, 0.5, 1.0):
        rows = [sequence_example(kappa, s) for s in strengths]
        axes[0].plot(
            strengths,
            [r["new_sequence_kl"] for r in rows],
            color=COLORS[kappa],
            label=f"kappa={kappa:g}",
        )
    old = sequence_example(0, 0)["old_sequence_kl"]
    axes[0].axhline(old, color="gray", ls=":", label="Original sequence KL")
    rows = [sequence_example(0.5, s) for s in strengths]
    axes[1].plot(
        strengths,
        [r["frozen_prefix_sum_kl"] for r in rows],
        color=COLORS[0.5],
        label="Fixed old prefix weights",
    )
    axes[1].plot(
        strengths,
        [r["new_sequence_kl"] for r in rows],
        color=COLORS[1.0],
        label="Refreshed state visits: true sequence KL",
    )
    for ax in axes:
        ax.set(
            xlabel="Interpolation strength",
            ylabel="KL",
            title="Conditional improvement does not imply sequence improvement",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, "sequence-scope-counterexample")


def save(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT / f"{name}.{extension}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None,
        )
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = checks()
    figures()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "generalization-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
