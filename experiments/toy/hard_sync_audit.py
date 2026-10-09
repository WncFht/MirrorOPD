"""Audit Chapter 4 in float64; this is a categorical oracle, not LLM training.

Semantic ledger: Gibbs=teal, Mean=orange; solid=measured, dashed=theory.
Geometry ledger: 5.5 inch two-panel paper figures, log axes only for positives.
Numerical outputs do not require plotting dependencies. --plot uses figure-viz.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import lambertw, logsumexp


def normalize(logp):
    return np.exp(logp - logsumexp(logp))


def kl(p, q):
    """Stable normalized KL, including cubic-convergence values near 1e-16."""
    d = p / q - 1
    small = np.abs(d) < 1e-3
    f = np.empty_like(d)
    f[small] = sum((-1.0) ** n * d[small] ** n / (n * (n - 1)) for n in range(2, 10))
    f[~small] = (1 + d[~small]) * np.log1p(d[~small]) - d[~small]
    return float(q @ f)


def span(p, t):
    return float(np.ptp(np.log(p / t)))


def gibbs(q, t, eta):
    return normalize((1 - eta) * np.log(q) + eta * np.log(t))


def mean(q, t, eta):
    """Independent SciPy solver, cross-checked against token_inner.mean_target."""
    r = np.log(t / q)
    g = eta * (r - q @ r)
    if np.ptp(g) < 1e-14:
        return q.copy(), 0.0, 0.0

    def probabilities(lam):
        return q * np.exp(g - lambertw(lam * np.exp(g)).real)

    # expm1 prevents subtraction of two nearly equal normalization constants.
    def residual(lam):
        return float(q @ np.expm1(g - lambertw(lam * np.exp(g)).real))

    high = 1.0
    while residual(high) > 0:
        high *= 2
    lam = brentq(residual, 0.0, high, xtol=5e-324, rtol=1e-14)
    p = probabilities(lam)
    w = p / q
    error = max(abs(p.sum() - 1), np.max(abs(np.log(w) + lam * w - g)))
    return p / p.sum(), lam, float(error)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(seed):
    import torch

    from mirroropd.token_inner import mean_target, mirror_target

    rng = np.random.default_rng(seed)
    max_identity = max_core = max_kkt = 0.0
    for size in (2, 4, 8):
        for _ in range(12):
            q, t = (normalize(rng.normal(size=size)) for _ in range(2))
            for eta in (0.2, 0.8, 1.0, 1.2, 1.8, 2.0, 2.2):
                p = gibbs(q, t, eta)
                identity = eta * kl(q, t) - eta * kl(p, t) - (1 - eta) * kl(p, q) - kl(q, p)
                max_identity = max(max_identity, abs(identity))
                assert abs(span(p, t) - abs(1 - eta) * span(q, t)) < 1e-12
                core = mirror_target(torch.tensor(q), torch.tensor(t), eta).numpy()
                assert np.max(abs(core - p)) < 1e-14
                pm, _, error = mean(q, t, eta)
                max_kkt = max(max_kkt, error)
                if eta <= 2:
                    assert span(pm, t) < span(q, t)
            # The repository reference uses nested bisection; audit 36 instances.
            core, _ = mean_target(torch.tensor(q), torch.tensor(np.log(t / q)), 1)
            independent, _, _ = mean(q, t, 1)
            max_core = max(max_core, float(np.max(abs(core.numpy() - independent))))
    assert max_identity < 1e-12
    assert max_core < 1e-12
    assert max_kkt < 1e-12

    # Variable-step products and deterministic output-bias floor.
    teacher = np.array([0.2, 0.3, 0.5])
    initial = normalize(np.log(teacher) + np.array([-0.6, 0.1, 0.4]))
    schedules = []
    for power in (1, 2):
        q = initial.copy()
        for k in range(1000):
            q = gibbs(q, teacher, 1 / (k + 2) ** power)
        predicted = 1 / 1001 if power == 1 else 1002 / (2 * 1001)
        actual = span(q, teacher) / span(initial, teacher)
        assert abs(actual - predicted) < 1e-11
        schedules.append({"power": power, "steps": 1000, "ratio": actual, "theory": predicted})
    for eta in (0.2, 0.8, 1.2, 1.8):
        q = initial.copy()
        error = np.array([-0.01, 0.0, 0.02])
        for _ in range(500):
            q = normalize((1 - eta) * np.log(q) + eta * np.log(teacher) + error)
        stationary = normalize(np.log(teacher) + error / eta)
        assert np.max(abs(q - stationary)) < 1e-13

    t = np.array([0.2, 0.3, 0.5])
    z = np.array([-1.0, 0.2, 0.8])
    cubic = []
    for eps in (0.4, 0.2, 0.1, 0.05, 0.02, 0.01):
        q = normalize(np.log(t) + eps * z)
        p, lam, error = mean(q, t, 1)
        cubic.append(
            {
                "epsilon": eps,
                "old_kl": kl(q, t),
                "new_kl": kl(p, t),
                "kl_cubic_ratio": kl(p, t) / kl(q, t) ** 3,
                "span_cubic_ratio": span(p, t) / span(q, t) ** 3,
                "multiplier": lam,
                "kkt_residual": error,
            }
        )
    assert abs(cubic[-1]["kl_cubic_ratio"] - 1) < 0.02

    trajectories = []
    for eta in (0.5, 1.5, 2.0, 2.1):
        for method, update in (("Gibbs", gibbs), ("Mean", mean)):
            q = normalize(np.log(t) + 0.2 * z)
            for k in range(10001):
                if k <= 30 or k in (100, 300, 1000, 3000, 10000):
                    trajectories.append(
                        {
                            "method": method,
                            "eta": eta,
                            "step": k,
                            "span": span(q, t),
                            "kl": kl(q, t),
                        }
                    )
                if k == 10000 or (eta != 2 and k == 30):
                    break
                q = update(q, t, eta)
                if method == "Mean":
                    q = q[0]
    boundary = next(
        r for r in trajectories if r["method"] == "Mean" and r["eta"] == 2 and r["step"] == 10000
    )
    assert abs(10000 * boundary["kl"] - 1 / 16) < 0.004

    # Same innovations across eta: paired Monte Carlo, 20000 independent paths.
    reps, steps, sigma = 20000, 240, 0.01
    innovations = rng.normal(size=(steps, reps, len(t))) * sigma
    innovations -= (innovations @ t)[..., None]
    base_energy = sigma**2 * (1 - t @ t)
    noise = []
    for mode in ("output", "reward"):
        for eta in (0.2, 0.5, 1.0, 1.5, 1.8):
            y = np.zeros((reps, len(t)))
            scale = 1 if mode == "output" else eta
            for innovation in innovations:
                y = (1 - eta) * y + scale * innovation
            energy = (y * y) @ t
            predicted = base_energy * scale**2 * (1 - (1 - eta) ** (2 * steps)) / (eta * (2 - eta))
            noise.append(
                {
                    "mode": mode,
                    "eta": eta,
                    "energy_mean": float(energy.mean()),
                    "energy_se": float(energy.std(ddof=1) / np.sqrt(reps)),
                    "energy_theory": float(predicted),
                    "reps": reps,
                    "steps": steps,
                    "sigma": sigma,
                }
            )
            assert abs(energy.mean() - predicted) < 5 * energy.std(ddof=1) / np.sqrt(reps)

    examples = []
    for initial, target, eta in (
        ([1e-12, 1 - 1e-12], [0.1, 0.9], 1.1),
        ([1 - 1e-12, 1e-12], [0.01, 0.99], 0.5),
    ):
        q, teacher = np.array(initial), np.array(target)
        p = gibbs(q, teacher, eta)
        examples.append({"eta": eta, "old_kl": kl(q, teacher), "new_kl": kl(p, teacher)})
    return {
        "cubic": cubic,
        "trajectories": trajectories,
        "noise": noise,
        "examples": examples,
        "schedules": schedules,
        "audit": {
            "max_three_point_residual": max_identity,
            "max_mean_core_difference": max_core,
            "max_mean_kkt_residual": max_kkt,
        },
    }


def plot(data, out, style_dir):
    sys.path.insert(0, str(style_dir))
    import fvstyle as fv
    import matplotlib.pyplot as plt

    fv.apply_rc("paper")
    colors = fv.series_colors("paper", 2)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), layout="constrained")
    for method, color in zip(("Gibbs", "Mean"), colors, strict=True):
        rows = [
            r
            for r in data["trajectories"]
            if r["method"] == method and r["eta"] == 2 and r["step"] > 0
        ]
        axes[0].loglog(
            [r["step"] for r in rows], [r["kl"] for r in rows], color=color, label=method
        )
    k = np.array([300, 1000, 3000, 10000])
    axes[0].loglog(k, 1 / (16 * k), "--", color=fv.NEUTRALS["baseline"], label=r"$1/(16k)$")
    axes[0].set(
        xlabel="Outer step k",
        ylabel=r"$\mathrm{KL}(\pi_k\Vert\pi_T)$",
        title=r"(a) Boundary $\eta=2$",
    )
    axes[0].legend(fontsize=7)
    for mode, color in zip(("output", "reward"), colors, strict=True):
        rows = [r for r in data["noise"] if r["mode"] == mode]
        x = [r["eta"] for r in rows]
        axes[1].errorbar(
            x,
            [r["energy_mean"] for r in rows],
            yerr=[1.96 * r["energy_se"] for r in rows],
            fmt="o",
            color=color,
            label=mode,
        )
        axes[1].plot(x, [r["energy_theory"] for r in rows], "--", color=color)
    axes[1].set(
        xlabel=r"Mirror step $\eta$", ylabel=r"$\mathbb{E}\|y\|_t^2$", title="(b) Noise placement"
    )
    axes[1].legend(fontsize=7)
    for ax in axes:
        fv.apply_frame(ax, "closed")
        fv.apply_grid(ax, "y-only")
    audit = fv.audit(fig)
    (out / "figure-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fv.save(fig, out / "hard-sync.png")
    fv.save(fig, out / "hard-sync.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/hard-sync-audit"))
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--plot-style", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = run(args.seed)
    for name in ("cubic", "trajectories", "noise", "examples", "schedules"):
        write_csv(args.out / f"{name}.csv", data[name])
    metadata = dict(
        seed=args.seed,
        source_commit=subprocess.check_output(  # noqa: S603 -- Fixed read-only Git command.
            [shutil.which("git"), "rev-parse", "HEAD"], text=True
        ).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dtype="float64",
        teacher=[0.2, 0.3, 0.5],
        direction=[-1, 0.2, 0.8],
        numpy=np.__version__,
        scipy=scipy.__version__,
        command="PYTHONPATH=src uv run --with numpy --with scipy --with torch --with matplotlib python experiments/toy/hard_sync_audit.py --plot-style ~/.agents/skills/figure-viz/assets/mpl",
        **data["audit"],
    )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.plot_style:
        plot(data, args.out, args.plot_style)
    print(json.dumps(metadata, indent=2))
    print("Cubic final:", data["cubic"][-1])


if __name__ == "__main__":
    main()
