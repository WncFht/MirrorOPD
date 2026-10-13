"""Exact finite-simplex diagnostics for proximal targets and score variance.

Reproduce from the repository root with:
uv run --no-project --with numpy --with matplotlib python \
    experiments/analysis/proximal_family_geometry.py
"""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "proximal-family-geometry"
COLORS = {0.0: "#335cba", 0.25: "#8d5bbb", 0.5: "#279681", 1.0: "#db7434"}


def logsumexp(x):
    maximum = np.max(x)
    return maximum + np.log(np.exp(x - maximum).sum())


def log_target(q, teacher, kappa, strength):
    lq, lt = np.log(q), np.log(teacher)
    if strength == 0:
        return lq
    if strength == 1:
        return lt
    if kappa == 0:
        raw = (1 - strength) * lq + strength * lt
    else:
        raw = np.logaddexp(np.log1p(-strength) + kappa * lq, np.log(strength) + kappa * lt) / kappa
    return raw - logsumexp(raw)


def signal(r, kappa, strength):
    if kappa == 0:
        return strength * r
    return np.logaddexp(np.log1p(-strength), np.log(strength) + kappa * r) / kappa


def centered_variance(q1, values):
    delta = values[0] - values[1]
    return q1 * (1 - q1) * (1 - 2 * q1) ** 2 * delta**2


def save_rows(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def variance_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    r = np.linspace(-40, 7, 800)
    for kappa in (0.0, 0.5, 1.0):
        axes[0].plot(r, signal(r, kappa, 0.1), color=COLORS[kappa], label=f"κ={kappa:g}, λ=0.1")
    axes[0].axhline(np.log(0.9), color=COLORS[1.0], ls=":", alpha=0.7)
    axes[0].axhline(0, color="#777777", lw=0.7)
    axes[0].set(
        xlabel="原始 log-ratio r = log(T/q)",
        ylabel="冻结信号 f(r)",
        title="A  缩放与负尾部平滑",
        ylim=(-4.2, 3.5),
    )
    axes[0].legend(fontsize=9)
    q1 = 0.25
    rows = []
    grid = np.linspace(2, 120, 360)
    for strength, style in ((0.1, "-"), (0.03, "--")):
        for kappa in (0.0, 1.0):
            curve = []
            for exponent in grid:
                epsilon = 10.0 ** (-exponent)
                values = np.array([np.log(epsilon / q1), np.log((1 - epsilon) / (1 - q1))])
                frozen = signal(values, kappa, strength)
                variance = centered_variance(q1, frozen)
                curve.append(variance)
                reward_variance = q1 * (1 - q1) * (frozen[0] - frozen[1]) ** 2
                rows.append(
                    {
                        "negative_log10_epsilon": exponent,
                        "q1": q1,
                        "kappa": kappa,
                        "strength": strength,
                        "centered_score_variance": variance,
                        "standardized_score_variance": variance / reward_variance,
                    }
                )
            kind = "几何" if kappa == 0 else "算术"
            axes[1].plot(
                grid, curve, color=COLORS[kappa], ls=style, label=f"{kind}, λ={strength:g}"
            )
    axes[1].set(
        xlabel=r"teacher 极小概率：$-\log_{10}\varepsilon$",
        ylabel="条件中心化 score 梯度方差",
        yscale="log",
        title="B  任意固定几何系数仍会发散",
    )
    axes[1].legend(fontsize=9)
    for strength in (0.1, 0.03):
        selected = [row for row in rows if row["kappa"] == 1 and row["strength"] == strength]
        axes[2].plot(
            grid,
            [row["standardized_score_variance"] for row in selected],
            ls="-" if strength == 0.1 else "--",
            lw=2,
            label=f"算术, λ={strength:g}",
        )
    axes[2].axhline((1 - 2 * q1) ** 2, color=COLORS[0.0], ls=":", lw=2, label="几何：所有 λ > 0")
    axes[2].set(
        xlabel=r"teacher 极小概率：$-\log_{10}\varepsilon$",
        ylabel="标准化后 score 梯度方差",
        title="C  二元例子中标准化完全抵消差异",
        ylim=(0, 0.5),
    )
    axes[2].legend(fontsize=9)
    fig.suptitle("固定 q=(0.25,0.75)，T=(ε,1−ε)；精确期望，无后缀回报或 PPO clipping", fontsize=12)
    for ax in axes:
        ax.grid(alpha=0.18)
    save_rows("variance.csv", rows)
    save_figure(fig, "variance")


def geometry_figure():
    q = np.array([0.65, 0.30, 0.05])
    teacher = np.array([0.03, 0.17, 0.80])
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    ax = axes[0]
    triangle = vertices[[0, 1, 2, 0]]
    ax.plot(triangle[:, 0], triangle[:, 1], color="#777777")
    for coordinate in range(3):
        for level in (0.2, 0.4, 0.6, 0.8):
            others = [i for i in range(3) if i != coordinate]
            endpoints = np.zeros((2, 3))
            endpoints[:, coordinate] = level
            endpoints[0, others[0]] = 1 - level
            endpoints[1, others[1]] = 1 - level
            xy = endpoints @ vertices
            ax.plot(xy[:, 0], xy[:, 1], color="#cccccc", lw=0.6)
    rows = []
    for kappa, color in COLORS.items():
        strengths = np.linspace(0, 1, 201)
        distributions = np.array([np.exp(log_target(q, teacher, kappa, s)) for s in strengths])
        xy = distributions @ vertices
        ax.plot(xy[:, 0], xy[:, 1], color=color, label=f"κ={kappa:g}", lw=2)
        point = np.exp(log_target(q, teacher, kappa, 0.2)) @ vertices
        ax.scatter(*point, color=color, s=35, zorder=4)
        for strength, p in zip(strengths, distributions, strict=True):
            rows.append({"kappa": kappa, "strength": strength, "p1": p[0], "p2": p[1], "p3": p[2]})
    for name, dist in (("q", q), ("T", teacher)):
        point = dist @ vertices
        ax.scatter(*point, color="#222222", s=45, zorder=5)
        ax.annotate(name, point, xytext=(7, 3), textcoords="offset points", fontsize=12)
    for j, point in enumerate(vertices):
        ax.annotate(
            f"p{j + 1}=1",
            point,
            xytext=(0, -16 if j != 2 else 9),
            textcoords="offset points",
            ha="center",
        )
    ax.set(
        aspect="equal",
        xlim=(-0.08, 1.08),
        ylim=(-0.07, 0.97),
        title="A  三维概率向量：不同几何给出不同路径",
    )
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=9)
    rng = np.random.default_rng(20261013)
    logq = rng.normal(0, 0.7, 64)
    logq -= logsumexp(logq)
    logt = rng.normal(0, 2.5, 64)
    logt[0], logt[1] = -35, 7
    logt -= logsumexp(logt)
    q64, t64 = np.exp(logq), np.exp(logt)
    ratios = logt - logq
    order = np.argsort(ratios)
    high_rows = []
    for kappa, color in COLORS.items():
        target_lp = log_target(q64, t64, kappa, 0.2)
        axes[1].plot(
            ratios[order],
            (target_lp - logq)[order],
            color=color,
            label=f"κ={kappa:g}",
            lw=2,
            marker=".",
            ms=3,
        )
        high_rows.extend(
            {
                "action": action,
                "kappa": kappa,
                "strength": 0.2,
                "anchor_probability": q64[action],
                "teacher_probability": t64[action],
                "teacher_log_ratio": ratios[action],
                "target_log_ratio": target_lp[action] - logq[action],
            }
            for action in range(64)
        )
    axes[1].set(
        xlabel=r"各 action 的 $\log(T_a/q_a)$",
        ylabel=r"归一化目标的 $\log(m_a/q_a)$",
        title="B  64 维：几何直线与幂均值曲线",
    )
    axes[1].grid(alpha=0.18)
    axes[1].legend(fontsize=9)
    fig.suptitle("归一化幂均值族：κ=0 几何，κ=1 算术；左图圆点均为 λ=0.2", fontsize=12)
    save_rows("simplex.csv", rows)
    save_rows("high-dimensional.csv", high_rows)
    save_figure(fig, "geometry")


def verify_identities():
    q = np.array([0.2, 0.3, 0.5])
    teacher = np.array([0.6, 0.35, 0.05])
    strength = 0.2
    errors = []
    for kappa in (0.25, 0.5, 0.75):
        lp = log_target(q, teacher, kappa, strength)
        p = np.exp(lp)
        mixture_power = (1 - strength) * q**kappa + strength * teacher**kappa
        gradient_p = -mixture_power * p ** (-kappa) / kappa
        assert np.ptp(gradient_p) < 1e-12
        gradient_logits = p * (gradient_p - p @ gradient_p)
        finite_difference = []
        for j in range(3):
            plus, minus = lp.copy(), lp.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            pp, pm = np.exp(plus - logsumexp(plus)), np.exp(minus - logsumexp(minus))
            fp = (1 - np.sum(mixture_power * pp ** (1 - kappa))) / (kappa * (1 - kappa))
            fm = (1 - np.sum(mixture_power * pm ** (1 - kappa))) / (kappa * (1 - kappa))
            finite_difference.append((fp - fm) / 2e-5)
        errors.append(float(np.max(np.abs(gradient_logits - finite_difference))))
        assert errors[-1] < 1e-8
    arithmetic = np.exp(log_target(q, teacher, 1, strength))
    assert np.max(np.abs(arithmetic - ((1 - strength) * q + strength * teacher))) < 1e-14
    geometric = np.exp(log_target(q, teacher, 0, strength))
    near_geometric = np.exp(log_target(q, teacher, 1e-5, strength))
    assert np.max(np.abs(near_geometric - geometric)) < 1e-5
    # General bound uses normalization of the teacher, not vocabulary size.
    root_left, root_right = 2.0, np.exp(2)
    for _ in range(80):
        root = (root_left + root_right) / 2
        if np.log(root) - 2 * (1 - 1 / root) < 0:
            root_left = root
        else:
            root_right = root
    constant = np.log(root) ** 2 / (root - 1)
    rng = np.random.default_rng(20261013)
    largest_ratio = 0
    for dimension in (3, 64, 1024):
        q0, t0 = rng.dirichlet(np.ones(dimension)), rng.dirichlet(np.ones(dimension))
        r = np.log(t0 / q0)
        for kappa in (0.1, 0.25, 0.5, 1.0):
            for lam in (0.03, 0.1, 0.5, 0.9):
                f = signal(r, kappa, lam)
                second = q0 @ f**2
                bound = (np.log1p(-lam) ** 2 + constant * lam) / kappa**2
                assert second <= bound * (1 + 1e-12)
                largest_ratio = max(largest_ratio, float(second / bound))
    return {
        "power_divergence_stationarity_max_error": max(errors),
        "geometric_limit_max_error": float(np.max(np.abs(near_geometric - geometric))),
        "positive_tail_constant": float(constant),
        "tested_dimensions": [3, 64, 1024],
        "largest_second_moment_to_bound_ratio": largest_ratio,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    )
    checks = verify_identities()
    variance_figure()
    geometry_figure()
    summary = {
        "date": "2026-10-13",
        "kind": "Exact finite-vocabulary analytic diagnostics",
        "question": "Distinguish signal scaling from uniform variance control; unify proximal targets",
        "success_conditions": "Endpoint identities, divergence KKT conditions and finite-difference errors below 1e-8",
        "rejection_conditions": "Any violated identity or bound rejects the proposed mathematical mapping",
        "variable_choices": "Interpolation strength, power geometry, fitting loss and estimator are independent",
        "seed": 20261013,
        "variance_anchor": [0.25, 0.75],
        "variance_teacher": "(epsilon, 1-epsilon), epsilon=10^(-x), x in [2,120]",
        "estimator": "One token score in first categorical logit; exact anchor conditional centering",
        "limits": "Not an LLM training benchmark; no suffix returns, sample group normalization or PPO clipping",
        "checks": checks,
        "source_commit": subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "reproduce": "uv run --no-project --with numpy --with matplotlib python experiments/analysis/proximal_family_geometry.py",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
