"""Exact checks for divergence barycenters, escort dynamics and tail orders.

All figures use analytic finite-simplex quantities, not sampled LLM gradients.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from proximal_family_geometry import (
    COLORS,
    OUT,
    ROOT,
    log_target,
    logsumexp,
    save_figure,
    save_rows,
    signal,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PALETTE = {**COLORS, 0.75: "#b54472"}
SHAPES = (0.0, 0.25, 0.5, 0.75, 1.0)


def divergence(u, p, kappa):
    r = np.log(u) - np.log(p)
    if kappa == 0:
        return float(-p @ r)
    if kappa == 1:
        return float(u @ r)
    return float(-p @ np.expm1(kappa * r) / (kappa * (1 - kappa)))


def escort(p, kappa):
    lp = kappa * np.log(p)
    return np.exp(lp - logsumexp(lp))


def power_sum(p, kappa):
    return float(np.exp(logsumexp(kappa * np.log(p))))


def effective_strength(q, teacher, kappa, strength):
    sq, st = power_sum(q, kappa), power_sum(teacher, kappa)
    return strength * st / ((1 - strength) * sq + strength * st)


def natural_velocity(q, teacher, kappa):
    r = np.log(teacher) - np.log(q)
    potential = r if kappa == 0 else np.expm1(kappa * r) / kappa
    return q * (potential - q @ potential)


def fisher_speed_squared(q, teacher, kappa):
    v = natural_velocity(q, teacher, kappa)
    return float(np.sum(v**2 / q))


def objective(q, teacher, p, kappa, strength):
    return (1 - strength) * divergence(q, p, kappa) + strength * divergence(teacher, p, kappa)


def target_scale(q, teacher, kappa, strength):
    raw = (
        np.logaddexp(
            np.log1p(-strength) + kappa * np.log(q),
            np.log(strength) + kappa * np.log(teacher),
        )
        / kappa
    )
    return float(np.exp(kappa * logsumexp(raw)))


def verify():
    rng = np.random.default_rng(20261013)
    errors = {
        "decomposition": 0.0,
        "escort_step": 0.0,
        "monotonic_identity": 0.0,
        "flow_derivative": 0.0,
        "hessian": 0.0,
        "escort_flow": 0.0,
        "hellinger_affinity_flow": 0.0,
        "parameter_gradient": 0.0,
        "projection_orthogonality": 0.0,
    }
    count = 0
    for dimension in (3, 64, 1024):
        for _ in range(5):
            q = rng.dirichlet(np.full(dimension, 2.0))
            teacher = rng.dirichlet(np.full(dimension, 2.0))
            p = rng.dirichlet(np.full(dimension, 2.0))
            for kappa in (0.25, 0.5, 0.75):
                strength = 0.2
                m = np.exp(log_target(q, teacher, kappa, strength))
                scale = target_scale(q, teacher, kappa, strength)
                gap = objective(q, teacher, p, kappa, strength) - objective(
                    q, teacher, m, kappa, strength
                )
                error = abs(gap - scale * divergence(m, p, kappa))
                errors["decomposition"] = max(errors["decomposition"], error)
                a = effective_strength(q, teacher, kappa, strength)
                eq, et, em = escort(q, kappa), escort(teacher, kappa), escort(m, kappa)
                errors["escort_step"] = max(
                    errors["escort_step"], float(np.max(np.abs(em - ((1 - a) * eq + a * et))))
                )
                teacher_progress = strength * (
                    divergence(teacher, q, kappa) - divergence(teacher, p, kappa)
                )
                rhs = (1 - strength) * divergence(q, p, kappa) + scale * (
                    divergence(m, q, kappa) - divergence(m, p, kappa)
                )
                errors["monotonic_identity"] = max(
                    errors["monotonic_identity"], abs(teacher_progress - rhs)
                )
                assert divergence(teacher, m, kappa) <= divergence(teacher, q, kappa) + 1e-12
                velocity = natural_velocity(q, teacher, kappa)
                if kappa == 0.5:
                    affinity = float(np.sqrt(q * teacher).sum())
                    affinity_derivative = float(0.5 * np.sqrt(teacher / q) @ velocity)
                    errors["hellinger_affinity_flow"] = max(
                        errors["hellinger_affinity_flow"],
                        abs(affinity_derivative - (1 - affinity**2)),
                    )
                derivative = float(
                    -np.exp(kappa * (np.log(teacher) - np.log(q))) @ velocity / kappa
                )
                errors["flow_derivative"] = max(
                    errors["flow_derivative"],
                    abs(derivative + fisher_speed_squared(q, teacher, kappa)),
                )
                dt = 1e-6
                finite = (escort(q + dt * velocity, kappa) - escort(q - dt * velocity, kappa)) / (
                    2 * dt
                )
                exact = power_sum(teacher, kappa) / power_sum(q, kappa) * (et - eq)
                errors["escort_flow"] = max(
                    errors["escort_flow"], float(np.max(np.abs(finite - exact)))
                )
                count += 1
    q, teacher = np.array([0.65, 0.30, 0.05]), np.array([0.03, 0.17, 0.80])
    for kappa in (0.25, 0.5, 0.75):
        strength = 0.2
        lp = log_target(q, teacher, kappa, strength)
        m = np.exp(lp)
        scale = target_scale(q, teacher, kappa, strength)
        s = (1 - strength) * q**kappa + strength * teacher**kappa
        hessian = np.empty((3, 3))
        for j in range(3):
            gradients = []
            for sign in (1, -1):
                perturbed = lp.copy()
                perturbed[j] += sign * 1e-5
                p = np.exp(perturbed - logsumexp(perturbed))
                gp = -s * p ** (-kappa) / kappa
                gradients.append(p * (gp - p @ gp))
            hessian[:, j] = (gradients[0] - gradients[1]) / 2e-5
        fisher = np.diag(m) - np.outer(m, m)
        errors["hessian"] = max(errors["hessian"], float(np.max(np.abs(hessian - scale * fisher))))
    # A shared conditional exponential family has a proper low-rank tangent
    # space; the projection check does not assume independent prefix logits.
    prefixes, actions, parameters = 3, 5, 4
    weights = np.array([0.2, 0.3, 0.5])
    qs = rng.dirichlet(np.ones(actions), size=prefixes)
    teachers = rng.dirichlet(np.ones(actions), size=prefixes)
    features = rng.normal(size=(prefixes, actions, parameters))
    means = np.sum(qs[:, :, None] * features, axis=1)
    jacobian = (qs[:, :, None] * (features - means[:, None, :])).reshape(-1, parameters)
    metric = (weights[:, None] / qs).reshape(-1)
    fisher = jacobian.T @ (metric[:, None] * jacobian)
    for kappa in (0.25, 0.5, 0.75):
        velocity = np.concatenate(
            [natural_velocity(q, t, kappa) for q, t in zip(qs, teachers, strict=True)]
        )
        gradient = -jacobian.T @ (metric * velocity)
        projected = jacobian @ np.linalg.pinv(fisher) @ (-gradient)
        orthogonality = jacobian.T @ (metric * (velocity - projected))
        errors["projection_orthogonality"] = max(
            errors["projection_orthogonality"], float(np.max(np.abs(orthogonality)))
        )
        assert np.sum(metric * projected**2) <= np.sum(metric * velocity**2) + 1e-12
        if kappa == 0.5:
            assert np.sum(metric * projected**2) <= 4 + 1e-12
        numerical = []
        for j in range(parameters):
            costs = []
            for sign in (1, -1):
                logits = np.log(qs) + sign * 1e-5 * features[:, :, j]
                conditional = np.array([np.exp(row - logsumexp(row)) for row in logits])
                costs.append(
                    sum(
                        w * divergence(t, q, kappa)
                        for w, t, q in zip(weights, teachers, conditional, strict=True)
                    )
                )
            numerical.append((costs[0] - costs[1]) / 2e-5)
        errors["parameter_gradient"] = max(
            errors["parameter_gradient"], float(np.max(np.abs(gradient - numerical)))
        )
    assert max(errors.values()) < 1e-7, errors
    assert count == 45
    return {
        "tested_dimensions": [3, 64, 1024],
        "random_checks": count,
        "shared_parameter_checks": {
            "prefixes": prefixes,
            "actions": actions,
            "parameters": parameters,
            "shapes": [0.25, 0.5, 0.75],
        },
        "max_errors": errors,
    }


def draw_simplex(ax, probabilities):
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    triangle = vertices[[0, 1, 2, 0]]
    ax.plot(triangle[:, 0], triangle[:, 1], color="#666666")
    for coordinate in range(3):
        others = [i for i in range(3) if i != coordinate]
        for level in (0.2, 0.4, 0.6, 0.8):
            ends = np.zeros((2, 3))
            ends[:, coordinate] = level
            ends[0, others[0]], ends[1, others[1]] = 1 - level, 1 - level
            xy = ends @ vertices
            ax.plot(xy[:, 0], xy[:, 1], color="#cccccc", lw=0.6)
    xy = np.asarray(probabilities) @ vertices
    ax.plot(xy[:, 0], xy[:, 1], color=PALETTE[0.5], lw=2)
    selected = np.array([0, 1, 2, 4, 8, 16, len(xy) - 1])
    ax.scatter(xy[selected, 0], xy[selected, 1], color=PALETTE[0.5], s=35, zorder=3)
    for index in (0, 2, 4, 8):
        ax.annotate(str(index), xy[index], xytext=(5, -8), textcoords="offset points", fontsize=9)
    for name, index in ((r"$q_0$", 0), ("T", -1)):
        ax.annotate(name, xy[index], xytext=(-18, 9), textcoords="offset points", fontsize=11)
    ax.set(aspect="equal", xlim=(-0.07, 1.07), ylim=(-0.05, 0.94))
    ax.axis("off")


def escort_figure():
    teacher = np.array([0.03, 0.17, 0.80])
    q = np.array([0.65, 0.30, 0.05])
    distributions = [q]
    rows = []
    for iteration in range(60):
        a = effective_strength(q, teacher, 0.5, 0.2)
        rows.append(
            {
                "iteration": iteration,
                "effective_strength": a,
                "anchor_power_sum": power_sum(q, 0.5),
                "teacher_power_sum": power_sum(teacher, 0.5),
            }
        )
        q = np.exp(log_target(q, teacher, 0.5, 0.2))
        distributions.append(q)
    distributions.append(teacher)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    draw_simplex(axes[0], distributions)
    draw_simplex(axes[1], [escort(q, 0.5) for q in distributions])
    axes[0].set_title("A  概率坐标：外层轨迹是一条曲线")
    axes[1].set_title(r"B  escort 坐标 $q^{1/2}/\sum q^{1/2}$：同一轨迹为直线")
    fig.suptitle("κ=1/2，固定 λ=0.2；圆点及数字表示精确外层迭代", fontsize=12)
    save_rows("escort-iterations.csv", rows)
    save_figure(fig, "escort")


def decomposition_figure():
    q, teacher = np.array([0.65, 0.30, 0.05]), np.array([0.03, 0.17, 0.80])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    rows = []
    for kappa in (0.25, 0.5, 0.75):
        m = np.exp(log_target(q, teacher, kappa, 0.2))
        values = []
        for delta in np.linspace(-3, 3, 161):
            lp = np.log(m) + delta * np.array([1, -1, 0.2])
            p = np.exp(lp - logsumexp(lp))
            distance = divergence(m, p, kappa)
            gap = objective(q, teacher, p, kappa, 0.2) - objective(q, teacher, m, kappa, 0.2)
            values.append((distance, gap))
            rows.append(
                {
                    "kappa": kappa,
                    "logit_displacement": delta,
                    "target_divergence": distance,
                    "objective_gap": gap,
                }
            )
        values.sort()
        axes[0].plot(*np.array(values).T, color=PALETTE[kappa], label=f"κ={kappa:g}")
    axes[0].set(
        xlabel=r"目标拟合误差 $D_\kappa(m\Vert p)$",
        ylabel=r"重心目标超额损失 $F(p)-F(m)$",
        title="A  精确分解：两者只有一个常数因子",
    )
    kappa = 0.5
    m = np.exp(log_target(q, teacher, kappa, 0.2))
    grid = np.linspace(-2.5, 2.5, 201)
    direction = np.array([1, -1, 0.2])
    variance = m @ direction**2 - (m @ direction) ** 2
    curves = {"Hellinger": [], "reverse KL": []}
    for delta in grid:
        lp = np.log(m) + delta * direction
        p = np.exp(lp - logsumexp(lp))
        curves["Hellinger"].append(divergence(m, p, 0.5))
        curves["reverse KL"].append(divergence(m, p, 0))
    axes[1].plot(grid, curves["Hellinger"], color=PALETTE[0.5], label=r"$D_{1/2}(m\Vert p)$")
    axes[1].plot(grid, curves["reverse KL"], color=PALETTE[0.0], label=r"$\mathrm{KL}(p\Vert m)$")
    axes[1].plot(
        grid, variance * grid**2 / 2, color="#777777", ls="--", label="共同的 Fisher 二阶项"
    )
    axes[1].set(
        xlabel="沿固定 logit 方向的位移",
        ylabel="目标拟合损失",
        title="B  相同 target、相同局部曲率，远处损失不同",
    )
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.18)
    save_rows("objective-decomposition.csv", rows)
    save_figure(fig, "decomposition")


def convergence_figure():
    teacher = np.array([0.03, 0.17, 0.80])
    initial = np.array([0.65, 0.30, 0.05])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    rows = []
    for kappa in SHAPES:
        q = initial.copy()
        initial_d = divergence(teacher, q, kappa)
        distances, steps = [], []
        for iteration in range(41):
            d = divergence(teacher, q, kappa)
            a = 0.2 if kappa == 0 else effective_strength(q, teacher, kappa, 0.2)
            distances.append(d / initial_d)
            steps.append(a)
            rows.append(
                {
                    "iteration": iteration,
                    "kappa": kappa,
                    "teacher_divergence": d,
                    "normalized_teacher_divergence": d / initial_d,
                    "effective_escort_strength": a,
                }
            )
            q = np.exp(log_target(q, teacher, kappa, 0.2))
        axes[0].plot(range(41), distances, color=PALETTE[kappa], label=f"κ={kappa:g}")
        axes[1].plot(range(41), steps, color=PALETTE[kappa], label=f"κ={kappa:g}")
        assert np.all(np.diff(distances) <= 1e-10)
    axes[0].set(
        xlabel="精确外层步数",
        ylabel=r"$D_\kappa(T\Vert q_k)/D_\kappa(T\Vert q_0)$",
        yscale="log",
        title="A  各自的 teacher 散度单调下降",
    )
    axes[1].set(
        xlabel="精确外层步数",
        ylabel="有效 escort 混合系数",
        title="B  接近 teacher 时，所有系数都趋于 λ=0.2",
    )
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
    fig.suptitle("不同 κ 的散度不同；曲线用于验证单调性，不能据此给训练算法排名", fontsize=12)
    save_rows("outer-convergence.csv", rows)
    save_figure(fig, "convergence")


def threshold_figure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    rows = []
    grid = np.linspace(1, 30, 240)
    for kappa in SHAPES:
        positive, negative = [], []
        for exponent in grid:
            epsilon = 10.0 ** (-exponent)
            positive.append(
                fisher_speed_squared(np.array([epsilon, 1 - epsilon]), np.array([0.5, 0.5]), kappa)
            )
            negative.append(
                fisher_speed_squared(
                    np.array([0.25, 0.75]), np.array([epsilon, 1 - epsilon]), kappa
                )
            )
        axes[0].plot(grid, positive, color=PALETTE[kappa], label=f"κ={kappa:g}")
        axes[1].plot(grid, negative, color=PALETTE[kappa], label=f"κ={kappa:g}")
        rows.extend(
            {"case": "rare_student", "kappa": kappa, "coordinate": exponent, "value": speed}
            for exponent, speed in zip(grid, positive, strict=True)
        )
        rows.extend(
            {"case": "rare_teacher", "kappa": kappa, "coordinate": exponent, "value": speed}
            for exponent, speed in zip(grid, negative, strict=True)
        )
    axes[0].axhline(4, color=PALETTE[0.5], ls=":", lw=1)
    axes[0].set(
        xlabel=r"学生稀有概率 $-\log_{10}\varepsilon$",
        ylabel="自然梯度 Fisher 范数平方",
        yscale="log",
        title="A  正尾部：κ>1/2 的速度可无界",
    )
    axes[1].set(
        xlabel=r"教师稀有概率 $-\log_{10}\varepsilon$",
        ylabel="自然梯度 Fisher 范数平方",
        yscale="log",
        title="B  负尾部：κ=0 的速度可无界",
    )
    strengths = np.logspace(-8, -1, 180)
    slopes = {}
    for kappa in (0.25, 0.5, 0.75, 1.0):
        seconds = []
        for strength in strengths:
            q1 = 0.5 * strength ** (1 / kappa)
            q = np.array([q1, 1 - q1])
            teacher = np.array([0.5, 0.5])
            f = signal(np.log(teacher) - np.log(q), kappa, strength)
            second = float(q @ f**2)
            seconds.append(second)
            rows.append(
                {
                    "case": "lambda_dependent_witness",
                    "kappa": kappa,
                    "coordinate": strength,
                    "value": second,
                }
            )
        slopes[str(kappa)] = float(np.polyfit(np.log(strengths[:30]), np.log(seconds[:30]), 1)[0])
        expected = min(2, 1 / kappa)
        assert abs(slopes[str(kappa)] - expected) < 0.01
        axes[2].plot(
            strengths,
            seconds,
            color=PALETTE[kappa],
            label=f"κ={kappa:g}，斜率≈{slopes[str(kappa)]:.2f}",
        )
    axes[2].set(
        xlabel=r"插值强度 $\lambda$",
        ylabel=r"信号二阶矩 $\mathbb{E}_q f^2$",
        xscale="log",
        yscale="log",
        title="C  最坏阶数在 κ=1/2 发生变化",
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.18)
    fig.suptitle("精确有限词表诊断：A/B 为自然梯度速度；C 为随 λ 改变的概率分布见证", fontsize=12)
    save_rows("tail-threshold.csv", rows)
    save_figure(fig, "threshold")
    return slopes


def hellinger_flow_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    times = np.linspace(0, 5, 201)
    rows = []
    for initial, color in ((0.1, PALETTE[0.0]), (0.5, PALETTE[0.5]), (0.9, PALETTE[1.0])):
        affinity = np.tanh(times + np.arctanh(initial))
        normalized = (1 - affinity) / (1 - initial)
        axes[0].plot(times, normalized, color=color, label=f"A(0)={initial:g}")
        rows.extend(
            {
                "initial_affinity": initial,
                "time": tau,
                "affinity": a,
                "normalized_hellinger": d,
                "fisher_speed": 2 * np.sqrt(1 - a**2),
            }
            for tau, a, d in zip(times, affinity, normalized, strict=True)
        )
        assert np.all(normalized >= np.exp(-2 * times) - 1e-12)
        assert np.all(normalized <= np.exp(-times) + 1e-12)
    axes[0].plot(times, np.exp(-times), color="#777777", ls=":", label=r"上界 $e^{-\tau}$")
    axes[0].plot(times, np.exp(-2 * times), color="#777777", ls="--", label=r"下界 $e^{-2\tau}$")
    axes[0].set(
        xlabel=r"自然梯度流时间 $\tau$",
        ylabel="Hellinger 散度 / 初始散度",
        yscale="log",
        title="A  显式解给出两条指数包络",
    )
    distance = np.linspace(0, np.pi, 201)
    axes[1].plot(distance, 2 * np.sin(distance / 2), color=PALETTE[0.5], lw=2)
    axes[1].axhline(2, color="#777777", ls=":")
    axes[1].set(
        xlabel=r"Fisher–Rao 距离 $2\arccos A$",
        ylabel="自然梯度 Fisher 范数",
        title="B  任意概率间隙下速度不超过 2",
    )
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.18)
    save_rows("hellinger-flow.csv", rows)
    save_figure(fig, "hellinger-flow")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=font).get_name(),
            "font.size": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        }
    )
    checks = verify()
    escort_figure()
    decomposition_figure()
    convergence_figure()
    hellinger_flow_figure()
    slopes = threshold_figure()
    summary = {
        "date": "2026-10-13",
        "seed": 20261013,
        "question": "Deepen section 9.3: exact objective gaps, natural flow, escort contraction and moment threshold",
        "success_conditions": "Algebraic identities and finite differences below 1e-7; tail exponents within 0.01 of predictions",
        "rejection_conditions": "Any identity, monotonicity or asymptotic order violation rejects the corresponding claim",
        "variable_choices": "Full-support target family and exact distributions; no production training changes",
        "scope": "Analytic finite-simplex checks, not an LLM benchmark or proof of novelty",
        "checks": checks,
        "tail_witness_slopes": slopes,
        "source_commit": subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "reproduce": "uv run --no-project --with numpy --with matplotlib python experiments/analysis/proximal_family_dynamics.py",
    }
    (OUT / "dynamics-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
