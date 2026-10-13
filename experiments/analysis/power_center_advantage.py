"""Exact teacher centers, variance witnesses and a CPU audit of cached LLM distributions."""

import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from mirroropd.proximal import minimax_velocity_shape, power_signal, worst_fisher_speed_squared

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHAPES = (0.0, 0.25, 0.5, 0.75, 1.0)
STRENGTHS = (0.03, 0.1, 0.3, 0.7)


def signals(q, teacher, kappa, eta):
    r = torch.as_tensor(np.log(teacher) - np.log(q), dtype=torch.float64)
    a = (r if eta == 1 else power_signal(r, kappa, eta) / eta).numpy()
    center = -float(q @ a)
    advantage = a + center
    positive = teacher / q - 1 - a
    return a, center, advantage, positive


def moments(q, teacher, kappa, eta):
    a, center, advantage, positive = signals(q, teacher, kappa, eta)
    norm2 = 1 - 2 * q + q @ q
    mean_grad_squared = float(np.sum((q * advantage) ** 2))
    optimal_baseline = float(q @ (a * norm2) / (q @ norm2))
    head = np.argpartition(-q, min(16, len(q)) - 1)[: min(16, len(q))]
    tail = np.ones(len(q), dtype=bool)
    tail[head] = False
    delta, mass = float(q[tail].sum()), float(teacher[tail].sum())
    lower = center
    upper = center
    if delta:
        tail_a = signals(np.array([1.0]), np.array([mass / delta]), kappa, eta)[0][0]
        lower = -float(q[head] @ a[head]) - delta * float(tail_a)
        upper = (
            -float(q[head] @ a[head]) - delta * np.log1p(-eta) / (kappa * eta)
            if kappa and eta < 1
            else None
        )
    return {
        "kappa": kappa,
        "eta": eta,
        "teacher_center": center,
        "teacher_reverse_kl": float(q @ np.log(q / teacher)),
        "advantage_variance": float(q @ advantage**2),
        "positive_center_estimator_variance": float(q @ (positive - center) ** 2),
        "uncentered_logit_gradient_variance": max(float(q @ (a**2 * norm2)) - mean_grad_squared, 0),
        "mean_centered_logit_gradient_variance": max(
            float(q @ (advantage**2 * norm2)) - mean_grad_squared, 0
        ),
        "optimal_baseline_logit_gradient_variance": max(
            float(q @ ((a - optimal_baseline) ** 2 * norm2)) - mean_grad_squared, 0
        ),
        "mean_logit_gradient_squared": mean_grad_squared,
        "q_top16_tail_mass": delta,
        "teacher_top16_tail_mass": mass,
        "bucket_center_lower": lower,
        "center_upper": upper,
    }


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"{name}.{extension}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None,
        )
    plt.close(fig)


def figures(output):
    q = np.array([0.5, 0.5])
    log_eps = np.linspace(-100, -2, 150)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for kappa in SHAPES:
        rows = [moments(q, np.array([10.0**e, 1 - 10.0**e]), kappa, 0.3) for e in log_eps]
        axes[0].plot(-log_eps, [r["teacher_center"] for r in rows], label=f"k={kappa}")
        axes[1].plot(-log_eps, [r["advantage_variance"] for r in rows], label=f"k={kappa}")
    r = np.linspace(-30, 15, 180)
    for kappa in SHAPES:
        a = (power_signal(torch.tensor(r), kappa, 0.3) / 0.3).numpy()
        axes[2].plot(r, a, label=f"k={kappa}")
    for ax, title in zip(
        axes,
        (
            "Teacher center: KL endpoint diverges",
            "Centered scalar variance: finite shape plateaus",
            "Signal A: negative tail is softened",
        ),
        strict=True,
    ):
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xlabel("-log10 teacher probability at action 1")
        ax.set_yscale("log")
    axes[2].set(xlabel="r = log(T/q)", ylabel="A = f/eta")
    save(fig, output, "negative-tail-center-variance")
    eta = np.geomspace(1e-6, 0.3, 150)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for kappa in (0.25, 0.5, 0.75, 1.0):
        values = []
        for e in eta:
            q = (
                np.array([0.25, 0.75])
                if kappa < 0.5
                else np.array([0.5 * e ** (1 / kappa), 1 - 0.5 * e ** (1 / kappa)])
            )
            teacher = np.array([0.6, 0.4]) if kappa < 0.5 else np.array([0.5, 0.5])
            values.append(moments(q, teacher, kappa, e)["advantage_variance"])
        axes[0].loglog(eta, values, label=f"k={kappa}")
    eps = np.geomspace(1e-10, 1e-2, 150)
    raw = []
    positive = []
    for e in eps:
        row = moments(np.array([e, 1 - e]), np.array([0.5, 0.5]), 0.5, 0.3)
        raw.append(row["advantage_variance"])
        positive.append(row["positive_center_estimator_variance"])
    axes[1].loglog(eps, raw, label="Signed estimator Y=-A")
    axes[1].loglog(eps, positive, label="Nonnegative estimator rho-1-A")
    axes[0].set(
        xlabel="eta (different distributions for the witnesses)",
        ylabel="Var_q(A)",
        title="Uniform small-eta threshold at k=0.5",
    )
    axes[1].set(
        xlabel="Rare anchor probability",
        ylabel="Variance of a single center estimate",
        title="Nonnegativity can reintroduce heavy tails",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend(fontsize=9)
    save(fig, output, "small-step-and-estimation")
    q = np.array([0.99, 0.01])
    teacher = np.array([0.5, 0.5])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    ks = np.linspace(0, 1, 101)
    rows = [moments(q, teacher, k, 0.3) for k in ks]
    axes[0].plot(ks, [x["teacher_center"] for x in rows], label="K decreases")
    axes[0].plot(
        ks, [x["advantage_variance"] for x in rows], label="Scalar variance need not decrease"
    )
    for key, label in (
        ("uncentered_logit_gradient_variance", "No baseline"),
        ("mean_centered_logit_gradient_variance", "Exact mean baseline"),
        ("optimal_baseline_logit_gradient_variance", "Score-weighted optimal baseline"),
    ):
        axes[1].plot(ks, [x[key] for x in rows], label=label)
    axes[0].set(xlabel="kappa", title="Smaller K is not a universal variance win")
    axes[1].set(
        xlabel="kappa",
        ylabel="Logit gradient covariance trace",
        title="Mean baseline is not variance-optimal",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    save(fig, output, "baseline-counterexample")


def response_cluster_intervals(repo, cached):
    clusters = {}
    centers = {}
    with gzip.open(repo / "data/power-center-20261014/prefix-audit.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            key = (float(row["kappa"]), float(row["eta"]), int(row["response"]))
            clusters[key] = clusters.get(key, 0.0) + float(row["advantage_variance"])
            centers.setdefault(key[:2], []).append(float(row["teacher_center"]))
    rng = np.random.default_rng(20261014)
    draws = rng.integers(0, 64, size=(2000, 64))
    for kappa in SHAPES:
        for eta in STRENGTHS:
            current = np.array([clusters[(kappa, eta, r)] for r in range(64)])
            baseline = np.array([clusters[(0.0, eta, r)] for r in range(64)])
            ratio = current[draws].sum(1) / baseline[draws].sum(1)
            cell = cached["conditions"][f"k{kappa}-eta{eta}"]
            cell["between_prefix_center_variance"] = float(np.var(centers[(kappa, eta)]))
            cell["pooled_raw_signal_variance"] = (
                cell["advantage_variance"] + cell["between_prefix_center_variance"]
            )
            cell["pooled_centered_signal_variance"] = cell["advantage_variance"]
            cell["scalar_variance_ratio_to_geometric"] = float(current.sum() / baseline.sum())
            cell["response_cluster_ratio_ci95"] = np.quantile(ratio, [0.025, 0.975]).tolist()
    cached["bootstrap"] = (
        "2000 paired response-cluster resamples across 64 responses; single prompt, not prompt-generalization uncertainty"
    )


def cached_figures(output, cached):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for eta in STRENGTHS:
        rows = [cached["conditions"][f"k{k}-eta{eta}"] for k in SHAPES]
        axes[0, 0].plot(SHAPES, [r["teacher_center"] for r in rows], marker="o", label=f"eta={eta}")
        axes[0, 1].plot(
            SHAPES, [r["advantage_variance"] for r in rows], marker="o", label=f"eta={eta}"
        )
        axes[1, 0].plot(
            SHAPES,
            [r["positive_center_estimator_variance"] / r["advantage_variance"] for r in rows],
            marker="o",
            label=f"eta={eta}",
        )
    for index, label in ((0, "median"), (1, "90th percentile")):
        axes[1, 1].plot(
            SHAPES[1:],
            [
                cached["conditions"][f"k{k}-eta0.3"]["top16_interval_width_quantiles"][index]
                for k in SHAPES[1:]
            ],
            marker="o",
            label=label,
        )
    axes[1, 0].axhline(1, color="gray", linestyle="--")
    titles = (
        "Mean exact K on 128 cached prefixes",
        "Mean conditional scalar variance",
        "Positive / signed estimator variance",
        "Top-16 certified interval width, eta=0.3",
    )
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set(xlabel="kappa", title=title)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    save(fig, output, "cached-model-center-estimation")


def sharp_limit_figure(output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    k = np.linspace(0, 2, 150)
    axes[0].plot(k, k * (4 - k))
    axes[0].set(
        xlabel="Small-eta K at kappa=0.5 = Hellinger squared",
        ylabel="Limiting scalar advantage variance",
        title="Exact identity: variance = K (4-K)",
    )
    shape = np.linspace(0.12, 0.5, 200)
    opt = minimax_velocity_shape()
    axes[1].plot(shape, [worst_fisher_speed_squared(s) for s in shape])
    axes[1].scatter([opt], [worst_fisher_speed_squared(opt)], color="#289b78")
    axes[1].set(
        xlabel="kappa",
        ylabel="Sharp worst-case small-eta scalar variance",
        title=f"Minimax scalar-variance shape: {opt:.6f}",
    )
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, output, "sharp-small-step-center-variance")


def local_estimator_figure(output):
    q = np.ones(3) / 3
    h = np.array([-1.0, -0.5, 1.5])
    eps = np.geomspace(1e-4, 0.5, 120)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for kappa, ax in zip((0.0, 0.5), axes, strict=True):
        rows = [moments(q, q * (1 + e * h), kappa, 0.3) for e in eps]
        ax.loglog(
            eps, [r["advantage_variance"] for r in rows], label="Signed estimator: order epsilon^2"
        )
        ax.loglog(
            eps,
            [r["positive_center_estimator_variance"] for r in rows],
            label="Nonnegative estimator: order epsilon^4",
        )
        ax.set(
            xlabel="epsilon in T/q = 1 + epsilon h",
            ylabel="Center estimator variance",
            title=f"Near-identical distributions, kappa={kappa}",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    save(fig, output, "local-center-estimator-variance")


def mathematical_checks():
    rng = np.random.default_rng(20261014)
    cases = 0
    worst = 0.0
    for dim in (2, 17, 127):
        for _ in range(8):
            q = np.exp(rng.normal(size=dim))
            q /= q.sum()
            teacher = np.exp(rng.normal(size=dim))
            teacher /= teacher.sum()
            kl = float(q @ np.log(q / teacher))
            for eta in STRENGTHS:
                previous = kl
                for kappa in SHAPES:
                    _a, k, b, positive = signals(q, teacher, kappa, eta)
                    assert k >= -1e-12
                    assert k <= previous + 1e-12
                    previous = k
                    np.testing.assert_allclose(q @ positive, k, atol=1e-12)
                    np.testing.assert_allclose(q @ b, 0, atol=1e-12)
                    row = moments(q, teacher, kappa, eta)
                    assert row["bucket_center_lower"] <= k + 1e-12
                    if row["center_upper"] is not None:
                        assert k <= row["center_upper"] + 1e-12
                    if kappa:
                        cap = -np.log1p(-eta) / (kappa * eta)
                        assert k <= cap + 1e-12
                        assert np.min(b) >= k - cap - 1e-12
                        if kappa <= 0.5:
                            affinity = float(q @ (teacher / q) ** kappa)
                            floor = (1 - affinity) / kappa
                            assert k >= floor - 1e-12
                            bound = (1 - affinity**2) / (kappa**2 * (1 - eta) ** 2)
                            assert q @ b**2 <= bound + 1e-12
                    worst = max(worst, abs(float(q @ positive) - k))
                    cases += 1
    example = moments(np.array([0.99, 0.01]), np.array([0.5, 0.5]), 0.5, 0.3)
    assert (
        example["mean_centered_logit_gradient_variance"]
        > example["uncentered_logit_gradient_variance"]
    )
    return {
        "cases": cases,
        "positive_representation_mean_max_error": worst,
        "baseline_counterexample": example,
    }


def cached_audit(repo):
    rows = []
    rng = np.random.default_rng(20261014)
    for rank in range(4):
        artifact = torch.load(
            repo / f"data/oneshot-effective-time-20261012/seed-42/rank-{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for index, seq in enumerate(artifact["sequences"]):
            selected = rng.choice(len(seq["selected"]), size=2, replace=False)
            for position in selected:
                qlog = seq["q_logp_bf16"][position].double()
                qlog -= qlog.logsumexp(-1)
                tlog = seq["teacher_logp"][position].double()
                tlog -= tlog.logsumexp(-1)
                q, teacher = qlog.exp().numpy(), tlog.exp().numpy()
                for kappa in SHAPES:
                    for eta in STRENGTHS:
                        row = moments(q, teacher, kappa, eta)
                        row.update(
                            response=rank * 16 + index, position=int(seq["selected"][position])
                        )
                        assert row["bucket_center_lower"] <= row["teacher_center"] + 1e-10
                        if row["center_upper"] is not None:
                            assert row["teacher_center"] <= row["center_upper"] + 1e-10
                        rows.append(row)
        del artifact
        gc.collect()
    raw = io.StringIO()
    writer = csv.DictWriter(raw, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = raw.getvalue().encode()
    relative = "data/power-center-20261014/prefix-audit.csv"
    archive = repo / (relative + ".gz")
    with (
        archive.open("wb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(content)
    manifest = json.loads((repo / "data-manifest.json").read_text())
    manifest["files"] = [x for x in manifest["files"] if x["path"] != relative] + [
        {
            "path": relative,
            "archive": relative + ".gz",
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
    ]
    (repo / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {}
    for kappa in SHAPES:
        for eta in STRENGTHS:
            subset = [r for r in rows if r["kappa"] == kappa and r["eta"] == eta]
            entry = {
                key: float(np.mean([r[key] for r in subset]))
                for key in (
                    "teacher_center",
                    "teacher_reverse_kl",
                    "advantage_variance",
                    "positive_center_estimator_variance",
                    "uncentered_logit_gradient_variance",
                    "mean_centered_logit_gradient_variance",
                    "optimal_baseline_logit_gradient_variance",
                )
            }
            if kappa:
                gaps = [r["center_upper"] - r["bucket_center_lower"] for r in subset]
                entry["top16_interval_width_quantiles"] = np.quantile(
                    gaps, [0.5, 0.9, 0.99]
                ).tolist()
                entry["top16_lower_bound_bias_mean"] = float(
                    np.mean([r["teacher_center"] - r["bucket_center_lower"] for r in subset])
                )
            summary[f"k{kappa}-eta{eta}"] = entry
    return {
        "prefixes": 128,
        "responses": 64,
        "precisions": ["native_bf16_anchor", "cached_teacher", "float64_exact_readout"],
        "rows": len(rows),
        "conditions": summary,
        "raw_sha256": hashlib.sha256(content).hexdigest(),
        "scope": "Two deterministic random reservoir positions per original response; single t22 prompt; exact conditional action moments; not new GPU training or full-parameter covariance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-models", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    repo = Path(__file__).resolve().parents[2]
    output = repo / "results/power-center-advantage"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"date": "2026-10-14", "mathematical_checks": mathematical_checks()}
    figures(output)
    sharp_limit_figure(output)
    local_estimator_figure(output)
    if args.cached_models:
        summary["cached_models"] = cached_audit(repo)
    elif (output / "summary.json").exists():
        old = json.loads((output / "summary.json").read_text())
        if "cached_models" in old:
            summary["cached_models"] = old["cached_models"]
    if "cached_models" in summary:
        response_cluster_intervals(repo, summary["cached_models"])
        cached_figures(output, summary["cached_models"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
