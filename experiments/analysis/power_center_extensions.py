"""Bounded tail centers, covariance BPO and exact conditional LLM-cache audits."""

import csv
import gc
import gzip
import hashlib
import io
import json
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import torch

from experiments.analysis.power_center_advantage import moments, signals

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tail_estimator(q, teacher, kappa, eta, head):
    a, center, _, phi = signals(q, teacher, kappa, eta)
    cap = -np.log1p(-eta) / (kappa * eta)
    keep = np.zeros(len(q), dtype=bool)
    keep[head] = True
    tail = ~keep
    if not tail.any():
        return {
            "lower": center,
            "upper": center,
            "width": 0.0,
            "variance": 0.0,
            "raw_tail_variance": 0.0,
            "mean_error": 0.0,
        }
    delta, mass = q[tail].sum(), teacher[tail].sum()
    rho = teacher[tail] / q[tail]
    reference = mass / delta
    abar = signals(np.array([1.0]), np.array([reference]), kappa, eta)[0][0]
    slope = reference ** (kappa - 1) / (1 - eta + eta * reference**kappa)
    intercept = cap + abar - reference * slope
    assert intercept >= -1e-10
    width = delta * (cap + abar)
    lower = float(q[keep] @ phi[keep] + delta * (reference - 1 - abar))
    upper = lower + width
    residual = -a[tail] + abar + slope * (rho - reference)
    assert residual.min() >= -1e-9
    residual = np.maximum(residual, 0)
    proposal = (slope * teacher[tail] + intercept * q[tail]) / width
    z = q[tail] * residual / proposal
    remainder = center - lower
    variance = float(proposal @ (z - remainder) ** 2)
    assert z.min() >= 0
    assert z.max() <= width + 1e-9
    assert variance <= (center - lower) * (upper - center) + 1e-9
    error = abs(float(proposal @ z) - remainder)
    assert error < 1e-9
    raw_mass = mass + (cap - 1) * delta
    raw_proposal = (teacher[tail] + (cap - 1) * q[tail]) / raw_mass
    raw_z = q[tail] * phi[tail] / raw_proposal
    raw_variance = float(raw_proposal @ (raw_z - q[tail] @ phi[tail]) ** 2)
    return {
        "lower": lower,
        "upper": float(upper),
        "width": float(width),
        "variance": variance,
        "raw_tail_variance": raw_variance,
        "mean_error": error,
    }


def logit_u_variance(q, h, eta, count):
    residual = (h - q @ h) / eta
    scalar = float(q @ residual**2)
    mean2 = float(np.sum((q * residual) ** 2))
    oracle = float(q @ (residual**2 * (1 - 2 * q + q @ q))) - mean2
    pair = scalar - mean2
    return {
        "oracle_one": oracle,
        "pair": pair,
        "u": ((count - 2) * oracle + 2 * pair) / (count * (count - 1)),
        "oracle_average": oracle / count,
        "second_moment_pair": scalar,
    }


def mixture_logit_variance(q, teacher, h, eta, count):
    residual = (h - q @ h) / eta
    proposal = (q + teacher) / 2
    measure = q * q / proposal
    s0, s1, s2 = measure.sum(), measure @ residual, measure @ residual**2
    mean2 = float(np.sum((q * residual) ** 2))
    pair = float(s0 * s2 - s1 * s1 - mean2)
    first = float((1 + q @ q) * s2 - 2 * s1 * np.sum(q * q * residual) + mean2 * s0 - 4 * mean2)
    return ((count - 2) * first + 2 * pair) / (count * (count - 1))


def checks():
    rng = np.random.default_rng(20261014)
    max_error = 0.0
    for _ in range(120):
        q = np.exp(rng.normal(size=8))
        q /= q.sum()
        t = np.exp(rng.normal(size=8))
        t /= t.sum()
        old_width = float("inf")
        old_raw = float("inf")
        order = np.argsort(-q)
        for size in (0, 1, 3, 7, 8):
            r = tail_estimator(q, t, 0.5, 0.3, order[:size])
            assert r["width"] <= old_width + 1e-10
            assert r["raw_tail_variance"] <= old_raw + 1e-10
            max_error = max(max_error, r["mean_error"])
            old_width, old_raw = r["width"], r["raw_tail_variance"]
    q = np.array([0.2, 0.3, 0.5])
    h = np.array([-1.0, 0.3, 1.7])
    eta = 0.3
    psi = np.eye(3) - np.array([0.25, 0.25, 0.5])[None, :]
    centered_psi = psi - q @ psi
    r = (h - q @ h) / eta
    mean = (q * r) @ psi
    oracle = float(q @ np.sum((r[:, None] * centered_psi - mean) ** 2, axis=1))
    pair_second = 0.0
    for x, y in product(range(3), repeat=2):
        g = (h[x] - h[y]) * (psi[x] - psi[y]) / (2 * eta)
        pair_second += q[x] * q[y] * np.sum((g - mean) ** 2)
    errors = []
    for count in (2, 3, 4):
        var = 0.0
        for outcome in product(range(3), repeat=count):
            ids = np.array(outcome)
            values = h[ids]
            scores = psi[ids]
            g = ((values - values.mean())[:, None] * scores).sum(0) / (eta * (count - 1))
            var += q[ids].prod() * np.sum((g - mean) ** 2)
        predicted = ((count - 2) * oracle + 2 * pair_second) / (count * (count - 1))
        errors.append(abs(var - predicted))
        assert errors[-1] < 1e-11
        assert predicted >= oracle / count - 1e-11
    mixture_errors = []
    teacher = np.array([0.5, 0.2, 0.3])
    proposal = (q + teacher) / 2
    true_gradient = q * (h - q @ h) / eta
    for count in (2, 3, 4):
        measured = 0.0
        for outcome in product(range(3), repeat=count):
            ids = np.array(outcome)
            w = q[ids] / proposal[ids]
            difference = h[ids, None] - h[None, ids]
            coefficients = w * (difference * w[None, :]).sum(-1) / (eta * count * (count - 1))
            gradient = np.zeros(3)
            np.add.at(gradient, ids, coefficients)
            measured += proposal[ids].prod() * np.sum((gradient - true_gradient) ** 2)
        error = abs(measured - mixture_logit_variance(q, teacher, h, eta, count))
        mixture_errors.append(error)
        assert error < 1e-10
    return {
        "mixture_covariance_identity_max_error": max(mixture_errors),
        "tail_cases": 600,
        "tail_mean_max_error": max_error,
        "covariance_trace_identity_max_error": max(errors),
    }


def save(fig, output, name):
    for ext in ("png", "pdf"):
        fig.savefig(
            output / f"{name}.{ext}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None,
        )
    plt.close(fig)


def figures(output):
    q = np.array([0.7, 0.2, 0.1])
    t = np.array([0.1, 0.2, 0.7])
    eta = 0.3
    _a, _k, b, _ = signals(q, t, 0.5, eta)
    m = moments(q, t, 0.5, eta)
    beta = np.linspace(0, 1, 101)
    covariance_trace = 1 - q @ q
    curves = np.array([logit_u_variance(q, -(1 - p) * eta * b, eta, 4)["u"] for p in beta])
    floor = m["mixture_center_estimator_variance"] * covariance_trace / 4
    oracle = np.array(
        [logit_u_variance(q, -(1 - p) * eta * b, eta, 4)["oracle_average"] for p in beta]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].semilogy(beta, oracle + floor, label="Cached K: 1 center draw, 4 gradient actions")
    axes[0].semilogy(beta, np.maximum(curves, 1e-12), label="BPO sample variance U4: no K")
    axes[0].semilogy(beta, np.maximum(oracle, 1e-12), label="Oracle two centers: 4 actions")
    axes[0].set(
        xlabel="Ideal target progress beta",
        ylabel="Exact logit gradient covariance trace",
        title="Fixed center errors leave a sampling noise floor",
    )
    counts = np.arange(2, 17)
    axes[1].plot(
        counts,
        [logit_u_variance(q, -eta * b, eta, int(n))["u"] for n in counts],
        label="BPO U statistic",
    )
    axes[1].plot(
        counts,
        [logit_u_variance(q, -eta * b, eta, int(n))["oracle_average"] for n in counts],
        label="Known exact centers",
    )
    axes[1].set(
        xlabel="Auxiliary actions per prefix",
        ylabel="Exact covariance trace",
        title="Removing centers adds variance at equal action budget",
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    save(fig, output, "center-noise-and-sample-variance")
    q = np.array([0.99, 0.01])
    t = np.array([0.5, 0.5])
    _a, k, _b, phi = signals(q, t, 0.5, 0.3)
    cap = -np.log1p(-0.3) / (0.5 * 0.3)
    grid = np.linspace(0.02, 0.98, 300)
    variance = [float(q @ (phi**2 / (1 + weight * (t / q - 1)))) - k * k for weight in grid]
    caps = np.maximum((cap - 1) / (1 - grid), 1 / grid)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(grid, variance)
    axes[1].plot(grid, caps)
    for ax in axes:
        ax.axvline(1 / cap, color="#289b78", linestyle="--", label="Minimax cap mixture")
        ax.set(xlabel="Teacher proposal weight")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Exact variance", title="Input variance criterion")
    axes[1].set(ylabel="Uniform sample cap", title="Different criterion: minimax cap")
    save(fig, output, "proposal-criteria")


def cached(repo, output):
    rows = []
    mixture_rows = []
    rng = np.random.default_rng(20261014)
    for rank in range(4):
        artifact = torch.load(
            repo / f"data/oneshot-effective-time-20261012/seed-42/rank-{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for index, seq in enumerate(artifact["sequences"]):
            positions = rng.choice(len(seq["selected"]), size=2, replace=False)
            for pos in positions:
                ql = seq["q_logp_bf16"][pos].double()
                ql -= ql.logsumexp(-1)
                tl = seq["teacher_logp"][pos].double()
                tl -= tl.logsumexp(-1)
                q, t = ql.exp().numpy(), tl.exp().numpy()
                parent = moments(q, t, 0.5, 0.3)
                teacher_signal = signals(q, t, 0.5, 0.3)[0]
                mixture_rows.append(
                    {
                        "response": rank * 16 + index,
                        "q_u4": logit_u_variance(q, -0.3 * teacher_signal, 0.3, 4)["u"],
                        "distinct_u2": (1 - float(q @ q) / 2) * parent["advantage_variance"]
                        - 1.5 * parent["mean_logit_gradient_squared"],
                        "mixture_u4": mixture_logit_variance(q, t, -0.3 * teacher_signal, 0.3, 4),
                        "mixture_u2": mixture_logit_variance(q, t, -0.3 * teacher_signal, 0.3, 2),
                    }
                )
                order = np.argsort(-q)
                previous_raw = parent["mixture_center_estimator_variance"]
                previous_width = -np.log1p(-0.3) / (0.5 * 0.3)
                for size in (16, 64, 256):
                    item = tail_estimator(q, t, 0.5, 0.3, order[:size])
                    assert item["raw_tail_variance"] <= previous_raw + 1e-9
                    assert item["width"] <= previous_width + 1e-9
                    previous_raw, previous_width = item["raw_tail_variance"], item["width"]
                    rows.append(
                        {
                            "response": rank * 16 + index,
                            "position": int(seq["selected"][pos]),
                            "head": size,
                            **item,
                            "global_envelope_variance": parent["mixture_center_estimator_variance"],
                            "signed_center_variance": parent["advantage_variance"],
                        }
                    )
        del artifact
        gc.collect()
        print("rank", rank, "audited", flush=True)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = text.getvalue().encode()
    name = "data/power-center-extensions-20261014/prefix-audit.csv"
    archive = repo / (name + ".gz")
    with archive.open("wb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
        zipped.write(raw)
    manifest = json.loads((repo / "data-manifest.json").read_text())
    manifest["files"] = [x for x in manifest["files"] if x["path"] != name] + [
        {
            "path": name,
            "archive": name + ".gz",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    ]
    (repo / "data-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    stats = {}
    cluster_draws = np.random.default_rng(20261014).integers(0, 64, size=(3000, 64))
    for head in (16, 64, 256):
        part = [r for r in rows if r["head"] == head]
        stats[str(head)] = {
            key: float(np.mean([r[key] for r in part]))
            for key in (
                "width",
                "variance",
                "raw_tail_variance",
                "global_envelope_variance",
                "signed_center_variance",
            )
        }
        stats[str(head)]["width_quantiles"] = np.quantile(
            [r["width"] for r in part], [0.5, 0.9, 0.99]
        ).tolist()
        paired = np.zeros((64, 2))
        for row in part:
            paired[row["response"]] += [row["variance"], row["global_envelope_variance"]]
        draws = paired[cluster_draws].sum(1)
        stats[str(head)]["interval_over_global_envelope_ratio"] = float(
            paired[:, 0].sum() / paired[:, 1].sum()
        )
        stats[str(head)]["response_cluster_ci95"] = np.quantile(
            draws[:, 0] / draws[:, 1], [0.025, 0.975]
        ).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    heads = [16, 64, 256]
    axes[0].plot(heads, [stats[str(h)]["width"] for h in heads], marker="o")
    axes[0].set(
        ylabel="Mean certified interval width", title="Head extension tightens sample range"
    )
    for key, label in (
        ("variance", "Tangent residual tail"),
        ("raw_tail_variance", "Exact head plus raw envelope tail"),
        ("global_envelope_variance", "Global envelope"),
        ("signed_center_variance", "Signed q estimator"),
    ):
        axes[1].plot(heads, [stats[str(h)][key] for h in heads], marker="o", label=label)
    axes[1].set(
        yscale="log",
        ylabel="Mean exact one-sample variance",
        title="128 actual-model prefixes; k=0.5, eta=0.3",
    )
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set(xlabel="Head size", xscale="log")
        ax.grid(alpha=0.2)
    save(fig, output, "cached-head-tail-estimation")
    u_stats = {}
    with gzip.open(repo / "data/power-center-20261014/prefix-audit.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            key = f"k{row['kappa']}-eta{row['eta']}"
            entry = u_stats.setdefault(key, [])
            scalar = float(row["advantage_variance"])
            g2 = float(row["mean_logit_gradient_squared"])
            vc = float(row["mean_centered_logit_gradient_variance"])
            vp = scalar - g2
            entry.append([vc / 4, (2 * vc + 2 * vp) / 12, scalar - g2])
    pair_values = [
        float(np.mean([r["distinct_u2"] for r in mixture_rows])),
        float(np.mean([r["mixture_u2"] for r in mixture_rows])),
        float(np.mean([r["q_u4"] for r in mixture_rows])),
        float(np.mean([r["mixture_u4"] for r in mixture_rows])),
    ]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.bar(
        ["q distinct pair (2)", "q/T mixture pair (2)", "q U4 (4)", "q/T mixture U4 (4)"],
        pair_values,
        color=["#3979ae", "#289b78", "#3979ae", "#289b78"],
    )
    ax.set(
        ylabel="Mean exact logit covariance trace",
        title="128 cached prefixes; counts show action budget; k=0.5, eta=0.3",
    )
    ax.tick_params(axis="x", labelsize=9)
    save(fig, output, "pair-proposal-variance")
    u_summary = {
        key: {
            "oracle_4_action_variance": float(np.mean(a, axis=0)[0]),
            "u4_variance": float(np.mean(a, axis=0)[1]),
            "u2_variance": float(np.mean(a, axis=0)[2]),
            "u4_over_oracle4_ratio": float(
                np.mean(a, axis=0)[1] / max(np.mean(a, axis=0)[0], 1e-30)
            ),
        }
        for key, a in u_stats.items()
    }
    return {
        "bootstrap": "3000 response-cluster draws; same 64 responses, two fixed prefixes each; single-prompt scope",
        "mixture_bpo": {
            "proposal": "(q+T)/2",
            "mean_u4_variance": float(np.mean([r["mixture_u4"] for r in mixture_rows])),
            "mean_q_u4_variance": float(np.mean([r["q_u4"] for r in mixture_rows])),
            "mean_distinct_u2_variance": float(np.mean([r["distinct_u2"] for r in mixture_rows])),
            "mean_u2_variance": float(np.mean([r["mixture_u2"] for r in mixture_rows])),
            "scope": "Same real q/T and logits metric; pairwise frozen q/proposal importance weights, no sample-weight normalization",
        },
        "prefixes": 128,
        "responses": 64,
        "native_anchor": "cached BF16; probabilities renormalized in FP64",
        "kappa": 0.5,
        "eta": 0.3,
        "heads": stats,
        "u_statistics_on_existing_20_cells": u_summary,
        "scope": "Exact scalar/logit conditional moments of actual q/T; ideal p-beta curves are analytic policies, not neural training histories",
    }


def main():
    torch.set_num_threads(8)
    repo = Path(__file__).resolve().parents[2]
    output = repo / "results/power-center-extensions"
    output.mkdir(exist_ok=True)
    summary = {"date": "2026-10-14", "checks": checks()}
    figures(output)
    summary["cached"] = cached(repo, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
