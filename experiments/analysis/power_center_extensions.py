"""Bounded head-tail teacher-center estimates and conditional LLM-cache audits."""

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
    return {"tail_cases": 600, "tail_mean_max_error": max_error}


def save(fig, output, name):
    for ext in ("png", "pdf"):
        fig.savefig(
            output / f"{name}.{ext}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None,
        )
    plt.close(fig)


def figures(output):
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
    return {
        "bootstrap": "3000 response-cluster draws; same 64 responses, two fixed prefixes each; single-prompt scope",
        "prefixes": 128,
        "responses": 64,
        "native_anchor": "cached BF16; probabilities renormalized in FP64",
        "kappa": 0.5,
        "eta": 0.3,
        "heads": stats,
        "scope": "Exact scalar center conditional moments of actual q/T; no new model training",
    }


def main():
    torch.set_num_threads(8)
    repo = Path(__file__).resolve().parents[2]
    output = repo / "results/power-center-extensions"
    output.mkdir(exist_ok=True)
    summary = {"date": "2026-10-15", "checks": checks()}
    figures(output)
    summary["cached"] = cached(repo, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
