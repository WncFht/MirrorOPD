#!/usr/bin/env python3
"""Float64, full-support oracles for the rebuilt book; no model training."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/book-reconstruction"
DTYPE = torch.float64
SEED = 20261010


def tensor(values):
    return torch.tensor(values, dtype=DTYPE)


def kl(p, q):
    return (p * (p.log() - q.log())).sum()


def losses(logits, q, f, eta):
    p = logits.softmax(0)
    u = p.log() - q.log()
    g = eta * (f - q @ f)
    log_z = torch.logsumexp(q.log() + eta * f, 0)
    part = ((u - eta * f + log_z).square() * q).sum() / 2
    mean = ((u - g).square() * q).sum() / 2
    bpo = ((u - g + kl(q, p)).square() * q).sum() / (2 * eta)
    mdpo = kl(p, q) / eta - p @ f
    return torch.stack([part, mean, bpo, mdpo])


def ta(q, p):
    return q[0] * (q[0] / p[0]).log() + (1 - q[0]) * ((1 - q[0]) / (1 - p[0])).log()


def bpo_approx_field(x, q, teacher, eta):
    p = torch.cat([x, x.new_zeros(1)]).softmax(0)
    u = p.log() - q.log()
    r = teacher.log() - q.log()
    residual = u + ta(q, p) - eta * (r + ta(q, teacher))
    return (q * residual - (q @ residual) * p)[:2] / eta


def sc_field(x):
    p = torch.cat([x, x.new_zeros(1)]).softmax(0)
    teacher = tensor([0.2, 0.3, 0.5])
    r = p.log() - teacher.log()
    tail_ratio = ((1 - p[0]) / (1 - teacher[0])).log()
    base = torch.autograd.grad(ta(p, teacher), x, create_graph=True)[0]
    score = torch.eye(3, dtype=DTYPE)[1] - p
    return base + (r[1] - tail_ratio) * score[:2]


def estimator_variances(p, teacher, head):
    r = p.log() - teacher.log()
    tail = torch.ones(len(p), dtype=torch.bool)
    tail[head] = False
    proxy = torch.where(tail, torch.zeros_like(r), r)
    scalar = (p * proxy).sum() + r - proxy
    score = torch.eye(len(p), dtype=DTYPE) - p[None, :]
    g_proxy = (p[:, None] * (proxy[:, None] * score)).sum(0)
    vector = g_proxy + (r - proxy)[:, None] * score
    direct_vector = r[:, None] * score

    def variance(values):
        centered = values - (p[:, None] * values).sum(0)
        return float((p[:, None] * centered.square()).sum())

    return {
        "scalar_direct": float(p @ (r - p @ r).square()),
        "scalar_hets": float(p @ (scalar - p @ scalar).square()),
        "vector_direct_trace": variance(direct_vector),
        "vector_hets_trace": variance(vector),
    }


def main():
    torch.manual_seed(SEED)
    maxima = dict.fromkeys(
        ["quotient", "part_bpo", "mean_bpo", "mdpo_gap", "anchor_gradient", "anchor_hessian"],
        0.0,
    )
    trials = 360
    for i in range(trials):
        v = [2, 3, 5, 8][i % 4]
        q = (0.6 * torch.randn(v, dtype=DTYPE)).softmax(0)
        f = torch.randn(v, dtype=DTYPE)
        eta = [0.2, 0.7, 1.0, 1.8, 2.2][i % 5]
        x = torch.randn(v, dtype=DTYPE).requires_grad_(True)
        p = x.softmax(0)
        pg = (q.log() + eta * f).softmax(0)
        h = (p / pg).log()
        e = h - q @ h
        a = torch.logsumexp(pg.log() + e, 0)
        delta = kl(q, pg)
        ls = losses(x, q, f, eta)
        errors = {
            "quotient": (p - pg * (e - a).exp()).abs().max(),
            "part_bpo": (ls[0] - eta * ls[2] - (delta - kl(q, p)).square() / 2).abs(),
            "mean_bpo": (ls[1] - eta * ls[2] - kl(q, p).square() / 2).abs(),
            "mdpo_gap": (
                ls[3] - kl(p, pg) / eta + torch.logsumexp(q.log() + eta * f, 0) / eta
            ).abs(),
        }
        anchor = q.log().requires_grad_(True)
        objective = partial(losses, q=q, f=f, eta=eta)
        jac = torch.autograd.functional.jacobian(objective, anchor)
        fisher = torch.diag(q) - q[:, None] * q[None, :]
        errors["anchor_gradient"] = torch.stack(
            [(jac[j] + scale * fisher @ f).abs().max() for j, scale in enumerate([eta, eta, 1, 1])]
        ).max()
        errors["anchor_hessian"] = torch.stack(
            [
                (
                    torch.autograd.functional.hessian(
                        lambda z, j=j, objective=objective: objective(z)[j], anchor
                    )
                    - scale * fisher
                )
                .abs()
                .max()
                for j, scale in enumerate([1 - delta, 1, 1 / eta])
            ]
        ).max()
        for key, value in errors.items():
            maxima[key] = max(maxima[key], float(value.detach()))
    assert max(maxima.values()) < 1e-11, maxima

    q, teacher = tensor([0.5, 0.3, 0.2]), tensor([0.2, 0.3, 0.5])
    x = tensor([-0.3, 0.6]).requires_grad_(True)
    jac = torch.autograd.functional.jacobian(lambda z: bpo_approx_field(z, q, teacher, 0.7), x)
    p = torch.cat([x, x.new_zeros(1)]).softmax(0)
    expected_curl = (p[0] - q[0]) * ((1 - q[0]) * p[1] / (1 - p[0]) - q[1]) / 0.7
    curl = float((jac[0, 1] - jac[1, 0]).detach())
    assert abs(curl - float(expected_curl.detach())) < 1e-12
    jac_sc = torch.autograd.functional.jacobian(sc_field, x)
    sc_curl = float((jac_sc[1, 0] - jac_sc[0, 1]).detach())
    assert abs(sc_curl - float((p[0] * p[2] / (1 - p[0])).detach())) < 1e-12

    scalar_variance = estimator_variances(
        tensor([0.18, 0.819, 0.001]), tensor([0.08, 0.67, 0.25]), 1
    )
    vector_variance = estimator_variances(
        tensor([0.075, 0.245, 0.68]), tensor([0.12, 0.875, 0.005]), 1
    )
    assert scalar_variance["scalar_hets"] > scalar_variance["scalar_direct"]
    assert vector_variance["vector_hets_trace"] > vector_variance["vector_direct_trace"]

    restricted_q = tensor([1 / 3, 1 / 3, 1 / 3])
    restricted_teacher = tensor([0.1, 0.7, 0.2])
    restricted_f = restricted_teacher.log() - restricted_q.log()
    bpo_optimum = restricted_teacher[0].log() - restricted_teacher[1:].log().mean()
    restricted_x = tensor([float(bpo_optimum), 0, 0]).requires_grad_(True)
    restricted_ls = losses(restricted_x, restricted_q, restricted_f, 1)
    part_derivative = float(
        torch.autograd.grad(restricted_ls[0], restricted_x, retain_graph=True)[0][0]
    )
    bpo_derivative = float(torch.autograd.grad(restricted_ls[2], restricted_x)[0][0])
    assert abs(bpo_derivative) < 1e-12
    assert abs(part_derivative) > 1e-3

    trajectory = []
    finite_eta, finite_alpha, finite_steps = 0.8, 0.4, 1000
    frozen_f = teacher.log() - q.log()
    frozen_pg = (q.log() + finite_eta * frozen_f).softmax(0)
    endpoints = {}
    for name, index, scale in [("part", 0, 1), ("mean", 1, 1), ("scaled_bpo", 2, finite_eta)]:
        current = q.log().clone()
        for step in range(finite_steps + 1):
            current = current.detach().requires_grad_(True)
            population_loss = losses(current, q, frozen_f, finite_eta)[index] * scale
            grad = torch.autograd.grad(population_loss, current)[0]
            probability = current.softmax(0).detach()
            tv = float((probability - frozen_pg).abs().sum() / 2)
            if step % 10 == 0:
                trajectory.append(
                    {
                        "method": name,
                        "step": step,
                        "gibbs_tv": tv,
                        "gradient_norm": float(grad.norm()),
                    }
                )
            if step == finite_steps:
                endpoints[name] = {
                    "gibbs_tv": tv,
                    "probability": probability.tolist(),
                    "gradient_norm": float(grad.norm()),
                }
                break
            current = current - finite_alpha * grad
    assert endpoints["part"]["gibbs_tv"] < 1e-10
    assert endpoints["scaled_bpo"]["gibbs_tv"] < 1e-10
    assert endpoints["mean"]["gibbs_tv"] > 1e-3
    assert endpoints["mean"]["gradient_norm"] < 1e-10

    grid = np.linspace(-3, 7, 241)
    rows = []
    q_binary, f_binary = tensor([0.5, 0.5]), tensor([2.0, -2.0])
    for value in grid:
        ls = losses(tensor([value, 0]), q_binary, f_binary, 1)
        rows.append(
            {"logit": value, "part": float(ls[0]), "mean": float(ls[1]), "bpo": float(ls[2])}
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "finite-inner.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=trajectory[0])
        writer.writeheader()
        writer.writerows(trajectory)
    with (OUT / "landscape.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seed": SEED,
        "dtype": "float64",
        "device": "cpu",
        "trials": trials,
        "vocab_sizes": [2, 3, 5, 8],
        "eta": [0.2, 0.7, 1, 1.8, 2.2],
        "tolerance": 1e-11,
        "counterexample_config": {
            "approx_bpo": {
                "q": q.tolist(),
                "teacher": teacher.tolist(),
                "independent_logits": x.detach().tolist(),
                "last_logit": 0,
                "eta": 0.7,
                "head": [0],
            },
            "scalar_hets": {"P": [0.18, 0.819, 0.001], "Q": [0.08, 0.67, 0.25], "head": [1]},
            "vector_hets": {"P": [0.075, 0.245, 0.68], "Q": [0.12, 0.875, 0.005], "head": [1]},
            "landscape": {"q": [0.5, 0.5], "f": [2, -2], "eta": 1, "last_logit": 0},
        },
        "max_absolute_errors": maxima,
        "approx_bpo_jacobian": jac.tolist(),
        "approx_bpo_curl": curl,
        "sc_curl": sc_curl,
        "restricted_projection": {
            "q": restricted_q.tolist(),
            "teacher": restricted_teacher.tolist(),
            "bpo_optimal_first_logit": float(bpo_optimum),
            "part_derivative_at_bpo_optimum": part_derivative,
            "bpo_derivative": bpo_derivative,
        },
        "finite_inner": {
            "q": q.tolist(),
            "teacher": teacher.tolist(),
            "eta": finite_eta,
            "optimizer": "full-vocabulary ordinary logit GD",
            "alpha": finite_alpha,
            "steps": finite_steps,
            "initialization": "anchor",
            "loss_scale": {"part": 1, "mean": 1, "bpo": finite_eta},
            "endpoints": endpoints,
        },
        "scalar_variance_counterexample": scalar_variance,
        "vector_variance_counterexample": vector_variance,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(  # noqa: S603 -- fixed read-only Git arguments
            [shutil.which("git"), "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": "uv run --with torch --with numpy python experiments/toy/book_reconstruction_audit.py",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scope": "enumerated finite-vocabulary identities; no neural training or benchmark",
    }
    (OUT / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
