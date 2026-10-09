"""Enumerate partition gradient means and variance without a V-by-V score matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def score_sum(p, weights):
    return weights - weights.sum() * p


def partition(p, t, head):
    r = np.log(p) - np.log(t)
    h = score_sum(p, p * r * head)
    tail = ~head
    mass = p[tail].sum()
    if not tail.any():
        return r, h, h, 0.0, np.zeros_like(p), float(np.dot(p, r))
    ell = float(np.log(mass / t[tail].sum()))
    tail_score = p * tail / mass - p
    coarse = h + mass * ell * tail_score
    value = float(np.dot(p * head, r) + mass * ell)
    return r, h, coarse, ell, tail_score, value


def residual_moments(p, rho, tail, r, ell, tail_score, method):
    c = r - ell if method == "sc_ta" else r
    mean = score_sum(p, p * tail * c)
    norm_score = 1 - 2 * p + np.dot(p, p)
    squared = c * c * norm_score
    if method == "resopd":
        mean -= p[tail].sum() * ell * tail_score
        cross = tail_score - np.dot(p, tail_score)
        squared += ell * ell * np.dot(tail_score, tail_score) - 2 * r * ell * cross
    second = float(np.sum(p * p / rho * tail * squared))
    variance = second - float(np.dot(mean, mean))
    if variance < -1e-10:
        raise AssertionError(f"Negative variance: {variance}")
    return mean, max(variance, 0.0)


def autograd_checks(p, t, head, oracle, coarse, head_gradient):
    z = torch.tensor(np.log(p), dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    prob = lp.exp()
    teacher = torch.tensor(t, dtype=torch.float64)
    mask = torch.tensor(head)
    full = (prob * (lp - teacher.log())).sum()
    hval = (prob[mask] * (lp[mask] - teacher[mask].log())).sum()
    if (~mask).any():
        pm = prob[~mask].sum()
        cval = hval + pm * (pm.log() - teacher[~mask].sum().log())
    else:
        cval = hval
    errors = []
    for value, expected in ((full, oracle), (cval, coarse), (hval, head_gradient)):
        actual = torch.autograd.grad(value, z, retain_graph=True)[0].numpy()
        errors.append(float(np.max(np.abs(actual - expected))))
    return max(errors)


def run(seed, cases):
    rng = np.random.default_rng(seed)
    rows = []
    max_error = 0.0
    autograd_error = 0.0
    scalar_violations = 0
    moment_relative_error = 0.0
    strongest = None
    for vocab in (3, 17, 128, 2048):
        for case in range(cases):
            scale = (0.5, 1.0, 3.0)[case % 3]
            p, t, stale = [softmax(rng.normal(size=vocab) * scale) for _ in range(3)]
            order = np.argsort(-t, kind="stable")
            oracle = score_sum(p, p * (np.log(p) - np.log(t)))
            full_value = float(np.dot(p, np.log(p) - np.log(t)))
            previous = None
            for k in sorted({k for k in (0, 1, 4, 32, vocab - 1, vocab) if k <= vocab}):
                head = np.zeros(vocab, dtype=bool)
                head[order[:k]] = True
                r, h, coarse, ell, tail_score, value = partition(p, t, head)
                error = float(np.linalg.norm(coarse - oracle))
                mass_gradient = score_sum(p, p * head)
                if vocab <= 17:
                    autograd_error = max(
                        autograd_error,
                        autograd_checks(p, t, head, oracle, coarse, h + mass_gradient),
                    )
                if value > full_value + 1e-10:
                    scalar_violations += 1
                if previous:
                    if value < previous["value"] - 1e-10:
                        scalar_violations += 1
                    increase = error - previous["error"]
                    if increase > 1e-10 and (strongest is None or increase > strongest["increase"]):
                        strongest = {
                            "vocab": vocab,
                            "case": case,
                            "from_k": previous["k"],
                            "to_k": k,
                            "from_gradient_error": previous["error"],
                            "to_gradient_error": error,
                            "increase": increase,
                            "student": p.tolist(),
                            "teacher": t.tolist(),
                        }
                previous = {"k": k, "value": value, "error": error}
                for proposal_name, rho in (("current", p), ("stale", stale)):
                    metrics = {}
                    for method in ("hets", "resopd", "sc_ta"):
                        mean, variance = residual_moments(p, rho, ~head, r, ell, tail_score, method)
                        deterministic = h if method == "hets" else coarse
                        gradient_error = float(np.max(np.abs(deterministic + mean - oracle)))
                        max_error = max(max_error, gradient_error)
                        if vocab <= 17:
                            scores = np.eye(vocab) - p[None, :]
                            explicit = r[:, None] * scores
                            if method == "sc_ta":
                                explicit = (r - ell)[:, None] * scores
                            elif method == "resopd":
                                explicit -= ell * tail_score[None, :]
                            explicit *= ((p / rho) * (~head))[:, None]
                            explicit_mean = (rho[:, None] * explicit).sum(axis=0)
                            explicit_variance = float(
                                (rho[:, None] * (explicit - explicit_mean) ** 2).sum()
                            )
                            moment_relative_error = max(
                                moment_relative_error,
                                float(np.max(np.abs(explicit_mean - mean))),
                                abs(explicit_variance - variance) / max(1.0, explicit_variance),
                            )
                        metrics[method] = {
                            "gradient_max_abs_error": gradient_error,
                            "variance_trace": variance,
                        }
                    # True head + stopped tail has the extra gradient of head mass.
                    tail_mean = score_sum(p, p * (~head) * r)
                    legacy_bias = h + mass_gradient + tail_mean - oracle
                    legacy_error = float(np.max(np.abs(legacy_bias - mass_gradient)))
                    max_error = max(max_error, legacy_error)
                    rows.append(
                        {
                            "vocab": vocab,
                            "case": case,
                            "logit_scale": scale,
                            "k": k,
                            "proposal": proposal_name,
                            "full_kl": full_value,
                            "coarse_kl": value,
                            "coarse_gradient_bias_norm": error,
                            "legacy_bias_identity_max_abs_error": legacy_error,
                            "estimators": metrics,
                        }
                    )
    passed = (
        max_error < 1e-10
        and autograd_error < 1e-10
        and moment_relative_error < 1e-10
        and scalar_violations == 0
    )
    return {
        "configuration": {
            "seed": seed,
            "cases_per_vocab": cases,
            "vocabularies": [3, 17, 128, 2048],
            "dtype": "float64",
            "support": "nested frozen teacher top-k",
            "scope": "fixed-prefix mathematical enumeration; not upstream runtime or GPU benchmark",
        },
        "summary": {
            "passed": passed,
            "rows": len(rows),
            "gradient_max_abs_error": max_error,
            "independent_autograd_max_abs_error": autograd_error,
            "explicit_score_moment_max_scaled_error": moment_relative_error,
            "scalar_monotonicity_violations": scalar_violations,
            "gradient_monotonicity_counterexample": strongest,
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.cases)
    result["provenance"] = {
        "source_commit": subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    summary = dict(result["summary"])
    counterexample = summary.pop("gradient_monotonicity_counterexample")
    summary["gradient_monotonicity_counterexample"] = (
        None
        if counterexample is None
        else {k: v for k, v in counterexample.items() if k not in {"student", "teacher"}}
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
