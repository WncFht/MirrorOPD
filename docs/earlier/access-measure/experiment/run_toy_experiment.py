#!/usr/bin/env python3
"""Small reproducible OPD outer/inner comparison.

The world has S independent prefixes and A tokens.  A fixed teacher T is distilled
from an initial student by K inner updates per outer step.  At every outer step the
anchor is frozen, actions are sampled from that anchor, and the four methods reuse
the same samples.  This isolates inner geometry from rollout noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mirroropd_algorithms import (
    inner_update,
    kl_rows,
    mirror_target,
    sample_actions,
    softmax,
)

METHODS = ("pmd_part", "pmd_mean", "bpo", "mdpo")


def run(
    method: str,
    seed: int,
    states: int,
    actions: int,
    outer_steps: int,
    inner_steps: int,
    samples: int,
    eta: float,
    lr: float,
) -> dict:
    rng = np.random.default_rng(seed)
    teacher_logits = rng.normal(0.0, 1.2, size=(states, actions))
    teacher = softmax(teacher_logits)
    logits = teacher_logits + rng.normal(0.0, 2.0, size=(states, actions))
    history = []
    for outer in range(outer_steps):
        anchor = softmax(logits)
        target = mirror_target(anchor, teacher, eta)
        actions_batch = sample_actions(anchor, samples, rng)
        for inner in range(inner_steps):
            logits, diag = inner_update(
                method, logits, anchor, teacher, actions_batch, eta, lr
            )
            if not np.all(np.isfinite(logits)):
                raise FloatingPointError(
                    f"unstable {method} at outer={outer}, inner={inner}"
                )
        student = softmax(logits)
        history.append(
            {
                "outer": outer,
                "kl_student_teacher": float(np.mean(kl_rows(student, teacher))),
                "kl_student_anchor": float(np.mean(kl_rows(student, anchor))),
                "kl_student_target": float(np.mean(kl_rows(student, target))),
                "target_anchor_kl": float(np.mean(kl_rows(target, anchor))),
                "inner_update_norm": diag["update_norm"],
                "inner_coefficient_var": diag["coefficient_var"],
            }
        )
    return {
        "method": method,
        "seed": seed,
        "eta": eta,
        "lr": lr,
        "outer_steps": outer_steps,
        "inner_steps": inner_steps,
        "samples_per_prefix": samples,
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--actions", type=int, default=32)
    ap.add_argument("--outer-steps", type=int, default=12)
    ap.add_argument("--inner-steps", type=int, default=40)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--eta", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=0.25)
    args = ap.parse_args()
    records = [
        run(
            m,
            args.seed,
            args.states,
            args.actions,
            args.outer_steps,
            args.inner_steps,
            args.samples,
            args.eta,
            args.lr,
        )
        for m in METHODS
    ]
    out = {"config": vars(args), "methods": records}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for rec in records:
        h = rec["history"]
        print(
            f"{rec['method']:9s} final KL(student||teacher)={h[-1]['kl_student_teacher']:.6f} "
            f"final KL(student||target)={h[-1]['kl_student_target']:.6f}"
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for rec in records:
            h = rec["history"]
            x = [v["outer"] for v in h]
            axes[0].plot(
                x, [v["kl_student_teacher"] for v in h], marker="o", label=rec["method"]
            )
            axes[1].plot(
                x, [v["kl_student_target"] for v in h], marker="o", label=rec["method"]
            )
            axes[2].plot(
                x,
                [v["inner_coefficient_var"] for v in h],
                marker="o",
                label=rec["method"],
            )
        axes[0].set_title("student to teacher")
        axes[1].set_title("student to mirror target")
        axes[2].set_title("inner coefficient variance")
        for ax in axes:
            ax.set_xlabel("outer step")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("mean KL")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(args.out).with_suffix(".png"), dpi=160)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
