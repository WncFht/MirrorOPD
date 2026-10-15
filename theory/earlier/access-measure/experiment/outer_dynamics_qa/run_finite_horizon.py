#!/usr/bin/env python3
"""Finite-horizon two-action audit of synced MirrorOPD outer dynamics.

Each outer step freezes pi_k, samples actions from it, runs N stochastic inner
updates, then hard-syncs the resulting policy as pi_{k+1}. Four estimators are
compared under fixed and linearly increasing eta schedules. The environment has
H independent time states with a terminal reward table; this isolates the
synchronization/occupancy mechanics while retaining finite-horizon returns.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirroropd_algorithms import (  # noqa: E402
    inner_update,
    kl_rows,
    mirror_target,
    sample_actions,
    softmax,
)

METHODS = ("pmd_part", "pmd_mean", "bpo", "mdpo")
SCHEDULES = ("fixed", "linear")


def reward_table(horizon: int) -> np.ndarray:
    """Deterministic reward-to-go table, shape [H, 2]."""
    t = np.arange(horizon, dtype=float)
    # Action 0 is best early; action 1 becomes best late, forcing adaptation.
    r0 = 0.65 + 0.20 * np.cos(0.9 * t)
    r1 = 0.45 + 0.20 * np.sin(0.9 * t + 0.4)
    return np.stack([r0, r1], axis=1)


def eta_at(schedule: str, outer: int, outer_steps: int, eta0: float, eta1: float) -> float:
    if schedule == "fixed":
        return eta0
    frac = outer / max(1, outer_steps - 1)
    return eta0 + frac * (eta1 - eta0)


def run(
    method: str,
    schedule: str,
    seed: int,
    horizon: int,
    outer_steps: int,
    inner_steps: int,
    samples: int,
    lr: float,
    eta0: float,
    eta1: float,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    rewards = reward_table(horizon)
    # The teacher is the Boltzmann policy of finite-horizon rewards.
    teacher = softmax(2.5 * rewards)
    logits = np.zeros((horizon, 2), dtype=float)
    records: list[dict[str, float | int | str]] = []
    for outer in range(outer_steps):
        eta = eta_at(schedule, outer, outer_steps, eta0, eta1)
        anchor = softmax(logits)
        target = mirror_target(anchor, teacher, eta)
        actions = sample_actions(anchor, samples, rng)
        last_diag: dict[str, float] = {}
        for inner in range(inner_steps):
            logits, last_diag = inner_update(method, logits, anchor, teacher, actions, eta, lr)
            if not np.all(np.isfinite(logits)):
                raise FloatingPointError(
                    f"unstable {method}/{schedule} outer={outer} inner={inner}"
                )
        policy = softmax(logits)
        # Occupancy is uniform over time states in this diagnostic MDP.
        value = float(np.mean(np.sum(policy * rewards, axis=1)))
        anchor_value = float(np.mean(np.sum(anchor * rewards, axis=1)))
        records.append(
            {
                "seed": seed,
                "method": method,
                "schedule": schedule,
                "outer": outer,
                "eta": eta,
                "value": value,
                "anchor_value": anchor_value,
                "value_gain": value - anchor_value,
                "kl_policy_teacher": float(np.mean(kl_rows(policy, teacher))),
                "kl_policy_anchor": float(np.mean(kl_rows(policy, anchor))),
                "kl_policy_target": float(np.mean(kl_rows(policy, target))),
                "target_kl_anchor": float(np.mean(kl_rows(target, anchor))),
                "inner_update_norm": last_diag.get("update_norm", np.nan),
                "inner_coefficient_var": last_diag.get("coefficient_var", np.nan),
            }
        )
    return records


def write_audit(path: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    finals = {}
    for row in rows:
        key = (row["method"], row["schedule"])
        if row["outer"] == args.outer_steps - 1:
            finals[key] = row
    lines = [
        "# MirrorOPD outer synchronization audit",
        "",
        "This experiment freezes the anchor policy pi_k at each outer step, samples from that anchor, performs N inner updates, and hard-syncs the resulting weights as pi_{k+1}.",
        "The finite horizon has H independent time states and two actions; rewards are deterministic and the occupancy is uniform over time states.",
        "",
        "## Reproducibility",
        "",
        f"Command: `python3 run_finite_horizon.py --seed {args.seed} --horizon {args.horizon} --outer-steps {args.outer_steps} --inner-steps {args.inner_steps} --samples {args.samples} --lr {args.lr} --eta0 {args.eta0} --eta1 {args.eta1}`",
        f"Seed={args.seed}; H={args.horizon}; outer={args.outer_steps}; inner={args.inner_steps}; samples={args.samples}; lr={args.lr}.",
        "",
        "## Interpretation",
        "",
        "For each row, `target_kl_anchor` is the exact entropy-mirror target displacement, while `kl_policy_target` measures the finite inner budget error. `value_gain` is the one-step reward improvement against the frozen anchor; it need not stay non-negative for a stochastic approximate inner solve.",
        "The fixed schedule keeps eta constant. The linear schedule interpolates eta0 to eta1 across outer steps. Since eta changes only at synchronization boundaries, the inner dynamics remain piecewise frozen; there is no hidden interpolation of pi_k inside an outer step.",
        "The four methods share teacher, samples, initialization, and hyperparameters. Differences therefore diagnose estimator/geometry effects rather than rollout noise.",
        "",
        "## Final rows",
        "",
        r"| method | schedule | value | KL(policy\|teacher) | KL(policy\|target) |",
        "|---|---|---:|---:|---:|",
    ]
    for (method, schedule), row in sorted(finals.items()):
        lines.append(
            f"| {method} | {schedule} | {row['value']:.6f} | {row['kl_policy_teacher']:.6f} | {row['kl_policy_target']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--outer-steps", type=int, default=16)
    ap.add_argument("--inner-steps", type=int, default=24)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.20)
    ap.add_argument("--eta0", type=float, default=0.20)
    ap.add_argument("--eta1", type=float, default=0.70)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for schedule in SCHEDULES:
        for method in METHODS:
            rows.extend(
                run(
                    method,
                    schedule,
                    args.seed,
                    args.horizon,
                    args.outer_steps,
                    args.inner_steps,
                    args.samples,
                    args.lr,
                    args.eta0,
                    args.eta1,
                )
            )
    csv_path = args.out_dir / "finite_horizon_outer_dynamics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit_path = args.out_dir / "finite_horizon_outer_dynamics_audit.md"
    write_audit(audit_path, args, rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for method in METHODS:
        for schedule in SCHEDULES:
            rs = [r for r in rows if r["method"] == method and r["schedule"] == schedule]
            x = [r["outer"] for r in rs]
            label = f"{method}/{schedule}"
            axes[0].plot(x, [r["value"] for r in rs], label=label)
            axes[1].plot(x, [r["kl_policy_target"] for r in rs], label=label)
            axes[2].plot(x, [r["value_gain"] for r in rs], label=label)
    axes[0].set_title("finite-horizon reward")
    axes[1].set_title("inner tracking error")
    axes[2].set_title("one-step value gain")
    for ax in axes:
        ax.set_xlabel("outer step k")
        ax.grid(alpha=0.25)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("mean KL(policy || target)")
    axes[0].set_ylabel("mean reward-to-go")
    axes[2].set_ylabel("value(policy k+1) - value(anchor k)")
    axes[0].legend(fontsize=6, ncol=2)
    fig.savefig(args.out_dir / "finite_horizon_outer_dynamics.png", dpi=180)
    fig.savefig(args.out_dir / "finite_horizon_outer_dynamics.pdf")
    print(csv_path)
    print(args.out_dir / "finite_horizon_outer_dynamics.png")
    print(args.out_dir / "finite_horizon_outer_dynamics.pdf")
    print(audit_path)


if __name__ == "__main__":
    main()
