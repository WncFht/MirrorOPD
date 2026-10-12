"""Prioritized full-pool LR calibration, same-state stage trials, and confirmation."""

import argparse
import json

from experiments.analysis.scalar_large_report import rows_at, summarize
from experiments.plotting.plot_scalar_large import plot_phase
from training.scalar_large_suite import ROOT, base_config, launch_phase, training_commands

LRS = (1.25e-7, 2.5e-7, 5e-7, 1e-6, 2e-6, 4e-6, 8e-6, 1.6e-5)
STAGES = (40, 120, 240)


def base(seed=1042):
    return base_config(seed) | {
        "full_batch": True,
        "batch_sequences": 64,
        "audit_every": 10,
        "minimal_monitoring": True,
    }


def pilot_configs(rates=LRS):
    return [
        (
            f"fullpilot-lr{lr:.3g}",
            base()
            | {
                "lr": lr,
                "maximum": max(1.6e-5, lr),
                "updates": 240,
                "keep_checkpoints": list(STAGES),
            },
        )
        for lr in rates
    ]


def select():
    rows = rows_at(ROOT / "runs/scalar-large/fullpilot")
    valid = sorted(
        (r for r in rows if r["row"]["complete_budget"]), key=lambda r: r["row"]["log_auc"]
    )
    if len(valid) < 2:
        raise RuntimeError("fewer than two complete pilot conditions; confirmation not started")
    result = {
        "criterion": "train full-BPO normalized log-loss AUC",
        "baselines": [
            {"id": r["row"]["id"], "lr": r["summary"]["config"]["lr"]} for r in valid[:2]
        ],
        "failed_or_incomplete": [r["row"]["id"] for r in rows if not r["row"]["complete_budget"]],
    }
    path = ROOT / "results/scalar-large/fullpool-lr-study/selection.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def forks(selection):
    winner = selection["baselines"][0]
    rates = sorted(set(LRS) | {r["lr"] for r in selection["baselines"]})
    return [
        (
            f"fullstage-parent{stage}-lr{lr:.3g}",
            base()
            | {
                "lr": lr,
                "updates": 40,
                "audit_every": 5,
                "audit_steps": [1, 2],
                "fork_checkpoint": str(
                    ROOT / f"runs/scalar-large/fullpilot/{winner['id']}/checkpoint-{stage}.pt"
                ),
            },
        )
        for stage in STAGES
        for lr in rates
    ]


def confirmation(selection):
    result = []
    for seed in (42, 142, 242):
        initial = selection["baselines"][0]["lr"]
        common = base(seed) | {"updates": 600, "lr": initial, "maximum": max(1.6e-5, 2 * initial)}
        result += [
            (f"fullconfirm-constant{i + 1}-s{seed}", common | {"lr": r["lr"]})
            for i, r in enumerate(selection["baselines"])
        ]
        result += [
            (f"fullconfirm-search-s{seed}", common | {"method": "search", "search_every": 4}),
            (f"fullconfirm-feedback-s{seed}", common | {"method": "feedback", "target_rate": 0.01}),
        ]
    return result


def execute(phase, configurations):
    launch_phase(
        phase,
        training_commands(phase, configurations),
        ROOT / f"runs/scalar-large/{phase}-controller",
    )
    output = ROOT / f"results/scalar-large/fullpool-lr-study/{phase}"
    summarize(ROOT / f"runs/scalar-large/{phase}", output, None)
    plot_phase(ROOT / f"runs/scalar-large/{phase}", output / "figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "results/scalar-large/fullpool-lr-study"
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "scope": "frozen 64-rollout full gradient; existing 256-token data",
        "pilot": pilot_configs(),
        "fork_stages": list(STAGES),
        "fork_rates": list(LRS),
        "fork_updates": 40,
        "confirmation_seeds": [42, 142, 242],
        "confirmation_updates": 600,
        "pilot_refinement": "add 3.2e-5 and 6.4e-5 if highest LR wins",
        "priority": [
            "active old jobs finish",
            "fullpilot",
            "fullstage",
            "fullconfirm",
            "remaining old confirm",
            "old ablate",
        ],
        "gpu_indices": [0, 1, 2, 3],
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.plan_only:
        return
    execute(
        "fullpilot-canary",
        [("fullpilot-canary", base() | {"lr": 5e-7, "updates": 4, "audit_every": 2})],
    )
    summary = json.loads(
        (ROOT / "runs/scalar-large/fullpilot-canary/fullpilot-canary/summary.json").read_text()
    )
    if summary["status"] != "complete" or summary["counts"]["backwards"] != 256:
        raise RuntimeError("full-pool canary failed")
    execute("fullpilot", pilot_configs())
    selection = select()
    if selection["baselines"][0]["lr"] == max(LRS):
        execute("fullpilot", pilot_configs((3.2e-5, 6.4e-5)))
        selection = select()
    execute("fullstage", forks(selection))
    execute("fullconfirm", confirmation(selection))
    # The old predeclared experiments retain their original conditions.
    for phase in ("confirm", "ablate"):
        import subprocess

        from training.scalar_large_suite import PYTHON

        subprocess.run(  # noqa: S603 -- fixed internal experiment argv
            [str(PYTHON), "-m", "training.scalar_large_suite", "--phase", phase],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
