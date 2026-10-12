"""Prioritized full-pool LR calibration, same-state stage trials, and confirmation."""

import argparse
import json

from experiments.analysis.scalar_large_report import rows_at, summarize
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
    from experiments.plotting.plot_scalar_large import plot_phase

    plot_phase(ROOT / f"runs/scalar-large/{phase}", output / "figures")
    if phase == "fullconfirm":
        from experiments.analysis.scalar_full_pool_report import report

        selection = json.loads(
            (ROOT / "results/scalar-large/fullpool-lr-study/selection.json").read_text()
        )
        report(ROOT / "runs/scalar-large/fullconfirm", output / "comparisons", selection)


def long_check(selection):
    """Validate short-pool LR choices on a separately frozen 7168-token pool."""
    if (ROOT / "results/scalar-large/fullpool-lr-study/long-transfer-complete.json").exists():
        return
    cache = "runs/scalar-large/cache-long-s42"
    launch_phase(
        "fulllong-prepare",
        [
            (
                "cache-long-s42",
                [
                    "experiments/rollout/scalar_large_cache.py",
                    "--seed",
                    "42",
                    "--train-sequences",
                    "64",
                    "--validation-sequences",
                    "16",
                    "--response-length",
                    "7168",
                    "--output",
                    cache,
                ],
            )
        ],
        ROOT / "runs/scalar-large/fulllong-prepare-controller",
    )
    common = base(42) | {"cache": cache, "updates": 80}
    execute(
        "fulllong-canary",
        [("fulllong-canary", common | {"lr": 5e-7, "updates": 1, "audit_every": 1})],
    )
    summary = json.loads(
        (ROOT / "runs/scalar-large/fulllong-canary/fulllong-canary/summary.json").read_text()
    )
    if summary["status"] != "complete" or summary["counts"]["backwards"] != 64:
        raise RuntimeError("long-pool full-gradient canary failed")
    rates = sorted({r["lr"] for r in selection["baselines"]} | {1e-6, 2e-6, 4e-6})
    execute(
        "fulllong",
        [(f"fulllong-lr{lr:.3g}", common | {"lr": lr, "maximum": max(1.6e-5, lr)}) for lr in rates],
    )

    (ROOT / "results/scalar-large/fullpool-lr-study/long-transfer-complete.json").write_text(
        json.dumps({"status": "finished", "rates": rates, "cache": cache, "updates": 80}, indent=2)
        + "\n"
    )


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
        "long_transfer": {
            "response_length": 7168,
            "train_sequences": 64,
            "validation_sequences": 16,
            "updates": 80,
            "seed": 42,
            "rates": "selected two constants plus 1e-6, 2e-6, 4e-6",
        },
        "pilot_refinement": "add 3.2e-5 and 6.4e-5 if highest LR wins",
        "priority": [
            "active old jobs finish",
            "fullpilot",
            "fullstage",
            "fullconfirm",
            "remaining old confirm",
            "fulllong transfer check",
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
        if phase == "ablate":
            long_check(selection)
        import subprocess

        from training.scalar_large_suite import PYTHON

        subprocess.run(  # noqa: S603 -- fixed internal experiment argv
            [str(PYTHON), "-m", "training.scalar_large_suite", "--phase", phase],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
