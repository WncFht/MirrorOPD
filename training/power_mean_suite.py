"""Validated serial power-target LLM pilot on the reserved physical GPUs 0--3."""

import argparse
import atexit
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from mirroropd.proximal import minimax_velocity_shape
from training.gpu_exclusive import GpuGroupLease
from training.oneshot_suite import evaluate_model, run_worker, write_status


def study_plan(calibration):
    base = {
        "seed": 42,
        "lr": 1e-6,
        "weight_decay": 0.0,
        "precision": "bf16",
        "anchor_interval": 0,
        "wandb": True,
        "wandb_project": "mirror-opd-power-family",
        "sample_count": 4,
        "sampler": "anchor",
        "updates": 300,
        "batch_size": 64,
        "train_positions_per_response": 16,
        "heldout_positions_per_response": 16,
        "gradient_probe_draws": 4,
        "fit": "reverse_kl",
        "retry_on_gpu_conflict": False,
    }
    result = []
    for kappa in (0.0, 0.1, 0.25, minimax_velocity_shape(), 0.5, 0.75, 1.0):
        for strength in (0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7):
            identifier = f"power-bpo-sweep-b64-k{kappa:.6g}-e{strength:g}"
            result.append(
                base
                | {
                    "id": identifier,
                    "kappa": kappa,
                    "strength": strength,
                    "comparison": "fixed_strength",
                    "fit": "bpo",
                    "sample_count": 0,
                }
            )
    result.append(
        base
        | {
            "id": "power-bpo-sweep-b64-teacher-e1",
            "kappa": 0.0,
            "strength": 1.0,
            "comparison": "fixed_strength",
            "fit": "bpo",
            "sample_count": 0,
            "equivalent_kappas": [0.0, 0.1, 0.25, minimax_velocity_shape(), 0.5, 0.75, 1.0],
        }
    )
    for kappa in (0.5, 1.0):
        result.append(
            base
            | {
                "id": f"power-native-k{kappa:g}-l0.1",
                "kappa": kappa,
                "strength": 0.1,
                "fit": "power",
                "comparison": "matched_divergence",
            }
        )
        result.append(
            base
            | {
                "id": f"power-student-k{kappa:g}-l0.1",
                "kappa": kappa,
                "strength": 0.1,
                "fit": "power",
                "sampler": "student",
                "comparison": "student_action_sampling",
            }
        )
        strength = calibration["bf16"]["matched_strengths"][str(kappa)]
        result.append(
            base
            | {
                "id": f"power-radius-k{kappa:g}",
                "kappa": kappa,
                "strength": strength,
                "comparison": "matched_initial_radius",
                "fit": "bpo",
                "sample_count": 0,
            }
        )
    result.append(
        base
        | {
            "id": "power-student-k0-l0.1",
            "kappa": 0.0,
            "strength": 0.1,
            "sampler": "student",
            "comparison": "student_action_sampling",
        }
    )
    result.extend(
        base
        | {
            "id": f"power-hellinger-k{kappa:g}-l0.1",
            "kappa": kappa,
            "fit_kappa": 0.5,
            "strength": 0.1,
            "fit": "power",
            "sampler": "student",
            "comparison": "decoupled_inner_shape",
        }
        for kappa in (0.0, 1.0)
    )
    result.extend(
        base
        | {
            "id": f"power-bpo-k{kappa:g}-l0.1",
            "kappa": kappa,
            "strength": 0.1,
            "fit": "bpo",
            "comparison": "exact_bpo_residual",
        }
        for kappa in (0.0, 0.5, 1.0)
    )
    kappa = minimax_velocity_shape()
    result.extend(
        base
        | {
            "id": f"power-minimax-{fit}-l0.1",
            "kappa": kappa,
            "strength": 0.1,
            "fit": fit,
            "fit_kappa": 0.5 if fit == "power" else kappa,
            "sampler": "student" if fit == "power" else "anchor",
            "comparison": "minimax_velocity_target",
        }
        for fit in ("power",)
    )
    return result


def canary_plan(plan):
    canaries = [
        plan[i]
        | {
            "id": f"power-canary-k{plan[i]['kappa']:g}",
            "updates": 2,
            "sample_count": 0,
            "gradient_probe_draws": 8,
        }
        for i in (2, 30, 44)
    ]
    canaries.extend(
        plan[0]
        | {
            "id": f"power-canary-half-{fit}",
            "kappa": 0.5,
            "fit_kappa": 0.5,
            "fit": fit,
            "sampler": sampler,
            "updates": 2,
            "sample_count": 4,
            "gradient_probe_draws": 8,
        }
        for fit, sampler in (("power", "student"), ("bpo", "anchor"))
    )
    canaries.extend(
        plan[30]
        | {
            "id": f"power-canary-bpo-u{count}",
            "updates": 2,
            "sample_count": count,
            "probe_sample_count": count,
            "gradient_estimator": "sample_variance",
            "gradient_probe_draws": 8,
        }
        for count in (2, 4)
    )
    canaries.append(
        plan[30]
        | {
            "id": "power-canary-bpo-u4-mix0.5",
            "updates": 2,
            "sample_count": 4,
            "probe_sample_count": 4,
            "gradient_estimator": "sample_variance",
            "sampler": "anchor_teacher_mixture",
            "proposal_teacher_weight": 0.5,
            "gradient_probe_draws": 8,
        }
    )
    canaries.append(
        plan[30]
        | {
            "id": "power-canary-bpo-distinct2",
            "updates": 2,
            "sample_count": 2,
            "probe_sample_count": 2,
            "gradient_estimator": "sample_variance",
            "sampler": "anchor_distinct_pair",
            "gradient_probe_draws": 8,
        }
    )
    return canaries


def audit(output, config):
    completion = json.loads((output / "completion.json").read_text())
    rows = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    if completion["state"] != "completed" or len(rows) != config["updates"]:
        raise RuntimeError("Power training update ledger is incomplete")
    if [row["update"] for row in rows] != list(range(1, config["updates"] + 1)):
        raise RuntimeError("Power training has missing or duplicated updates")
    checkpoint = output / f"checkpoints/update-{config['updates']:06d}"
    if not (checkpoint / "model.safetensors").exists():
        raise RuntimeError("Power training did not produce its final checkpoint")
    initial = json.loads((output / "train-native.jsonl").read_text().splitlines()[0])
    if initial["anchor_to_student_kl"] > 1e-5:
        raise RuntimeError("Power training's native initial anchor does not match")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    suite = args.suite
    suite.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(
        (repo / "results/power-mean-20261013/training-radius-calibration.json").read_text()
    )
    cached = json.loads(
        (repo / "results/power-mean-20261013/cached-audit-summary.json").read_text()
    )
    if cached["prefixes"] != 4096 or not (suite / "ready.json").exists():
        raise RuntimeError("Power study requires completed CPU model audit and readiness checks")
    readiness = json.loads((suite / "ready.json").read_text())
    for relative, expected in readiness["source_sha256"].items():
        if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(
                f"Validated experiment source changed before GPU execution: {relative}"
            )
    cache = repo / "data/oneshot-effective-time-20261012/seed-42"
    plan = study_plan(calibration)
    (suite / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    status = suite / "status.json"
    directory = repo.parent / "runs/gpu-locks"
    directory.mkdir(parents=True, exist_ok=True)
    lease = GpuGroupLease(directory / "physical-gpu-0-3.lease")
    while not lease.acquire():
        write_status(status, state="waiting_for_gpu_reservation", pid=os.getpid())
        time.sleep(5)
    atexit.register(lease.release)
    canaries = canary_plan(plan)
    (suite / "canary-plan.json").write_text(json.dumps(canaries, indent=2) + "\n")
    evaluations = {
        "initial": str(repo / "runs/oneshot-effective-time-20261012/evaluation/initial-s42")
    }
    evaluation_file = suite / "evaluations.json"
    if evaluation_file.exists():
        evaluations.update(json.loads(evaluation_file.read_text()))
    for config in canaries + plan:
        output = suite / config["id"]
        if not (output / "completion.json").exists():
            run_worker(
                repo,
                suite,
                "train",
                config,
                output,
                cache,
                worker_entry="training/power_mean_worker.py",
            )
        audit(output, config)
        write_status(status, state="condition_verified", stage=config["id"], output=str(output))
        if config["comparison"] == "fixed_strength" and not config["id"].startswith("power-canary"):
            model = output / f"checkpoints/update-{config['updates']:06d}"
            evaluations[config["id"]] = str(
                evaluate_model(repo, suite, config["id"] + "-final", model)
            )
            evaluation_file.write_text(json.dumps(evaluations, indent=2) + "\n")
        subprocess.run(  # noqa: S603 -- fixed checked CPU progress analysis
            [
                str(repo.parent / "envs/opd/bin/python"),
                str(repo / "experiments/analysis/power_training_report.py"),
                "--suite",
                str(suite),
                "--output",
                str(repo / "results/power-mean-20261013"),
            ],
            cwd=repo,
            check=True,
        )
    subprocess.run(  # noqa: S603 -- fixed checked CPU analysis argv
        [
            str(repo.parent / "envs/opd/bin/python"),
            str(repo / "experiments/analysis/power_training_report.py"),
            "--suite",
            str(suite),
            "--output",
            str(repo / "results/power-mean-20261013"),
        ],
        cwd=repo,
        check=True,
    )
    write_status(
        status,
        state="all_training_and_evaluation_completed",
        conditions=len(plan),
        canaries=len(canaries),
        evaluations=len(evaluations),
    )


if __name__ == "__main__":
    main()
