"""Verify completed math runs from local records and online W&B history."""

import argparse
import csv
import json
import math
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from mirroropd.training.evaluation import audit_evaluation_records


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(run_dir):
    config = OmegaConf.load(run_dir / "resolved-config.yaml")
    outer_steps = config.trainer.total_training_steps
    epochs = config.actor_rollout_ref.actor.ppo_epochs
    if (run_dir / "exit-code.txt").read_text().strip() != "0":
        raise ValueError("Training process did not exit successfully")
    inner = read_rows(run_dir / "inner.jsonl")
    expected_indices = [(outer, step) for outer in range(outer_steps) for step in range(epochs)]
    indices = [(row["outer_call"], row["inner_step"]) for row in inner]
    if indices != expected_indices:
        raise ValueError(f"Incorrect inner updates: {indices}")
    for row in inner:
        if not all(math.isfinite(value) for value in row["metrics"].values()):
            raise ValueError("Nonfinite inner metrics")
        if row["inner_step"] == 0:
            center = row["metrics"]["mirror/student_center/rms"]
            if abs(center) > 1e-6:
                raise ValueError(f"Anchor synchronization failed: center RMS {center}")
    outer = read_rows(run_dir / "outer.jsonl")
    updates = [row for row in outer if "mirror/rollout_weight_version" in row["data"]]
    if [row["step"] for row in updates] != list(range(1, outer_steps + 1)):
        raise ValueError("Missing outer training records")
    if [row["data"]["mirror/rollout_weight_version"] for row in updates] != list(
        range(outer_steps)
    ):
        raise ValueError("Incorrect rollout weight versions")
    evaluations = [row for row in outer if "evaluation/generated_examples" in row["data"]]
    expected_evaluations = [0] if config.trainer.val_before_train else []
    frequency = config.trainer.test_freq
    if frequency > 0:
        expected_evaluations.extend(
            step
            for step in range(1, outer_steps + 1)
            if step % frequency == 0 or step == outer_steps
        )
    if [row["step"] for row in evaluations] != expected_evaluations:
        raise ValueError("Missing periodic evaluations")
    max_samples = config.data.val_max_samples
    expected_questions = min(570, max_samples) if max_samples > 0 else 570
    evaluation_audit = audit_evaluation_records(run_dir, config, evaluations, expected_questions)
    expected_examples = evaluation_audit["evaluation_examples"]
    reference = json.loads((run_dir / "wandb-run.json").read_text())
    remote = wandb.Api().run(f"{reference['entity']}/{reference['project']}/{reference['id']}")
    history = list(remote.scan_history(page_size=1000))
    remote_inner = [
        row for row in history if row.get("04_kl_centers/inner_student_center/mean") is not None
    ]
    if [row["00_axes/inner_update"] for row in remote_inner] != list(range(len(inner))):
        raise ValueError("W&B inner history is incomplete or duplicated")
    remote_evaluations = [
        row for row in history if row.get("01_quality/generated_examples") is not None
    ]
    if [row["00_axes/evaluation_outer"] for row in remote_evaluations] != expected_evaluations:
        raise ValueError("W&B evaluation history is incomplete")
    if any(
        row.get("01_quality/generated_examples") != expected_examples for row in remote_evaluations
    ):
        raise ValueError("W&B evaluation counts disagree with resolved validation protocol")
    if remote.state != "finished":
        raise ValueError(f"W&B run state is {remote.state}")
    with (run_dir / "gpu.csv").open() as handle:
        samples = list(csv.DictReader(handle))
    gpu_indices = {int(row[" index"]) for row in samples}
    if gpu_indices != {0, 1, 2, 3}:
        raise ValueError(f"Unexpected monitored GPUs: {gpu_indices}")
    peak_mib = max(float(row[" memory.used [MiB]"].split()[0]) for row in samples)
    return {
        "run": str(run_dir),
        "method": config.distillation.distillation_loss.loss_mode,
        "outer_steps": len(updates),
        "inner_updates": len(inner),
        "evaluation_steps": expected_evaluations,
        **evaluation_audit,
        "gpu_indices": sorted(gpu_indices),
        "gpu_peak_mib": peak_mib,
        "wandb": {**reference, "state": remote.state, "history_rows": len(history)},
        "evaluations": evaluations,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "evaluations"}, indent=2)
    )


if __name__ == "__main__":
    main()
