"""Completion derives from optimizer events, input identities and complete evaluation files."""

import json
import math
from pathlib import Path

from mirroropd.identity import content_id
from mirroropd.runtime.config import Job
from mirroropd.runtime.evaluation import Protocol, audit_file
from mirroropd.runtime.events import EventReader
from mirroropd.runtime.provenance import verify_snapshot


def audit_run(directory):
    directory = Path(directory)
    recorded_job = json.loads((directory / "job.json").read_text())
    job = Job.from_dict(recorded_job)
    config_id = content_id(recorded_job)
    inputs = json.loads((directory / "inputs.json").read_text())
    manifest = json.loads((directory / "run.json").read_text())
    exit_state = json.loads((directory / "backend-exit.json").read_text())
    if exit_state["return_code"] != 0:
        raise ValueError("Native training did not exit successfully")
    snapshot = verify_snapshot(directory / "source")
    if snapshot["source_id"] != manifest["source_id"] or config_id != manifest["config_id"]:
        raise ValueError("Recorded job/source identity does not match the executed run")
    events = EventReader(directory / "events").read()
    if any(event["run_id"] != manifest["run_id"] for event in events):
        raise ValueError("Run event streams contain a different run identity")
    updates = sorted(
        (event["payload"] for event in events if event["event"] == "optimization.step"),
        key=lambda value: value["update"],
    )
    expected = job.training.outer_steps * job.training.inner_steps
    if [value["update"] for value in updates] != list(range(expected)):
        raise ValueError("Optimization updates are incomplete or duplicated")
    for value in updates:
        if (
            value["algorithm"] != job.objective.algorithm
            or value["loss_kind"] != job.objective.loss_kind
        ):
            raise ValueError("Optimization event algorithm/objective mismatch")
        if (value["outer"], value["inner"]) != divmod(value["update"], job.training.inner_steps):
            raise ValueError("Optimization window/inner coordinates are inconsistent")
        metrics = value["metrics"]
        if (
            any(not math.isfinite(item) for item in metrics.values())
            or metrics["distill/valid_tokens"] <= 0
        ):
            raise ValueError("Invalid global optimization measurements")
        if value["inner"] == 0 and metrics.get("distill/student_anchor_kl/rms", 0) > 1e-5:
            raise ValueError("Fresh-window student and frozen anchor differ")
    outer = sorted(
        (event["payload"] for event in events if event["event"] == "training.outer_completed"),
        key=lambda value: value["outer_completed"],
    )
    if [value["outer_completed"] for value in outer] != list(
        range(1, job.training.outer_steps + 1)
    ):
        raise ValueError("Completed outer boundaries are incomplete")
    if [value["metrics"]["policy/rollout_weight_version"] for value in outer] != list(
        range(job.training.outer_steps)
    ):
        raise ValueError("Rollout weight-version sequence is incorrect")
    steps = ([0] if job.evaluation.before_training else []) + [
        step
        for step in range(1, job.training.outer_steps + 1)
        if step % job.evaluation.every == 0 or step == job.training.outer_steps
    ]
    evaluations = sorted(
        (event["payload"] for event in events if event["event"] == "evaluation.completed"),
        key=lambda value: value["model_version"],
    )
    if [value["model_version"] for value in evaluations] != steps:
        raise ValueError("Evaluation schedule is incomplete or duplicated")
    protocol = Protocol.from_dict(inputs["protocol"])
    measurements = []
    for evaluation in evaluations:
        step = evaluation["model_version"]
        measured = audit_file(
            directory / "validation" / f"{step}.jsonl",
            protocol,
            inputs["evaluation_questions"],
            model_version=step,
        )
        if measured != evaluation["metrics"]:
            raise ValueError("Generation evidence disagrees with reported metrics")
        measurements.append({"model_version": step, "metrics": measured})
    return {
        "schema_version": 2,
        "passed": True,
        "run_id": manifest["run_id"],
        "source_id": manifest["source_id"],
        "config_id": config_id,
        "algorithm": job.objective.algorithm,
        "loss_kind": job.objective.loss_kind,
        "outer_steps": len(outer),
        "optimizer_updates": len(updates),
        "evaluations": measurements,
        "expected_questions": inputs["evaluation_questions"],
        "protocol": protocol.to_dict(),
    }
