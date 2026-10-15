"""One persisted execution identity for events, W&B and local artifacts."""

import json
import re
import uuid
from pathlib import Path

from mirroropd.runtime.artifacts import atomic_json


def execution_identity(directory, job, *, group=None, run_id=None):
    """Create once in an exclusively owned run directory; verify repeated reads.

    Names describe conditions; run IDs distinguish executions even when two
    directories have the same basename. Config IDs continue to describe Jobs.
    """
    path = Path(directory) / "identity.json"
    conditions = {
        "algorithm": job.objective.algorithm,
        "seed": job.training.seed,
        "job_name": job.name,
    }
    if path.exists():
        value = json.loads(path.read_text())
        if (
            value.get("schema_version") != 2
            or not re.fullmatch(r"[0-9a-f]{32}", value.get("run_id", ""))
            or any(value.get(key) != item for key, item in conditions.items())
            or value.get("name") != execution_name(job, value["run_id"])
            or not isinstance(value.get("group"), str)
            or not value["group"].strip()
            or (
                (job.logging.wandb.group or group) is not None
                and value["group"] != (job.logging.wandb.group or group)
            )
        ):
            raise ValueError("Execution identity does not match the requested Job")
        return value
    run_id = run_id or uuid.uuid4().hex
    value = {
        "schema_version": 2,
        "run_id": run_id,
        "name": execution_name(job, run_id),
        "group": job.logging.wandb.group or group or job.name,
        **conditions,
    }
    atomic_json(path, value)
    return value


def execution_name(job, run_id):
    return f"{job.name}-{job.objective.algorithm}-s{job.training.seed}-{run_id[:8]}"


def run_budget(job):
    """Input-free counts, with token limits explicitly marked as upper bounds."""
    evaluation_rounds = (
        int(job.evaluation.before_training)
        + job.training.outer_steps // job.evaluation.every
        + int(job.training.outer_steps % job.evaluation.every != 0)
    )
    sequences = (
        job.training.outer_steps * job.training.batch_size * job.training.responses_per_prompt
    )
    return {
        "outer_steps": job.training.outer_steps,
        "optimizer_updates": job.training.outer_steps * job.training.inner_steps,
        "training_sequences": sequences,
        "training_response_tokens_upper_bound": sequences * job.training.response_tokens,
        "evaluation_rounds": evaluation_rounds,
        "samples_per_question": job.evaluation.samples_per_question,
        "question_limit": job.evaluation.question_limit,
    }
