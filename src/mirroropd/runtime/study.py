"""Expand small explicit grids to ordinary Jobs and summarize audited executions."""

import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mirroropd.core.config import API_VERSION, decode
from mirroropd.runtime.audit import audit_run
from mirroropd.runtime.config import Job
from mirroropd.runtime.experiments import run_budget
from mirroropd.runtime.profiles import Selection, profile


@dataclass(frozen=True, slots=True)
class Study:
    name: str
    base: Selection
    grid: dict[str, list[str | int | float | bool | None]] = field(default_factory=dict)
    max_trials: int = 256
    schema_version: int = API_VERSION

    def __post_init__(self):
        if self.schema_version != API_VERSION:
            raise ValueError("Unsupported study schema version")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", self.name):
            raise ValueError("Study name must be a simple path-safe label")
        if self.max_trials < 1 or any(not values for values in self.grid.values()):
            raise ValueError("Study grids and max_trials must be nonempty/positive")
        if math.prod(map(len, self.grid.values())) > self.max_trials:
            raise ValueError("Study exceeds its explicit max_trials budget")


def plan_study(path, *, variables=None, overrides=()):
    """Validate every combination before creating any output directory or process.

    Profile axes replace selections; dotted axes use the ordinary Job override
    parser. Collapsed/duplicate Jobs are rejected rather than silently retrained.
    """
    path = profile(str(path), "studies")
    study = decode(Study, json.loads(path.read_text()))
    keys = sorted(study.grid)
    if any(override.partition("=")[0] in keys for override in overrides):
        raise ValueError("A CLI override cannot shadow a study grid axis")
    selection_axes = {"task", "recipe", "models", "dataset", "evaluation", "algorithm"}
    trials = []
    identities = set()
    for values in itertools.product(*(study.grid[key] for key in keys)):
        coordinates = dict(zip(keys, values, strict=True))
        selection = asdict(study.base)
        selection["variables"].update(variables or {})
        settings = [*selection["overrides"], *overrides]
        for key, value in coordinates.items():
            if key in selection_axes:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Study profile axis {key} requires nonempty strings")
                selection[key] = value
            else:
                settings.append(key + "=" + json.dumps(value, allow_nan=False))
        selection["overrides"] = settings
        job = decode(Selection, selection).load(relative_to=path.parent)
        for key, value in coordinates.items():
            if key not in selection_axes:
                resolved = job.to_dict()
                for part in key.split("."):
                    resolved = resolved[part]
                if resolved != value:
                    raise ValueError(
                        f"Study axis {key} is overridden by algorithm/profile selection"
                    )
        if job.config_id in identities:
            raise ValueError(f"Study grid normalizes to a duplicate Job: {coordinates}")
        identities.add(job.config_id)
        trials.append(
            {
                "key": "trial-" + job.config_id.removeprefix("sha256:")[:16],
                "config_id": job.config_id,
                "coordinates": coordinates,
                "job": job.to_dict(),
                "budget": run_budget(job),
            }
        )
    return {"schema_version": API_VERSION, "name": study.name, "trials": trials}


def summarize_study(directory):
    """Reaudit completed trials; retain pending and failed entries in the table."""
    from mirroropd.runtime.control import run_status

    directory = Path(directory)
    plan = json.loads((directory / "study.json").read_text())
    validate_plan(plan)
    rows = []
    for trial in plan["trials"]:
        path = directory / trial["key"]
        row = {key: trial[key] for key in ("key", "config_id", "coordinates", "budget")}
        row.update(directory=str(path.resolve()), state="pending")
        if (path / "state.json").exists():
            status = run_status(path)
            row["state"] = status["state"]
            if "error" in status:
                row["error"] = status["error"]
        if (path / "identity.json").exists():
            row["identity"] = json.loads((path / "identity.json").read_text())
        if row["state"] == "completed":
            try:
                result = audit_run(path)
                if result["requested_config_id"] != trial["config_id"]:
                    raise ValueError("Study trial identity changed")  # noqa: TRY301 -- report an invalid audited row without dropping other trials
                row["optimizer_updates"] = result["optimizer_updates"]
                row["source_id"] = result.get("source_id")
                row["executed_config_id"] = result.get("config_id")
                row["metrics"] = result["evaluations"][-1]["metrics"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(state="invalid", error=str(error))
        wandb = path / "wandb.json"
        if wandb.exists():
            row["wandb"] = json.loads(wandb.read_text())
        rows.append(row)
    return {"schema_version": API_VERSION, "name": plan["name"], "trials": rows}


def validate_plan(plan):
    """Reject edited trial identities and unsafe paths before executing/retrying."""
    if plan.get("schema_version") != API_VERSION or not plan.get("trials"):
        raise ValueError("Invalid or empty frozen study plan")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", plan.get("name", "")):
        raise ValueError("Invalid frozen study name")
    keys = set()
    for trial in plan["trials"]:
        job = Job.from_dict(trial["job"])
        key = "trial-" + job.config_id.removeprefix("sha256:")[:16]
        if trial["key"] != key or trial["config_id"] != job.config_id or key in keys:
            raise ValueError("Frozen study trial identity is inconsistent or duplicated")
        if trial["budget"] != run_budget(job):
            raise ValueError("Frozen study budget is inconsistent")
        keys.add(key)


def retry_study(directory):
    """Return a frozen plan for unsuccessful trials; callers use a new root."""
    from mirroropd.runtime.control import run_status

    directory = Path(directory).resolve()
    if run_status(directory)["state"] in {"starting", "preparing", "running", "verifying"}:
        raise ValueError("Stop the original study before retrying its unfinished trials")
    original = json.loads((directory / "study.json").read_text())
    summary = summarize_study(directory)
    pending = {item["key"] for item in summary["trials"] if item["state"] != "completed"}
    trials = [item for item in original["trials"] if item["key"] in pending]
    if not trials:
        raise ValueError("Study has no unsuccessful trials to retry")
    return original | {"trials": trials, "retry_from": str(directory)}
