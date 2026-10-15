"""Canonical local events and optional W&B views of the same optimization records."""

import json
import math
import os
from numbers import Real
from pathlib import Path

from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import Job
from mirroropd.runtime.events import EventReader, environment_writer


def numeric_metrics(values):
    result = {}
    for key, value in values.items():
        if not isinstance(value, Real):
            continue
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Nonfinite reported metric: {key}")
        result[key] = float(value)
    return result


class RunTracker:
    def __init__(self, config, wandb_module=None):
        del config
        self.writer = environment_writer("driver")
        path = os.environ.get("MIRROR_JOB_PATH")
        if self.writer is None or path is None:
            raise ValueError(
                "Mirror tracking requires the versioned job launcher and event directory"
            )
        self.directory = Path(path).parent
        self.job = Job.from_dict(json.loads(Path(path).read_text()))
        self.reader = EventReader(
            self.writer.directory, cursor_path=self.directory / "tracking-cursor.json"
        )
        self.sdk = wandb_module
        self.last_update = -1
        self.history_step = (
            max(0, int(getattr(wandb_module.run, "step", 0) or 0)) if wandb_module else 0
        )
        if self.sdk:
            self._configure()

    def _configure(self):
        run = self.sdk.run
        for axis in ("outer", "update", "evaluation"):
            run.define_metric("axis/" + axis)
        for prefix in ("distill", "optimization"):
            run.define_metric(prefix + "/*", step_metric="axis/update", step_sync=False)
        run.define_metric("evaluation/*", step_metric="axis/evaluation", step_sync=False)
        for prefix in ("timing", "resources", "task", "policy", "training"):
            run.define_metric(prefix + "/*", step_metric="axis/outer", step_sync=False)
        run.summary["schema_version"] = 2
        run.summary["objective"] = self.job.objective.to_dict()
        run.summary["config_id"] = self.job.config_id
        identity_path = self.directory / "identity.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            run.summary["run_identity"] = identity
            self.sdk.config.update({"mirroropd": self.job.to_dict(), "run_identity": identity})
        atomic_json(
            self.directory / "wandb.json",
            {
                "schema_version": 2,
                "entity": run.entity,
                "project": run.project,
                "id": run.id,
                "run_id": os.environ.get("MIRROR_RUN_ID"),
                "mode": self.job.logging.wandb.effective_mode,
                "url": run.url if self.job.logging.wandb.effective_mode != "offline" else None,
            },
        )

    def _publish(self, metrics, axes):
        if self.sdk is None or not metrics:
            return
        self.history_step += 1
        self.sdk.log(metrics | axes, step=self.history_step)

    def drain(self):
        count = 0
        for event in self.reader.read():
            if event["event"] != "optimization.step":
                continue
            payload = event["payload"]
            update = int(payload["update"])
            if update <= self.last_update:
                raise ValueError("Optimization update events are duplicated or non-monotone")
            self.last_update = update
            self._publish(numeric_metrics(payload["metrics"]), {"axis/update": update})
            count += 1
        return count

    def log(self, data, step):
        self.drain()
        values = {}
        for key, value in data.items():
            if key.startswith("actor/distill/") or key in {
                "actor/grad_norm",
                "distillation/loss",
                "actor/distillation/loss",
            }:
                continue
            if key.startswith("timing_s/agent_loop/") and (
                "/num_preempted/" in key or key.endswith("/num_preempted")
            ):
                # Native -1 means unavailable, rather than a negative count.
                if value >= 0:
                    values["resources/" + key.removeprefix("timing_s/")] = value
            elif key in {
                "timing_s/agent_loop/slowest/prompt_length",
                "timing_s/agent_loop/slowest/response_length",
            }:
                name = key.removeprefix("timing_s/").removesuffix("_length")
                values["resources/" + name + "_tokens"] = value
            elif key.startswith("timing_s/"):
                values["timing/" + key.removeprefix("timing_s/") + "_seconds"] = value
            elif key.startswith("perf/"):
                values["resources/" + key.removeprefix("perf/")] = value
            elif key == "actor/lr":
                values["optimization/learning_rate"] = value
            elif key == "distill/rollout_weight_version":
                values["policy/rollout_weight_version"] = value
            elif key.startswith(("evaluation/", "training/", "task/")):
                values[key] = value
        measured = numeric_metrics(values)
        axes = {"axis/outer": step}
        if any(key.startswith("evaluation/") for key in measured):
            axes["axis/evaluation"] = step
        if "optimization/learning_rate" in measured:
            axes["axis/update"] = max(0, self.last_update)
        if "training/global_step" in measured:
            self.writer.emit(
                "training.outer_completed",
                outer_completed=step,
                updates_completed=self.last_update + 1,
                metrics={
                    key: value
                    for key, value in measured.items()
                    if not key.startswith("evaluation/")
                },
            )
        self._publish(measured, axes)
