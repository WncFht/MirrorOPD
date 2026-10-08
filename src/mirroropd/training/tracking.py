"""W&B presentation names; original JSONL metric names remain unchanged."""

import json
import os
from pathlib import Path


def metric_name(key: str, scope: str) -> str:
    """Partition diagnostics without losing the measured field or its scope."""
    name = key.removeprefix("actor/").removeprefix("mirror/")
    category = {
        "importance_ess": "02_offpolicy",
        "importance_ess_fraction": "02_offpolicy",
        "importance_ratio": "02_offpolicy",
        "sampled_log_ratio": "02_offpolicy",
        "grad_norm": "03_optimization",
        "score_coefficient": "03_optimization",
        "student_center": "04_kl_centers",
        "teacher_center": "04_kl_centers",
        "student_entropy": "05_entropy",
        "anchor_head_mass": "06_support_mass",
        "anchor_tail_mass": "06_support_mass",
        "student_head_mass": "06_support_mass",
        "student_tail_mass": "06_support_mass",
        "sampled_reward": "07_reward",
        "valid_tokens": "08_rollout",
    }
    field = name.split("/", 1)[0]
    if name.endswith(("/min", "/max")) and field != "importance_ratio":
        return f"90_detail_extrema/{scope}_{name}"
    if field in category:
        return f"{category[field]}/{scope}_{name}"
    if key.startswith("evaluation/"):
        return "01_quality/" + key.removeprefix("evaluation/")
    if key.startswith(("timing_s/", "timing_per_token_ms/", "perf/")) or "max_memory_" in key:
        return f"09_resources/{key}"
    if key == "actor/lr":
        return "03_optimization/learning_rate"
    fallback = (
        "08_rollout"
        if key.startswith(("response_length/", "prompt_length/", "response/"))
        else "99_framework"
    )
    return f"{fallback}/{key}"


def grouped_metrics(data: dict, scope: str) -> dict:
    return {metric_name(key, scope): value for key, value in data.items()}


def configure_wandb(run, config: dict) -> None:
    """Define axes and record a resumable reference without exposing credentials."""
    run.define_metric("00_axes/outer_step")
    run.define_metric("00_axes/inner_update")
    run.define_metric("00_axes/evaluation_outer")
    for category in (
        "02_offpolicy",
        "03_optimization",
        "04_kl_centers",
        "05_entropy",
        "06_support_mass",
        "07_reward",
        "08_rollout",
        "90_detail_extrema",
    ):
        run.define_metric(f"{category}/outer_*", step_metric="00_axes/outer_step")
        run.define_metric(f"{category}/inner_*", step_metric="00_axes/inner_update")
    run.define_metric("09_resources/*", step_metric="00_axes/outer_step")
    run.define_metric("99_framework/*", step_metric="00_axes/outer_step")
    run.define_metric("01_quality/*", step_metric="00_axes/evaluation_outer")
    loss = config.get("distillation", {}).get("distillation_loss", {})
    if loss:
        run.summary["setting/method"] = loss.get("loss_mode")
        run.summary["setting/topk"] = loss.get("topk")
        run.summary["setting/eta"] = loss.get("mirror_eta")
    path = os.environ.get("MIRROR_WANDB_RUN_PATH")
    if path:
        Path(path).write_text(
            json.dumps(
                {
                    "entity": run.entity,
                    "project": run.project,
                    "id": run.id,
                    "url": run.url,
                },
                indent=2,
            )
            + "\n"
        )


class WandbLogger:
    """Thin adapter called only for W&B; console/file logs keep their schema."""

    def __init__(self, wandb_module, config):
        self.wandb = wandb_module
        configure_wandb(wandb_module.run, config or {})

    def log(self, data, step):
        self.wandb.log({"00_axes/outer_step": step, **grouped_metrics(data, "outer")}, step=step)

    def finish(self, exit_code=0):
        self.wandb.finish(exit_code=exit_code)
