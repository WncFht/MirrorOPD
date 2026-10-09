"""Verify live inner diagnostics share one monotone W&B history."""

import json
from pathlib import Path

from mirroropd.training.tracking import WandbLogger


class FakeRun:
    def __init__(self):
        self.step = 0
        self.summary = {}
        self.logged = []
        self.axes = []

    def define_metric(self, name, **kwargs):
        self.axes.append((name, kwargs))

    def log(self, data, step):
        self.logged.append((step, data))
        self.step = step


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()

    def log(self, data, step):
        self.run.log(data, step)


def test_inner_records_are_uploaded_incrementally(tmp_path: Path):
    wandb = FakeWandb()
    logger = WandbLogger(wandb, {})
    path = tmp_path / "inner.jsonl"
    path.write_text(
        json.dumps(
            {
                "outer_call": 0,
                "inner_step": 0,
                "metrics": {"mirror/importance_ess_fraction": 0.9},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert logger.log_inner_records(path) == 1
    assert logger.log_inner_records(path) == 0
    logger.log({"actor/lr": 1e-6}, step=1)
    assert [step for step, _ in wandb.run.logged] == [1, 2]
    assert wandb.run.logged[0][1]["00_axes/inner_update"] == 0
    assert wandb.run.logged[1][1]["00_axes/outer_step"] == 1


def test_test_metrics_keep_outer_axis_after_inner_uploads(tmp_path: Path):
    del tmp_path
    wandb = FakeWandb()
    logger = WandbLogger(wandb, {})
    key = "val-core/openai/gsm8k/acc/mean@1"
    logger.log({key: 0.2, "evaluation/generated_examples": 1319}, step=0)
    logger.log({"actor/grad_norm": 4.0}, step=1)
    logger.log({key: 0.3}, step=5)
    first, training, last = [payload for _, payload in wandb.run.logged]
    assert first["00_axes/evaluation_outer"] == 0
    assert first["01_quality/" + key] == 0.2
    assert first["01_quality/generated_examples"] == 1319
    assert "00_axes/evaluation_outer" not in training
    assert last["00_axes/evaluation_outer"] == 5
    assert ("01_quality/*", {"step_metric": "00_axes/evaluation_outer"}) in wandb.run.axes
    assert [step for step, _ in wandb.run.logged] == [1, 2, 3]
