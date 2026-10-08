"""Verify live inner diagnostics share one monotone W&B history."""

import json
from pathlib import Path

from mirroropd.training.tracking import WandbLogger


class FakeRun:
    def __init__(self):
        self.step = 0
        self.summary = {}
        self.logged = []

    def define_metric(self, _name, **_kwargs):
        pass

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
