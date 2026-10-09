"""Verify live inner diagnostics share one monotone W&B history."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_failure_recovery_skips_uploaded_rows_and_preserves_failure(tmp_path, monkeypatch):
    from training import publish_wandb_diagnostics as publisher

    run = FakeRun()
    finished = []
    run.finish = lambda **kwargs: finished.append(kwargs)
    history = [{"00_axes/inner_update": 0}, {"00_axes/outer_step": 5}]
    fake = SimpleNamespace(
        Api=lambda: SimpleNamespace(
            run=lambda _path: SimpleNamespace(scan_history=lambda: iter(history))
        ),
        init=lambda **_kwargs: run,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv("MIRROR_WANDB_RUN_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["publisher", "--run", str(tmp_path), "--include-outer"])
    (tmp_path / "wandb-run.json").write_text(
        json.dumps({"entity": "entity", "project": "project", "id": "id"})
    )
    (tmp_path / "inner.jsonl").write_text(
        "\n".join(
            json.dumps({"outer_call": 0, "inner_step": step, "metrics": {"grad_norm": 2}})
            for step in range(2)
        )
    )
    (tmp_path / "outer.jsonl").write_text(
        "\n".join(
            json.dumps({"step": step, "data": {"evaluation/generated_examples": 16}})
            for step in (0, 5)
        )
    )
    (tmp_path / "exit-code.txt").write_text("1\n")
    publisher.main()
    payloads = [payload for _, payload in run.logged]
    assert len(payloads) == 2
    assert payloads[0]["00_axes/evaluation_outer"] == 0
    assert payloads[1]["00_axes/inner_update"] == 1
    assert run.summary["training/process_exit_code"] == 1
    assert finished == [{"exit_code": 1}]
