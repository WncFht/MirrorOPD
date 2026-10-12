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
                "outer_call": 1,
                "inner_step": 0,
                "epochs": 16,
                "metrics": {
                    "mirror/importance_ess_fraction": 0.9,
                    "mirror/teacher_center/mean": 0.4,
                },
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
    assert wandb.run.logged[0][1]["00_axes/outer_rollout_step"] == 1
    assert wandb.run.logged[0][1]["04_kl_centers/rollout_teacher_center/mean"] == 0.4
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
    assert (
        "01_quality/*",
        {"step_metric": "00_axes/evaluation_outer", "step_sync": False},
    ) in wandb.run.axes
    assert [step for step, _ in wandb.run.logged] == [1, 2, 3]


def test_rollout_snapshot_does_not_share_aggregate_names(tmp_path):
    wandb = FakeWandb()
    logger = WandbLogger(wandb, {})
    path = tmp_path / "inner.jsonl"
    path.write_text(
        json.dumps(
            {
                "outer_call": 0,
                "inner_step": 0,
                "epochs": 16,
                "metrics": {"mirror/student_center/mean": 0.0},
            }
        )
        + "\n"
    )
    logger.log_inner_records(path)
    logger.log({"actor/mirror/student_center/mean": 0.00048}, step=1)
    snapshot, aggregate = [payload for _, payload in wandb.run.logged]
    assert snapshot["04_kl_centers/rollout_student_center/mean"] == 0
    assert aggregate["04_kl_centers/aggregate_student_center/mean"] == 0.00048
    assert "04_kl_centers/rollout_student_center/mean" not in aggregate
    assert "00_axes/outer_rollout_step" not in aggregate
    assert all(
        kwargs.get("step_sync") is False for _, kwargs in wandb.run.axes if "step_metric" in kwargs
    )


def test_real_update_axis_survives_changed_epochs_and_multiple_minibatches(tmp_path):
    wandb = FakeWandb()
    logger = WandbLogger(wandb, {})
    path = tmp_path / "inner.jsonl"
    records = [
        {
            "outer_call": k,
            "inner_step": n,
            "epochs": epochs,
            "metrics": {"mirror/student_center/mean": 0},
        }
        for k, epochs, updates in [(0, 1, 2), (1, 3, 6)]
        for n in range(updates)
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    logger.log_inner_records(path)
    assert [data["00_axes/inner_update"] for _, data in wandb.run.logged] == list(range(8))
    logger.log({"timing_s/step": 11.0}, step=2)
    logger.log({"val-core/MATH-500/acc/mean@16": 0.6}, step=2)
    last = wandb.run.logged[-1][1]
    assert last["00_axes/updates_completed"] == 8
    assert last["00_axes/training_seconds"] == 11.0


def test_failure_recovery_skips_uploaded_rows_and_preserves_failure(tmp_path, monkeypatch):
    from training import publish_wandb_diagnostics as publisher

    run = FakeRun()
    finished = []
    run.finish = lambda **kwargs: finished.append(kwargs)
    history = [{"00_axes/inner_update": 0, "_step": 10}, {"00_axes/outer_step": 5, "_step": 11}]
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
    assert [step for step, _ in run.logged] == [13, 14]
    assert payloads[0]["00_axes/evaluation_outer"] == 0
    assert payloads[1]["00_axes/inner_update"] == 1
    assert run.summary["training/process_exit_code"] == 1
    assert finished == [{"exit_code": 1}]
