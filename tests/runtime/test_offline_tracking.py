"""CLI mode selection reaches the backend environment and produces real offline SDK files."""

import json
from pathlib import Path

import pytest

from mirroropd.cli import main
from mirroropd.integrations.verl.compiler import backend_values
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import Job
from mirroropd.runtime.events import EventWriter
from mirroropd.runtime.runner import public_environment
from mirroropd.runtime.tracking import RunTracker
from mirroropd.runtime.transfer import pack_transfer, unpack_transfer


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_cli_tracking_mode_compiles_to_environment_and_logger(tmp_path, mode):
    plan = tmp_path / "plan.json"
    assert (
        main(
            [
                "plan",
                "--recipe",
                "smoke",
                "--models",
                "deepseek-justrl",
                "--dataset",
                "dapo-math",
                "--var",
                "model_root=/models",
                "--var",
                "data_root=/data",
                "--wandb",
                mode,
                "--output",
                str(plan),
            ]
        )
        == 0
    )
    job = Job.from_dict(json.loads(plan.read_text())["job"])
    environment = public_environment(job, tmp_path, tmp_path / "scratch")
    assert environment["WANDB_MODE"] == mode
    assert environment["WANDB_DIR"] == str(tmp_path / "wandb")
    evidence = {
        "dataset": {"splits": {"train": {"paths": []}, "validation": {"paths": []}}},
        "models": {"teacher": {"identity": "teacher"}},
        "vocabulary_id": "tokens",
        "vocab_size": 32,
        "protocol": {},
        "evaluation_questions": 4,
    }
    compiled = backend_values(job, evidence, tmp_path, tmp_path / "source", environment)
    assert ("wandb" in compiled["trainer"]["logger"]) == (mode != "disabled")
    with pytest.raises(ValueError, match="W&B mode"):
        Job.from_dict(job.to_dict() | {"logging": {"wandb": {"mode": "unknown"}}})


def test_real_offline_sdk_records_survive_manual_transfer(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    directory = tmp_path / "run"
    directory.mkdir()
    spec = Job.from_dict(
        {
            "name": "offline",
            "models": {"student": "s", "teacher": "t"},
            "data": {"train": ["train"], "validation": ["test"]},
            "logging": {"wandb": {"enabled": True, "mode": "offline"}},
        }
    )
    atomic_json(directory / "job.json", spec.to_dict())
    atomic_json(directory / "state.json", {"state": "completed"})
    for name, value in public_environment(spec, directory, tmp_path / "scratch").items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("WANDB_BASE_URL", "http://127.0.0.1:1")
    for name in ("wandb", "wandb-cache", "wandb-data"):
        (directory / name).mkdir()
    run = wandb.init(project="mirroropd-offline-test", mode="offline", dir=str(directory / "wandb"))
    try:
        tracker = RunTracker({}, wandb)
        worker = EventWriter(directory / "events", directory.name, "worker-0")
        worker.emit(
            "optimization.step", update=0, outer=0, inner=0, metrics={"distill/reward/mean": 0.5}
        )
        assert tracker.drain() == 1
        assert run.settings.mode == "offline"
    finally:
        run.finish()
    assert list((directory / "wandb").rglob("*.wandb"))
    identity = json.loads((directory / "wandb.json").read_text())
    assert identity["mode"] == "offline"
    assert identity["url"] is None
    packed = pack_transfer("wandb", run=directory, output=tmp_path / "package")
    unpack_transfer(packed["manifest"], tmp_path / "restored")
    assert list((tmp_path / "restored").rglob("*.wandb"))
    assert Path(packed["manifest"]).exists()
