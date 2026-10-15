"""Artifact/source isolation, CLI composition, telemetry and completion gates."""

import json
from types import SimpleNamespace

import pytest

from mirroropd.cli import main
from mirroropd.identity import content_id
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.audit import audit_run
from mirroropd.runtime.config import Job
from mirroropd.runtime.evaluation import Protocol, audit_rows, request_seed
from mirroropd.runtime.events import EventWriter, environment_writer
from mirroropd.runtime.experiments import execution_identity
from mirroropd.runtime.provenance import FingerprintCache, snapshot_sources, verify_snapshot
from mirroropd.runtime.tracking import RunTracker


def job():
    return Job.from_dict(
        {
            "name": "test",
            "models": {"student": "s", "teacher": "t"},
            "data": {"train": ["tr"], "validation": ["val"]},
            "objective": {"algorithm": "on_policy"},
            "training": {"outer_steps": 2, "inner_steps": 1},
            "evaluation": {"every": 1, "samples_per_question": 2},
        }
    )


class SDK:
    def __init__(self):
        self.definitions = []
        self.history = []
        self.run = SimpleNamespace(
            step=0,
            summary={},
            entity="test",
            project="test",
            id="test",
            url="https://example.com/run",
            define_metric=lambda *args, **kwargs: self.definitions.append((args, kwargs)),
        )

    def log(self, values, step):
        self.history.append((step, values))


def test_tracking_uses_exact_worker_events_and_independent_axes(tmp_path, monkeypatch):
    spec = job()
    atomic_json(tmp_path / "job.json", spec.to_dict())
    monkeypatch.setenv("MIRROR_JOB_PATH", str(tmp_path / "job.json"))
    monkeypatch.setenv("MIRROR_EVENT_DIR", str(tmp_path / "events"))
    monkeypatch.setenv("MIRROR_RUN_ID", "run")
    sdk = SDK()
    tracker = RunTracker({}, sdk)
    worker = EventWriter(tmp_path / "events", "run", "worker-0")
    worker.emit(
        "optimization.step", update=0, outer=0, inner=0, metrics={"distill/reward/mean": 0.2}
    )
    assert tracker.drain() == 1
    assert tracker.drain() == 0
    tracker.log(
        {
            "training/global_step": 1,
            "actor/distill/reward/mean": 99,
            "distill/rollout_weight_version": 0,
            "actor/lr": 1e-6,
        },
        1,
    )
    tracker.log({"evaluation/source/mean@2": 0.5}, 1)
    assert [step for step, _ in sdk.history] == [1, 2, 3]
    assert sdk.history[0][1]["axis/update"] == 0
    assert sdk.history[-1][1]["axis/evaluation"] == 1
    assert not any("actor/distill" in key for _, row in sdk.history for key in row)
    assert sdk.run.summary["objective"] == spec.objective.to_dict()
    monkeypatch.setenv("MIRROR_EVENT_DIR", str(tmp_path / "other"))
    monkeypatch.setenv("MIRROR_RUN_ID", "other")
    assert environment_writer("driver").run_id == "other"


def test_native_timing_counts_and_durations_keep_units(tmp_path, monkeypatch):
    atomic_json(tmp_path / "job.json", job().to_dict())
    monkeypatch.setenv("MIRROR_JOB_PATH", str(tmp_path / "job.json"))
    monkeypatch.setenv("MIRROR_EVENT_DIR", str(tmp_path / "events"))
    monkeypatch.setenv("MIRROR_RUN_ID", "units")
    sdk = SDK()
    tracker = RunTracker({}, sdk)
    tracker.log(
        {
            "timing_s/agent_loop/slowest/prompt_length": 2048,
            "timing_s/agent_loop/slowest/response_length": 7168,
            "timing_s/agent_loop/num_preempted/mean": -1,
            "timing_s/agent_loop/slowest/num_preempted": 3,
            "timing_s/agent_loop/tool_calls/mean": 0.25,
            "timing_s/gen": 1.5,
        },
        1,
    )
    values = sdk.history[-1][1]
    assert values["resources/agent_loop/slowest/prompt_tokens"] == 2048
    assert values["resources/agent_loop/slowest/response_tokens"] == 7168
    assert values["resources/agent_loop/slowest/num_preempted"] == 3
    assert not any("num_preempted/mean" in key for key in values)
    assert values["timing/agent_loop/tool_calls/mean_seconds"] == 0.25
    assert values["timing/gen_seconds"] == 1.5
    assert not any("length_seconds" in key or "num_preempted_seconds" in key for key in values)


def completed_run(tmp_path):
    spec = job()
    atomic_json(tmp_path / "job.json", spec.to_dict())
    protocol = Protocol.from_job(spec, "scorer")
    atomic_json(
        tmp_path / "inputs.json", {"protocol": protocol.to_dict(), "evaluation_questions": 1}
    )
    source = tmp_path / "source"
    source.mkdir()
    atomic_json(
        source / "manifest.json", {"schema_version": 2, "source_id": content_id([]), "files": []}
    )
    atomic_json(
        tmp_path / "run.json",
        {"run_id": "run", "config_id": spec.config_id, "source_id": content_id([])},
    )
    atomic_json(tmp_path / "backend-exit.json", {"return_code": 0})
    worker = EventWriter(tmp_path / "events", "run", "worker-0")
    driver = EventWriter(tmp_path / "events", "run", "driver")
    for outer in range(2):
        worker.emit(
            "optimization.step",
            algorithm="on_policy",
            loss_kind="score_surrogate",
            update=outer,
            outer=outer,
            inner=0,
            metrics={
                "distill/valid_tokens": 2,
                "distill/student_anchor_kl/rms": 1e-8,
                "optimization/gradient_norm": 0.1,
            },
        )
        driver.emit(
            "training.outer_completed",
            outer_completed=outer + 1,
            metrics={"policy/rollout_weight_version": outer},
        )
    validation = tmp_path / "validation"
    validation.mkdir()
    for version in (0, 1, 2):
        rows = [
            {
                "schema_version": 2,
                "protocol_id": protocol.protocol_id,
                "model_version": version,
                "question_uid": "q",
                "data_source": "source",
                "sample_index": index,
                "request_seed": request_seed(42, "evaluation", version, "q", index),
                "score": float(index),
                "response_tokens": 10,
                "finish_reason": "stop",
            }
            for index in (0, 1)
        ]
        metrics = audit_rows(rows, protocol, 1, model_version=version)
        (validation / f"{version}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        driver.emit("evaluation.completed", model_version=version, metrics=metrics)
    return worker, driver


def test_completion_audit_proves_schedule_instead_of_only_exit_success(tmp_path):
    worker, _ = completed_run(tmp_path)
    assert audit_run(tmp_path)["optimizer_updates"] == 2
    worker.emit(
        "optimization.step",
        algorithm="on_policy",
        loss_kind="score_surrogate",
        update=1,
        outer=1,
        inner=0,
        metrics={},
    )
    with pytest.raises(ValueError, match="duplicated"):
        audit_run(tmp_path)


def test_new_tracking_defaults_do_not_change_a_persisted_job_identity(tmp_path):
    completed_run(tmp_path)
    recorded = json.loads((tmp_path / "job.json").read_text())
    recorded["logging"]["wandb"].pop("mode")
    atomic_json(tmp_path / "job.json", recorded)
    manifest = json.loads((tmp_path / "run.json").read_text())
    manifest["config_id"] = content_id(recorded)
    atomic_json(tmp_path / "run.json", manifest)
    assert audit_run(tmp_path)["config_id"] == manifest["config_id"]


def test_audit_binds_requested_job_and_execution_identity(tmp_path):
    completed_run(tmp_path)
    spec = job()
    identity = execution_identity(tmp_path, spec)
    atomic_json(tmp_path / "requested-job.json", spec.to_dict())
    manifest = json.loads((tmp_path / "run.json").read_text())
    manifest.update(
        run_id=identity["run_id"], identity=identity, requested_config_id=spec.config_id
    )
    atomic_json(tmp_path / "run.json", manifest)
    for path in (tmp_path / "events").glob("*.jsonl"):
        rows = [
            json.loads(line) | {"run_id": identity["run_id"]}
            for line in path.read_text().splitlines()
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert audit_run(tmp_path)["requested_config_id"] == spec.config_id
    atomic_json(tmp_path / "requested-job.json", spec.to_dict() | {"name": "edited"})
    with pytest.raises(ValueError, match="Requested job identity"):
        audit_run(tmp_path)
    atomic_json(tmp_path / "requested-job.json", spec.to_dict())
    atomic_json(tmp_path / "identity.json", identity | {"name": "edited"})
    with pytest.raises(ValueError, match="Execution identity"):
        audit_run(tmp_path)


def test_snapshot_preserves_non_python_runtime_files_and_detects_changes(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("x=1\n")
    backend = tmp_path / "backend/verl/verl"
    (backend / "version").mkdir(parents=True)
    (backend / "__init__.py").write_text("x=2\n")
    (backend / "version/version").write_text("0.9\n")
    target = tmp_path / "snapshot"
    manifest = snapshot_sources(package, tmp_path / "backend", target)
    assert (target / "verl/version/version").read_text() == "0.9\n"
    assert verify_snapshot(target)["source_id"] == manifest["source_id"]
    (target / "mirroropd/__init__.py").write_text("x=3\n")
    with pytest.raises(ValueError, match="source changed"):
        verify_snapshot(target)


def test_fingerprint_cache_tracks_changed_bytes_not_just_path(tmp_path):
    file = tmp_path / "weight.bin"
    file.write_bytes(b"a")
    cache = FingerprintCache(tmp_path / "fingerprints.json")
    first = cache.digest(file)
    cache.save()
    assert FingerprintCache(cache.path).digest(file) == first
    file.write_bytes(b"b")
    assert cache.digest(file) != first


def test_cli_profiles_do_not_require_workspace_shell(tmp_path, capsys):
    output = tmp_path / "plan.json"
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
                "model_root=/m",
                "--var",
                "data_root=/d",
                "--algorithm",
                "on_policy",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    value = json.loads(output.read_text())
    assert value["job"]["models"]["teacher"] == "/m/JustRL-DeepSeek-1.5B"
    assert value["job"]["training"]["inner_steps"] == 1
    assert (
        main(["plan", "--recipe", "smoke", "--models", "deepseek-justrl", "--dataset", "dapo-math"])
        == 1
    )
    assert "Missing explicit" in capsys.readouterr().err
