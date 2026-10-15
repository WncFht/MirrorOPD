"""Profile precedence, execution identity and explicit grid orchestration."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirroropd.cli import main
from mirroropd.runtime import control, runner, study
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import Job, with_algorithm
from mirroropd.runtime.experiments import execution_identity, run_budget
from mirroropd.runtime.profiles import Selection


def spec():
    return Job.from_dict(
        {
            "name": "example",
            "models": {"student": "s", "teacher": "t"},
            "data": {"train": ["train"], "validation": ["val"]},
            "training": {"outer_steps": 2, "inner_steps": 2},
        }
    )


def test_task_relative_profiles_and_explicit_overrides(tmp_path):
    atomic_json(tmp_path / "recipe.json", spec().to_dict())
    atomic_json(tmp_path / "model.json", {"student": "${root}/s", "teacher": "${root}/t"})
    atomic_json(tmp_path / "data.json", {"train": ["new-train"], "validation": ["new-val"]})
    atomic_json(tmp_path / "eval.json", {"samples_per_question": 3, "response_tokens": 64})
    atomic_json(
        tmp_path / "task.json",
        {
            "recipe": "recipe.json",
            "models": "model.json",
            "dataset": "data.json",
            "evaluation": "eval.json",
        },
    )
    job = Selection(
        task=str(tmp_path / "task.json"), variables={"root": "/models"}, algorithm="on_policy"
    ).load()
    assert job.models.student == "/models/s"
    assert job.data.train == ["new-train"]
    assert job.evaluation.samples_per_question == 3
    assert job.training.inner_steps == 1
    other = Selection(
        task=str(tmp_path / "task.json"),
        models="deepseek-justrl",
        variables={"model_root": "/other"},
        overrides=["evaluation.samples_per_question=5"],
    ).load()
    assert other.models.student.startswith("/other/")
    assert other.evaluation.samples_per_question == 5
    assert other.data == job.data


def test_builtin_task_cli_budget_and_unknown_task_fields(tmp_path):
    path = tmp_path / "plan.json"
    assert (
        main(
            [
                "plan",
                "--task",
                "math-smoke",
                "--algorithm",
                "on_policy",
                "--var",
                "model_root=/models",
                "--var",
                "data_root=/data",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    value = json.loads(path.read_text())
    assert value["budget"]["optimizer_updates"] == 2
    assert value["budget"]["evaluation_rounds"] == 2
    assert value["job"]["evaluation"]["response_tokens"] == 128
    atomic_json(
        tmp_path / "bad.json", {"recipe": "smoke", "models": "x", "dataset": "x", "unknown": True}
    )
    with pytest.raises(ValueError, match="Unknown Task"):
        Selection(task=str(tmp_path / "bad.json")).load()


def test_execution_identity_is_unique_persisted_and_algorithm_specific(tmp_path):
    first = execution_identity(tmp_path / "a/run", spec(), group="comparison")
    again = execution_identity(tmp_path / "a/run", spec())
    second = execution_identity(tmp_path / "b/run", spec())
    opd = execution_identity(
        tmp_path / "suite/on_policy", with_algorithm(spec(), "on_policy"), group="comparison"
    )
    assert first == again
    assert len({first["run_id"], second["run_id"], opd["run_id"]}) == 3
    assert first["group"] == opd["group"] == "comparison"
    assert "on_policy" in opd["name"]
    assert "bpo" in first["name"]
    env = runner.public_environment(spec(), tmp_path / "a/run", tmp_path / "scratch")
    assert env["MIRROR_RUN_ID"] == env["WANDB_RUN_ID"] == first["run_id"]
    assert env["WANDB_NAME"] == first["name"]
    assert env["WANDB_RUN_GROUP"] == "comparison"
    with pytest.raises(ValueError, match="does not match"):
        execution_identity(tmp_path / "a/run", with_algorithm(spec(), "mdpo"))


@pytest.mark.parametrize(
    "settings",
    [
        {"training": {"optim": {"optimizer": "SGD"}}},
        {"training": {"optim": {"betas": [0.9]}}},
        {"training": {"optim": {"betas": [0.9, 1.0]}}},
        {"training": {"optim": {"eps": 0}}},
        {"training": {"optim": {"lr_scheduler_type": "linear"}}},
        {"training": {"outer_steps": 2, "optim": {"lr_warmup_steps": 3}}},
        {"training": {"optim": {"lr_warmup_steps": 1, "lr_warmup_steps_ratio": 0.5}}},
        {"training": {"optim": {"lr_warmup_steps_ratio": float("nan")}}},
        {"checkpoints": {"max_to_keep": 0}},
        {"checkpoints": {"save_contents": ["optimizer"]}},
        {"checkpoints": {"save_contents": ["model", "model"]}},
        {"logging": {"wandb": {"tags": [""]}}},
        {"logging": {"wandb": {"tags": ["a,b"]}}},
        {"logging": {"wandb": {"tags": ["x" * 65]}}},
    ],
)
def test_native_options_fail_at_configuration_boundary(settings):
    with pytest.raises((ValueError, TypeError)):
        Job.from_dict(spec().to_dict() | settings)


def write_study(tmp_path, grid):
    atomic_json(tmp_path / "recipe.json", spec().to_dict())
    atomic_json(
        tmp_path / "grid.json",
        {"name": "comparison", "base": {"recipe": "recipe.json"}, "grid": grid},
    )
    return tmp_path / "grid.json"


def test_study_expands_all_combinations_with_budget_and_profile_axes(tmp_path):
    path = write_study(
        tmp_path, {"training.learning_rate": [1e-6, 2e-6], "training.seed": [42, 43]}
    )
    plan = study.plan_study(path)
    assert len(plan["trials"]) == len({row["key"] for row in plan["trials"]}) == 4
    assert [row["coordinates"]["training.seed"] for row in plan["trials"]] == [42, 43, 42, 43]
    assert all(row["budget"]["optimizer_updates"] == 4 for row in plan["trials"])
    assert study.plan_study(path) == plan
    with pytest.raises(ValueError, match="shadow"):
        study.plan_study(path, overrides=["training.seed=0"])
    path = write_study(tmp_path, {"algorithm": ["on_policy", "mdpo", "bpo"]})
    trials = study.plan_study(path)["trials"]
    assert [row["budget"]["optimizer_updates"] for row in trials] == [2, 4, 4]
    assert run_budget(Job.from_dict(trials[0]["job"])) == trials[0]["budget"]


@pytest.mark.parametrize(
    "grid",
    [
        {"training.learning_rate": []},
        {"training.seed": [42, 42]},
        {"training.learning_rate": [1e-6, -1]},
        {"training.sead": [42]},
        {"algorithm": ["on_policy"], "training.inner_steps": [1, 2]},
        {"training.seed": list(range(257))},
    ],
)
def test_invalid_or_collapsed_studies_fail_before_launch(tmp_path, grid):
    with pytest.raises((ValueError, TypeError)):
        study.plan_study(write_study(tmp_path, grid))
    assert not (tmp_path / "controller.json").exists()


def test_sequential_study_retains_failures_and_retries_frozen_jobs(tmp_path, monkeypatch):
    plan = study.plan_study(write_study(tmp_path, {"training.seed": [42, 43]}))
    directory = tmp_path / "execution"
    # Exercise the actual controller and artifact contract while replacing only
    # GPU execution and its independent audit (already tested with full evidence).
    monkeypatch.setattr(
        control.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=123)
    )
    monkeypatch.setattr(control.signal, "signal", lambda *_args: None)
    job = Job.from_dict(plan["trials"][0]["job"])
    control.start_run(job, directory=directory, backend="unused", python=sys.executable, study=plan)
    visited = []

    def run(child_job, *, directory, **_kwargs):
        visited.append(child_job.training.seed)
        atomic_json(
            directory / "state.json",
            {"state": "failed" if child_job.training.seed == 42 else "completed"},
        )
        if child_job.training.seed == 42:
            raise RuntimeError("injected GPU failure")

    def audit(path):
        trial = next(row for row in plan["trials"] if row["key"] == path.name)
        return {
            "requested_config_id": trial["config_id"],
            "optimizer_updates": 4,
            "evaluations": [{"metrics": {"score": 0.5}}],
        }

    monkeypatch.setattr(runner, "run_job", run)
    monkeypatch.setattr(study, "audit_run", audit)
    assert control.controller_main(directory) == 1
    assert visited == [42, 43]
    summary = study.summarize_study(directory)
    assert [row["state"] for row in summary["trials"]] == ["failed", "completed"]
    retry = study.retry_study(directory)
    assert retry["trials"] == [plan["trials"][0]]
    assert retry["retry_from"] == str(directory)
    assert json.loads((directory / "study.json").read_text()) == plan
    retry["trials"][0]["key"] = "../escape"
    with pytest.raises(ValueError, match="identity"):
        control.start_run(
            job, directory=tmp_path / "retry", backend="unused", python=sys.executable, study=retry
        )
    assert not (tmp_path / "retry").exists()


def test_study_shutdown_does_not_start_another_trial(tmp_path, monkeypatch):
    plan = study.plan_study(write_study(tmp_path, {"training.seed": [42, 43]}))
    directory = tmp_path / "interrupted"
    monkeypatch.setattr(
        control.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=123)
    )
    monkeypatch.setattr(control.signal, "signal", lambda *_args: None)
    control.start_run(
        Job.from_dict(plan["trials"][0]["job"]),
        directory=directory,
        backend="unused",
        python=sys.executable,
        study=plan,
    )
    visited = []

    def interrupted(job, **_kwargs):
        visited.append(job.training.seed)
        raise InterruptedError("stop")

    monkeypatch.setattr(runner, "run_job", interrupted)
    assert control.controller_main(directory) == 1
    assert visited == [42]
    assert json.loads((directory / "state.json").read_text())["state"] == "interrupted"


def test_automatic_output_and_suite_share_group_but_not_execution_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        control.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=123)
    )
    monkeypatch.setattr(control.signal, "signal", lambda *_args: None)
    _, record = control.start_run(
        spec(),
        directory=None,
        backend="unused",
        python=sys.executable,
        algorithms=["on_policy", "mdpo", "bpo"],
        prepare_only=True,
    )
    identities = []

    def prepared(_job, *, directory, **_kwargs):
        identities.append(json.loads((directory / "identity.json").read_text()))

    monkeypatch.setattr(runner, "run_job", prepared)
    assert control.controller_main(record["directory"]) == 0
    assert Path(record["directory"]).name.endswith(record["run_id"][:8])
    assert (
        len({item["run_id"] for item in identities})
        == len({item["name"] for item in identities})
        == 3
    )
    assert {item["group"] for item in identities} == {record["group"]}
