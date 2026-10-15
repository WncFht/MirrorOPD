"""Profiles compose once; malformed settings never fall back to implicit recipes."""

import json

import pytest

from mirroropd.runtime.config import Job, load_job, materialize_defaults, with_algorithm


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def base():
    return {
        "name": "test",
        "models": {"student": "student", "teacher": "teacher"},
        "data": {"train": ["train.parquet"], "validation": ["test.parquet"]},
    }


def test_model_dataset_and_recipe_profiles_are_independent(tmp_path):
    recipe = write(tmp_path / "recipe.json", {"name": "test", "training": {"learning_rate": 2e-6}})
    models = write(
        tmp_path / "models.json", {"student": "${root}/small", "teacher": "${root}/large"}
    )
    data = write(
        tmp_path / "data.json",
        {"train": ["a.parquet"], "validation": ["b.parquet"], "scorer": {"name": "gsm8k"}},
    )
    job = load_job(
        recipe,
        models=models,
        dataset=data,
        variables={"root": "/models"},
        overrides=["objective.algorithm=on_policy", "training.inner_steps=1"],
    )
    assert job.models.student == "/models/small"
    assert job.data.scorer.name == "gsm8k"
    assert job.training.learning_rate == 2e-6
    assert job.training.inner_steps == 1
    assert job.max_context_tokens == 35840
    assert Job.from_dict(job.to_dict()) == job
    other = write(tmp_path / "other.json", {"student": "m1", "teacher": "m2"})
    changed = load_job(recipe, models=other, dataset=data)
    assert changed.models.student == "m1"
    assert changed.data == job.data
    assert changed.config_id != job.config_id


@pytest.mark.parametrize(
    "override",
    [
        "training.inner_stpes=4",
        "unknown=1",
        "objective.mirror_eta=0.2",
        'logging.wandb.enabled="false"',
        "resources.gpu_ids=[true]",
        "training.batch_size=true",
        "training.seed=1.5",
        "objective.algorithm=opd_is",
    ],
)
def test_unknown_paths_and_wrong_types_fail_before_launch(tmp_path, override):
    recipe = write(tmp_path / "recipe.json", base())
    with pytest.raises((TypeError, ValueError)):
        load_job(recipe, overrides=[override])


def test_no_implicit_environment_substitution_and_onpolicy_schedule_validation(
    tmp_path, monkeypatch
):
    value = base()
    value["models"]["student"] = "${MODEL_ROOT}/student"
    recipe = write(tmp_path / "recipe.json", value)
    monkeypatch.setenv("MODEL_ROOT", "/implicit")
    with pytest.raises(ValueError, match="Missing explicit"):
        load_job(recipe)
    value = base()
    recipe = write(recipe, value)
    with pytest.raises(ValueError, match="exactly one update"):
        load_job(recipe, overrides=["objective.algorithm=on_policy"])
    job = load_job(recipe)
    assert with_algorithm(job, "on_policy").training.inner_steps == 1
    assert with_algorithm(job, "mdpo").training.inner_steps == 16


def test_invalid_schema_paths_and_partial_batches_are_rejected():
    for update in [
        {"schema_version": 1},
        {"name": "../escape"},
        {"resources": {"gpu_ids": [0, 1, 2]}},
        {"evaluation": {"temperature": 0}},
        {"evaluation": {"samples_per_question": 0}},
        {"data": {"train": "train.parquet", "validation": ["val"]}},
    ]:
        value = base() | update
        with pytest.raises((TypeError, ValueError)):
            Job.from_dict(value)
    complete = materialize_defaults(Job, base())
    assert complete["objective"]["signal"] == {"kappa": 0.5, "eta": 0.2}
