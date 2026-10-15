"""Resolve task composition through the single strict Job loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from mirroropd.core.config import API_VERSION, decode
from mirroropd.runtime.config import load_job


def profile(value, category, *, relative_to=None):
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        candidate = Path(relative_to) / path
        if candidate.is_file():
            return candidate.resolve()
    if path.is_file():
        return path.resolve()
    path = Path(__file__).resolve().parents[1] / "profiles" / category / (value + ".json")
    if not path.is_file():
        raise ValueError(f"Unknown {category} profile: {value}")
    return path


@dataclass(frozen=True, slots=True)
class Task:
    recipe: str
    models: str
    dataset: str
    evaluation: str | None = None
    schema_version: int = API_VERSION

    def __post_init__(self):
        if self.schema_version != API_VERSION:
            raise ValueError("Unsupported task schema version")


@dataclass(frozen=True, slots=True)
class Selection:
    task: str | None = None
    recipe: str | None = None
    models: str | None = None
    dataset: str | None = None
    evaluation: str | None = None
    algorithm: str | None = None
    variables: dict[str, str] = field(default_factory=dict)
    overrides: list[str] = field(default_factory=list)

    def load(self, *, relative_to=None):
        values = {key: getattr(self, key) for key in ("recipe", "models", "dataset", "evaluation")}
        defaults = {}
        task_root = relative_to
        if self.task is not None:
            path = profile(self.task, "tasks", relative_to=relative_to)
            task_root = path.parent
            task = decode(Task, json.loads(path.read_text()))
            defaults = {key: getattr(task, key) for key in values}
        paths = {}
        categories = {
            "recipe": "recipes",
            "models": "models",
            "dataset": "datasets",
            "evaluation": "evaluations",
        }
        for key, category in categories.items():
            value = values[key] if values[key] is not None else defaults.get(key)
            root = relative_to if values[key] is not None else task_root
            paths[key] = profile(value, category, relative_to=root) if value is not None else None
        if paths["recipe"] is None:
            raise ValueError("Specify --task or --recipe")
        return load_job(
            **paths,
            variables=self.variables,
            overrides=self.overrides,
            algorithm=self.algorithm,
        )
