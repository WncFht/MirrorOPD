"""One strict job schema, composed from model, dataset and recipe profiles."""

import copy
import dataclasses
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mirroropd.core.config import API_VERSION, Objective, decode, strict_fields
from mirroropd.identity import content_id
from mirroropd.runtime.memory import Memory


@dataclass(frozen=True, slots=True)
class Models:
    student: str
    teacher: str
    chat_template: str | None = None
    trust_remote_code: bool = False

    def __post_init__(self):
        if not self.student or not self.teacher:
            raise ValueError("Student and teacher artifact paths are required")


@dataclass(frozen=True, slots=True)
class Scorer:
    name: str = "exact"
    kind: str = "binary"
    parameters: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"binary", "scalar"}:
            raise ValueError("Scorer kind must be binary or scalar")
        if self.name not in {"exact", "gsm8k", "math"} and ":" not in self.name:
            raise ValueError("Custom scorer must use module:function syntax")
        if set(self.parameters) & {"data_source", "solution_str", "ground_truth", "extra_info"}:
            raise ValueError("Scorer parameters cannot override example fields")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in self.parameters.values()
        ):
            raise ValueError("Scorer parameters must contain finite JSON values")


@dataclass(frozen=True, slots=True)
class Columns:
    prompt: str = "prompt"
    answer: str = "reward_model.ground_truth"
    source: str = "data_source"


@dataclass(frozen=True, slots=True)
class Dataset:
    train: list[str]
    validation: list[str]
    scorer: Scorer = field(default_factory=Scorer)
    columns: Columns = field(default_factory=Columns)
    source_name: str | None = None
    ability: str = "general"
    requires_validation_reference: bool = True

    def __post_init__(self):
        if not self.train or not self.validation:
            raise ValueError("Explicit training and validation parquet paths are required")
        if any(
            not isinstance(value, str) or not value for value in (*self.train, *self.validation)
        ):
            raise ValueError("Dataset paths must be nonempty strings")


@dataclass(frozen=True, slots=True)
class Precision:
    parameters: str = "fp32"
    teacher: str = "bf16"
    forward: str = "bf16"
    reduction: str = "fp32"
    buffers: str = "fp32"

    def __post_init__(self):
        if any(value not in {"fp32", "bf16"} for value in asdict(self).values()):
            raise ValueError("Precision fields must be fp32 or bf16")


@dataclass(frozen=True, slots=True)
class Optimizer:
    """Validated options implemented by the pinned verl FSDP optimizer.

    The native worker advances its scheduler once per outer, after all inner
    updates. Warmup and decay therefore use outer units, not optimizer updates.
    """

    optimizer: str = "AdamW"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    lr_scheduler_type: str = "constant"
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5

    def __post_init__(self):
        if self.optimizer not in {"Adam", "AdamW"}:
            raise ValueError("The pinned FSDP integration supports torch.optim Adam/AdamW")
        if len(self.betas) != 2 or any(not math.isfinite(x) or not 0 <= x < 1 for x in self.betas):
            raise ValueError("Optimizer betas must contain two finite values in [0, 1)")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Optimizer eps must be finite and positive")
        if self.lr_scheduler_type not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler_type must be constant or cosine")
        if self.lr_warmup_steps < -1:
            raise ValueError("lr_warmup_steps must be -1 or nonnegative")
        if any(
            not math.isfinite(x) or not 0 <= x <= 1
            for x in (self.lr_warmup_steps_ratio, self.min_lr_ratio)
        ):
            raise ValueError("Warmup and minimum learning-rate ratios must be in [0, 1]")
        if not math.isfinite(self.num_cycles) or self.num_cycles <= 0:
            raise ValueError("num_cycles must be finite and positive")


@dataclass(frozen=True, slots=True)
class Training:
    outer_steps: int = 50
    inner_steps: int = 16
    batch_size: int = 64
    micro_batch_size: int = 1
    responses_per_prompt: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    clip_grad: float = 2.0
    prompt_tokens: int = 2048
    response_tokens: int = 7168
    temperature: float = 1.0
    seed: int = 42
    gradient_checkpointing: bool = True
    parameter_offload: bool = True
    optimizer_offload: bool = True
    optim: Optimizer = field(default_factory=Optimizer)

    def __post_init__(self):
        for name in (
            "outer_steps",
            "inner_steps",
            "batch_size",
            "micro_batch_size",
            "responses_per_prompt",
            "prompt_tokens",
            "response_tokens",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        for name in ("learning_rate", "temperature", "clip_grad"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("training.weight_decay must be finite and nonnegative")
        if not 0 <= self.seed < 2**31:
            raise ValueError("training.seed must be in [0, 2**31)")
        if self.optim.lr_warmup_steps > self.outer_steps:
            raise ValueError("Optimizer warmup cannot exceed training.outer_steps")
        if self.optim.lr_warmup_steps > 0 and self.optim.lr_warmup_steps_ratio != 0:
            raise ValueError("Set either warmup steps or a warmup ratio, not both")


@dataclass(frozen=True, slots=True)
class Evaluation:
    every: int = 5
    before_training: bool = True
    samples_per_question: int = 16
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    prompt_tokens: int = 4096
    response_tokens: int = 31744
    question_limit: int | None = None
    seed: int = 42

    def __post_init__(self):
        for name in ("every", "samples_per_question", "prompt_tokens", "response_tokens"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"evaluation.{name} must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError(
                "Evaluation uses sampled avg@N with positive temperature and top_p in (0,1]"
            )
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("evaluation.top_k must be -1 or positive")
        if self.question_limit is not None and self.question_limit < 1:
            raise ValueError("evaluation.question_limit must be positive or null")
        if not 0 <= self.seed < 2**31:
            raise ValueError("evaluation.seed must be in [0, 2**31)")


@dataclass(frozen=True, slots=True)
class Resources:
    gpu_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    cpus: int = 24
    rollout_memory_fraction: float = 0.25
    max_sequences: int = 32
    max_batch_tokens: int = 9216
    object_store_bytes: int = 4 * 1024**3

    def __post_init__(self):
        if (
            not self.gpu_ids
            or len(set(self.gpu_ids)) != len(self.gpu_ids)
            or any(not isinstance(index, int) or index < 0 for index in self.gpu_ids)
        ):
            raise ValueError("resources.gpu_ids must be distinct nonnegative physical IDs")
        if not 0 < self.rollout_memory_fraction < 1:
            raise ValueError("Rollout memory fraction must lie in (0,1)")
        if any(
            getattr(self, name) < 1
            for name in ("cpus", "max_sequences", "max_batch_tokens", "object_store_bytes")
        ):
            raise ValueError("Resource budgets must be positive")


@dataclass(frozen=True, slots=True)
class Wandb:
    enabled: bool = False
    mode: str = "online"
    project: str = "mirroropd"
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in {"online", "offline"}:
            raise ValueError("W&B mode must be online or offline; enabled=false disables tracking")
        if not self.project.strip() or (self.group is not None and not self.group.strip()):
            raise ValueError("W&B project and an explicitly supplied group must be nonempty")
        if any(not tag.strip() or len(tag) > 64 or "," in tag for tag in self.tags) or len(
            set(self.tags)
        ) != len(self.tags):
            raise ValueError("W&B tags must be distinct strings of 1–64 characters without commas")

    @property
    def effective_mode(self):
        return self.mode if self.enabled else "disabled"


@dataclass(frozen=True, slots=True)
class Logging:
    console: bool = True
    wandb: Wandb = field(default_factory=Wandb)


@dataclass(frozen=True, slots=True)
class Checkpoints:
    every: int = 0
    max_to_keep: int = 2
    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])

    def __post_init__(self):
        if not isinstance(self.every, int) or self.every < 0:
            raise ValueError("checkpoints.every must be nonnegative; zero disables saving")
        if self.max_to_keep < 1:
            raise ValueError("checkpoints.max_to_keep must be positive")
        if (
            not self.save_contents
            or len(set(self.save_contents)) != len(self.save_contents)
            or set(self.save_contents) - {"model", "optimizer", "extra", "hf_model"}
            or not set(self.save_contents) & {"model", "hf_model"}
        ):
            raise ValueError(
                "Checkpoint contents must be distinct supported fields including model or hf_model"
            )


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    models: Models
    data: Dataset
    objective: Objective = field(default_factory=Objective)
    memory: Memory = field(default_factory=Memory)
    training: Training = field(default_factory=Training)
    evaluation: Evaluation = field(default_factory=Evaluation)
    precision: Precision = field(default_factory=Precision)
    resources: Resources = field(default_factory=Resources)
    logging: Logging = field(default_factory=Logging)
    checkpoints: Checkpoints = field(default_factory=Checkpoints)
    schema_version: int = API_VERSION

    def __post_init__(self):
        if self.schema_version != API_VERSION:
            raise ValueError(f"Job schema must be {API_VERSION}")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", self.name):
            raise ValueError("Job name must be a simple path-safe label")
        if self.objective.algorithm == "on_policy" and self.training.inner_steps != 1:
            raise ValueError("On-policy requires exactly one update per fresh full batch")
        sequences = self.training.batch_size * self.training.responses_per_prompt
        if sequences % (self.training.micro_batch_size * len(self.resources.gpu_ids)):
            raise ValueError("Global sequence batch must divide GPU count times micro batch")

    @property
    def config_id(self):
        return content_id(self.to_dict())

    @property
    def max_context_tokens(self):
        return max(
            self.training.prompt_tokens + self.training.response_tokens,
            self.evaluation.prompt_tokens + self.evaluation.response_tokens,
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return decode(cls, value)


def materialize_defaults(cls, value):
    data = copy.deepcopy(strict_fields(cls, value))
    for item in dataclasses.fields(cls):
        if dataclasses.is_dataclass(item.type):
            data[item.name] = materialize_defaults(item.type, data.get(item.name, {}))
        elif item.name not in data:
            if item.default is not dataclasses.MISSING:
                data[item.name] = copy.deepcopy(item.default)
            elif item.default_factory is not dataclasses.MISSING:
                data[item.name] = item.default_factory()
            else:
                data[item.name] = None
    return data


def _expand(value, variables):
    if isinstance(value, dict):
        return {key: _expand(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, variables) for item in value]
    if isinstance(value, str):

        def substitute(match):
            name = match.group(1)
            if name not in variables:
                raise ValueError(f"Missing explicit configuration variable: {name}")
            return str(variables[name])

        return re.sub(r"\$\{([a-zA-Z0-9_]+)\}", substitute, value)
    return value


def set_override(data, override):
    name, separator, raw = override.partition("=")
    if not separator:
        raise ValueError("Overrides use section.field=value")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    current = data
    parts = name.split(".")
    for key in parts[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise ValueError(f"Unknown configuration path: {name}")
        current = current[key]
    if parts[-1] not in current:
        raise ValueError(f"Unknown configuration field: {name}")
    current[parts[-1]] = value


def load_job(
    recipe,
    *,
    models=None,
    dataset=None,
    evaluation=None,
    variables=None,
    overrides=(),
    algorithm=None,
):
    data = json.loads(Path(recipe).read_text())
    if models is not None:
        data["models"] = json.loads(Path(models).read_text())
    if dataset is not None:
        data["data"] = json.loads(Path(dataset).read_text())
    if evaluation is not None:
        data["evaluation"] = json.loads(Path(evaluation).read_text())
    data = _expand(data, variables or {})
    # Materialize defaults once before applying strict, path-checked overrides.
    complete = materialize_defaults(Job, data)
    for override in overrides:
        set_override(complete, override)
    if algorithm is not None:
        complete["objective"]["algorithm"] = algorithm
        if algorithm == "on_policy":
            complete["training"]["inner_steps"] = 1
    return Job.from_dict(complete)


def with_algorithm(job, algorithm):
    data = copy.deepcopy(job.to_dict())
    data["objective"]["algorithm"] = algorithm
    if algorithm == "on_policy":
        data["training"]["inner_steps"] = 1
    return Job.from_dict(data)
