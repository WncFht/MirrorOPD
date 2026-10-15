"""Versioned, orthogonal algorithm, signal, estimator and diagnostic settings."""

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from types import UnionType
from typing import Union, get_args, get_origin

API_VERSION = 2
ALGORITHMS = frozenset({"on_policy", "mdpo", "bpo"})


def strict_fields(cls, value):
    if not isinstance(value, dict):
        raise TypeError(f"{cls.__name__} requires an object")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return value


def _matches_type(value, expected):
    origin = get_origin(expected)
    if origin in {UnionType, Union}:
        return any(_matches_type(value, item) for item in get_args(expected))
    if origin is list:
        return isinstance(value, list) and all(
            _matches_type(item, get_args(expected)[0]) for item in value
        )
    if origin is dict:
        key_type, value_type = get_args(expected)
        return isinstance(value, dict) and all(
            _matches_type(key, key_type) and _matches_type(item, value_type)
            for key, item in value.items()
        )
    if expected is float:
        return type(value) in {int, float}
    return type(value) is expected


def decode(cls, value):
    data = dict(strict_fields(cls, value))
    for item in fields(cls):
        if item.name not in data:
            continue
        value = data[item.name]
        if is_dataclass(item.type):
            data[item.name] = decode(item.type, value)
        elif not _matches_type(value, item.type):
            raise TypeError(
                f"{cls.__name__}.{item.name} expects {item.type}, got {type(value).__name__}"
            )
        elif item.type is float:
            data[item.name] = float(value)
    return cls(**data)


@dataclass(frozen=True, slots=True)
class Signal:
    kappa: float = 0.5
    eta: float = 0.2

    def __post_init__(self):
        if not math.isfinite(self.kappa) or not 0 <= self.kappa <= 1:
            raise ValueError("kappa must be finite and in [0, 1]")
        if not math.isfinite(self.eta) or not 0 < self.eta <= 1:
            raise ValueError("eta must be finite and in (0, 1]")


@dataclass(frozen=True, slots=True)
class Estimator:
    center: str = "topj_tail"
    integration: str = "sampled"
    topj: int = 16
    chunk_size: int = 64

    def __post_init__(self):
        if self.center not in {"topj_tail", "full"}:
            raise ValueError("center must be topj_tail or full")
        if self.integration not in {"sampled", "full"}:
            raise ValueError("integration must be sampled or full")
        if self.integration == "full" and self.center != "full":
            raise ValueError("Full integration requires full centers")
        if isinstance(self.topj, bool) or not isinstance(self.topj, int) or self.topj < 0:
            raise ValueError("topj must be a nonnegative integer")
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class Objective:
    algorithm: str = "bpo"
    signal: Signal = field(default_factory=Signal)
    estimator: Estimator = field(default_factory=Estimator)
    diagnostics: str = "full"

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(ALGORITHMS)}")
        if not isinstance(self.signal, Signal) or not isinstance(self.estimator, Estimator):
            raise TypeError("Objective requires typed Signal and Estimator settings")
        if self.diagnostics not in {"none", "basic", "full"}:
            raise ValueError("diagnostics must be none, basic or full")

    @classmethod
    def from_dict(cls, value):
        return decode(cls, value)

    def to_dict(self):
        return asdict(self)

    @property
    def loss_kind(self):
        return (
            "conditional_objective"
            if self.estimator.integration == "full" and self.algorithm != "on_policy"
            else "score_surrogate"
        )


@dataclass(frozen=True, slots=True)
class CacheContext:
    anchor_version: int
    teacher_id: str
    tokenizer_id: str
    vocab_size: int
    temperature: float
    signal: Signal
    estimator: Estimator
    schema_version: int = API_VERSION
    action_only: bool = False

    def __post_init__(self):
        if self.schema_version != API_VERSION:
            raise ValueError(f"Cache schema must be {API_VERSION}")
        if type(self.action_only) is not bool:
            raise TypeError("Cache action_only must be boolean")
        if not isinstance(self.anchor_version, int) or self.anchor_version < 0:
            raise ValueError("anchor_version must be a nonnegative integer")
        if not self.teacher_id or not self.tokenizer_id:
            raise ValueError("Cache requires teacher and tokenizer identities")
        if self.vocab_size < 1 or self.estimator.topj > self.vocab_size:
            raise ValueError("topj must fit the declared vocabulary")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Cache temperature must be finite and positive")

    def validate(self, objective, *, anchor_version=None):
        if self.signal != objective.signal or self.estimator != objective.estimator:
            raise ValueError("Objective does not match the frozen cache signal/estimator")
        if anchor_version is not None and anchor_version != self.anchor_version:
            raise ValueError("Stale anchor version")
        if self.action_only and not (
            objective.algorithm == "mdpo"
            and objective.estimator.integration == "sampled"
            and objective.diagnostics == "none"
        ):
            raise ValueError("Action-only cache cannot supply this objective's required statistics")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return decode(cls, value)
