"""Versioned, orthogonal algorithm, signal, estimator and diagnostic settings."""

import math
from dataclasses import asdict, dataclass, field, fields

API_VERSION = 2
ALGORITHMS = frozenset({"on_policy", "mdpo", "bpo"})


def strict_fields(cls, value):
    if not isinstance(value, dict):
        raise TypeError(f"{cls.__name__} requires an object")
    unknown = set(value) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return value


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
        data = dict(strict_fields(cls, value))
        if "signal" in data:
            data["signal"] = Signal(**strict_fields(Signal, data["signal"]))
        if "estimator" in data:
            data["estimator"] = Estimator(**strict_fields(Estimator, data["estimator"]))
        return cls(**data)

    def to_dict(self):
        return asdict(self)


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

    def __post_init__(self):
        if self.schema_version != API_VERSION:
            raise ValueError(f"Cache schema must be {API_VERSION}")
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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(strict_fields(cls, value))
        data["signal"] = Signal(**strict_fields(Signal, data["signal"]))
        data["estimator"] = Estimator(**strict_fields(Estimator, data["estimator"]))
        return cls(**data)
