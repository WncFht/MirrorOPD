"""Explicit cache budgets and phase-local memory observations.

Budgets cover frozen evidence, not model weights or allocator workspace. Device
peaks are meaningful only in the worker that owns the CUDA context. Ordinary
observations never initialize CUDA, reset its counters, or synchronize it.
"""

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Memory:
    cache_bytes: int = 2 * 1024**3
    scoring_buffer_bytes: int = 64 * 1024**2
    token_chunk_size: int = 64
    vocab_chunk_size: int = 4096
    projection: str = "chunked"
    profile_phases: bool = False
    teacher_parameter_offload: bool | None = None

    def __post_init__(self):
        for name in ("cache_bytes", "scoring_buffer_bytes", "token_chunk_size", "vocab_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"memory.{name} must be a positive integer")
        if self.projection not in {"eager", "chunked"}:
            raise ValueError("memory.projection must be eager or chunked")
        if (
            self.teacher_parameter_offload is not None
            and type(self.teacher_parameter_offload) is not bool
        ):
            raise TypeError("memory.teacher_parameter_offload must be boolean or null")
        if type(self.profile_phases) is not bool:
            raise TypeError("memory.profile_phases must be boolean")


def cache_capacity(
    objective, *, prefixes, vocab_size, bytes_per_probability=4, sequence_slots=None
):
    """Upper bound for compact evidence; includes optional diagnostics fields."""
    if prefixes < 0 or vocab_size < 1:
        raise ValueError("Cache dimensions must be nonnegative prefixes and positive vocabulary")
    slots = prefixes if sequence_slots is None else sequence_slots
    if slots < prefixes:
        raise ValueError("Sequence slots cannot be fewer than valid predictors")
    j = objective.estimator.topj
    full = int(objective.estimator.center == "full") + int(
        objective.estimator.integration == "full"
    )
    minimal = (
        objective.algorithm == "mdpo"
        and objective.estimator.integration == "sampled"
        and objective.diagnostics == "none"
    )
    if minimal:
        full, j = 0, 0
    # bool valid, int64 action, head IDs/LP, four scalar LP/center fields,
    # optional teacher support. Conservative even for the MDPO minimal path.
    sparse = 1 + 8 + j * (8 + bytes_per_probability) + 4 * bytes_per_probability
    if minimal:
        sparse = 1 + 8 + 2 * bytes_per_probability
    if objective.diagnostics == "full":
        sparse += j * 8
    return {
        "prefixes": prefixes,
        "vocab_size": vocab_size,
        "full_fields": full,
        "sequence_slots": slots,
        "sparse_bytes": slots * sparse,
        "full_bytes": prefixes * vocab_size * bytes_per_probability * full,
        "total_bytes": slots * sparse + prefixes * vocab_size * bytes_per_probability * full,
        "layout": "response_predictors",
    }


def check_cache_capacity(objective, memory, *, prefixes, vocab_size, sequence_slots=None):
    estimate = cache_capacity(
        objective, prefixes=prefixes, vocab_size=vocab_size, sequence_slots=sequence_slots
    )
    if estimate["total_bytes"] > memory.cache_bytes:
        raise MemoryError(
            f"Frozen cache requires at most {estimate['total_bytes']} bytes for {prefixes} "
            f"response predictors, exceeding memory.cache_bytes={memory.cache_bytes}. "
            "Reduce the batch/response budget or explicitly raise the host cache budget. "
            "Changing center/integration changes the estimator and must be a separate experiment."
        )
    return estimate


def scoring_tile_size(objective, memory, *, vocab_size):
    """Bound one scoring tile including Top-J IDs, probabilities and diagnostics."""
    single = cache_capacity(objective, prefixes=1, vocab_size=vocab_size)
    per_prefix = single["full_bytes"] + max(512, single["sparse_bytes"])
    tile = min(memory.token_chunk_size, memory.scoring_buffer_bytes // per_prefix)
    if tile < 1:
        raise MemoryError("memory.scoring_buffer_bytes cannot hold one scoring predictor")
    return tile


def tensor_bytes(values):
    """Unique retained storage bytes, including jagged values and offsets."""
    seen = set()
    total = 0
    for value in values:
        parts = (value.values(), value.offsets()) if value.is_nested else (value,)
        for part in parts:
            identity = (str(part.device), part.untyped_storage().data_ptr())
            if identity not in seen:
                seen.add(identity)
                total += part.untyped_storage().nbytes()
    return total


def memory_snapshot():
    result = {}
    path = Path("/proc/self/smaps_rollup")
    if path.exists():
        fields = {}
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Rss", "Pss", "Private_Clean", "Private_Dirty", "Locked"}:
                fields[key] = int(value.split()[0]) * 1024
        result.update(
            rss_bytes=fields.get("Rss"),
            pss_bytes=fields.get("Pss"),
            private_bytes=fields.get("Private_Clean", 0) + fields.get("Private_Dirty", 0),
            locked_bytes=fields.get("Locked"),
        )
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        result.update(
            cuda_device=torch.cuda.current_device(),
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            process_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            process_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    return result
