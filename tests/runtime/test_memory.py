"""Capacity admission and producer-local observations do not mutate execution."""

import json
from dataclasses import replace

import pytest

from mirroropd.core.config import Estimator, Objective
from mirroropd.runtime.events import EventWriter
from mirroropd.runtime.memory import Memory, cache_capacity, check_cache_capacity


def test_dense_cache_rejected_before_launch_and_sparse_budget_scales_with_head():
    dense = Objective(estimator=Estimator("full", "full", 16))
    estimate = cache_capacity(dense, prefixes=64 * 7168, vocab_size=151665)
    assert estimate["full_bytes"] == 64 * 7168 * 151665 * 4 * 2
    with pytest.raises(MemoryError, match=r"memory\.cache_bytes"):
        check_cache_capacity(dense, Memory(), prefixes=64 * 7168, vocab_size=151665)
    sparse = check_cache_capacity(Objective(), Memory(), prefixes=64 * 7168, vocab_size=151665)
    assert sparse["full_fields"] == 0
    assert sparse["total_bytes"] < estimate["total_bytes"] / 100
    assert (
        check_cache_capacity(
            dense,
            replace(Memory(), cache_bytes=estimate["total_bytes"]),
            prefixes=64 * 7168,
            vocab_size=151665,
        )
        == estimate
    )


@pytest.mark.parametrize(
    "values",
    [
        {"cache_bytes": 0},
        {"token_chunk_size": True},
        {"vocab_chunk_size": -1},
        {"projection": "fused"},
        {"profile_phases": 1},
    ],
)
def test_invalid_memory_controls_are_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        Memory(**values)


def test_nonprofiling_phase_does_not_reset_or_synchronize_cuda(tmp_path, monkeypatch):
    import torch

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal execution cannot synchronize/reset profiling state")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbidden)
    writer = EventWriter(tmp_path, "run", "worker-0")
    with writer.phase("cache.teacher"):
        pass
    rows = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["phase.started", "phase.completed"]
    assert rows[-1]["payload"]["memory_scope"] == "process_peak"
    assert "private_bytes" in rows[-1]["payload"]["memory"]


def test_minimal_mdpo_capacity_does_not_charge_unused_full_centers():
    spec = Objective("mdpo", estimator=Estimator("full", "sampled", 16), diagnostics="none")
    estimate = check_cache_capacity(spec, Memory(), prefixes=64 * 7168, vocab_size=151665)
    assert estimate["full_fields"] == 0
    assert estimate["total_bytes"] == 64 * 7168 * 17


def test_worker_profiling_is_explicit_and_bounds_its_own_peak(tmp_path, monkeypatch):
    import torch

    from mirroropd.runtime import memory

    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: calls.append("reset"))
    monkeypatch.setattr(memory, "memory_snapshot", lambda: {"allocated_bytes": 100})
    writer = EventWriter(tmp_path, "run", "worker-0")
    with writer.phase("actor", profile_memory=True):
        calls.append("forward_backward")
    assert calls == ["synchronize", "reset", "forward_backward", "synchronize"]
    rows = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert rows[-1]["payload"]["memory_scope"] == "phase_peak"


def test_teacher_offload_control_compiles_independently_of_actor():
    from mirroropd.integrations.verl.compiler import backend_values
    from mirroropd.runtime.config import Job

    job = Job.from_dict(
        {
            "name": "independent-offload",
            "models": {"student": "s", "teacher": "t"},
            "data": {"train": ["train"], "validation": ["val"]},
            "memory": {"teacher_parameter_offload": False},
        }
    )
    evidence = {
        "models": {"teacher": {"identity": "t"}},
        "vocabulary_id": "v",
        "vocab_size": 32,
        "dataset": {"splits": {"train": {"paths": ["train"]}, "validation": {"paths": ["val"]}}},
        "protocol": {},
        "evaluation_questions": 1,
    }
    result = backend_values(job, evidence, "run", "source", {})
    assert result["actor_rollout_ref"]["actor"]["fsdp_config"]["param_offload"] is True
    assert result["actor_rollout_ref"]["ref"]["fsdp_config"]["param_offload"] is False
    assert result["distillation"]["mirror"]["memory"]["projection"] == "chunked"
