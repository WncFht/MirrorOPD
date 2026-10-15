"""Global weighted statistics, event provenance, interrupted streams and failures."""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from mirroropd.identity import content_id
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.events import EventReader, EventWriter
from mirroropd.runtime.metrics import finalize_moments, record_update, token_moments


def reduce_ranks(ranks):
    result = {}
    for key, (reduction, _) in ranks[0].items():
        values = [rank[key][1] for rank in ranks]
        if reduction == "sum":
            result[key] = sum(values) / len(ranks)
        elif reduction == "min":
            result[key] = min(values)
        else:
            result[key] = max(values)
    return result


def test_four_unequal_ranks_and_micro_partitions_produce_global_token_statistics(tmp_path):
    ranks = []
    offset = 1
    for count in (1, 2, 3, 4):
        values = torch.arange(offset, offset + count, dtype=torch.float64)
        ranks.append(
            token_moments({"importance_ratio": values}, torch.ones(count, dtype=torch.bool))
        )
        offset += count
    globally_reduced = reduce_ranks(ranks) | {"grad_norm": 4.0, "loss": 17.0}
    measured = finalize_moments(globally_reduced, dp_size=4)
    assert measured["loss"] == 17.0
    assert measured["distill/valid_tokens"] == 10
    assert measured["distill/importance_ratio/mean"] == 5.5
    assert measured["distill/importance_ratio/std"] == math.sqrt(8.25)
    assert measured["distill/importance_ratio/min"] == 1
    assert measured["distill/importance_ratio/max"] == 10
    assert measured["distill/importance_ess"] == 55**2 / 385
    assert measured["distill/importance_ess_fraction"] == 55**2 / (10 * 385)
    writer = EventWriter(tmp_path, "run", "worker-0")
    for outer in range(2):
        for inner in range(16):
            for rank in (3, 1, 2, 0):
                record_update(
                    measured,
                    outer=outer,
                    inner=inner,
                    epochs=16,
                    algorithm="mdpo",
                    loss_kind="score_surrogate",
                    clip_grad=2,
                    global_rank=rank,
                    writer=writer,
                )
    records = EventReader(tmp_path).read()
    assert len(records) == 32
    assert [row["payload"]["update"] for row in records] == list(range(32))
    assert all(row["payload"]["metrics"]["optimization/clip_scale"] == 0.5 for row in records)


def test_padding_nonfinite_and_empty_diagnostics():
    mask = torch.tensor([[True, False, True]])
    values = torch.tensor([[2.0, float("nan"), 4.0]])
    result = finalize_moments(reduce_ranks([token_moments({"reward": values}, mask)]), 1)
    assert result["distill/reward/mean"] == 3
    assert result["distill/reward/std"] == 1
    assert token_moments({}, mask) == {"distill/raw/count": ("sum", 2.0)}
    empty = token_moments({"reward": values}, torch.zeros_like(mask))
    assert empty["distill/raw/count"] == ("sum", 0.0)
    reduced = finalize_moments(reduce_ranks([empty, token_moments({"reward": values}, mask)]), 2)
    assert reduced["distill/valid_tokens"] == 2
    assert reduced["distill/reward/min"] == 2
    assert reduced["distill/reward/max"] == 4
    with pytest.raises(FloatingPointError):
        token_moments({"reward": values}, torch.ones_like(mask))
    with pytest.raises(ValueError, match="Unregistered"):
        token_moments({"teacher_kl": values}, mask)


def test_producer_restarts_and_threads_never_duplicate_event_ids(tmp_path):
    writer = EventWriter(tmp_path, "run", "driver")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda index: writer.emit("item", index=index), range(40)))
    second = EventWriter(tmp_path, "run", "driver")
    second.emit("run.completed")
    records = EventReader(tmp_path).read()
    assert len(records) == len({row["event_id"] for row in records}) == 41
    assert sorted(
        row["sequence"] for row in records if row["producer_id"] == writer.producer_id
    ) == list(range(1, 41))


def test_incremental_reader_waits_for_complete_rows_and_restores_cursor(tmp_path):
    writer = EventWriter(tmp_path / "events", "run", "worker-0")
    row = writer.emit("first")
    cursor = tmp_path / "cursor.json"
    reader = EventReader(writer.directory, cursor_path=cursor)
    assert reader.read() == [row]
    assert reader.read() == []
    second = writer.emit("second")
    encoded = writer.path.read_bytes()
    writer.path.write_bytes(encoded[:-5])
    assert reader.read() == []
    writer.path.write_bytes(encoded)
    assert EventReader(writer.directory, cursor_path=cursor).read() == [second]
    writer.path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        reader.read()


def test_phase_failure_is_durable_and_secrets_are_not_structured_metrics(tmp_path):
    writer = EventWriter(
        tmp_path, "run", "controller", context={"api_key": "secret-value", "tokens": 17}
    )
    with pytest.raises(RuntimeError), writer.phase("update", outer=1):
        raise RuntimeError("synthetic failure")
    records = EventReader(tmp_path).read()
    assert [row["event"] for row in records] == ["phase.started", "phase.failed"]
    assert records[-1]["payload"]["error_type"] == "RuntimeError"
    assert records[-1]["payload"]["duration_seconds"] >= 0
    assert records[0]["context"] == {"api_key": "[redacted]", "tokens": 17}
    with pytest.raises(ValueError, match="Out of range"):
        writer.emit("invalid", score=float("nan"))
    row = writer.emit("after.failure")
    assert row["sequence"] == 3
    assert "secret-value" not in writer.path.read_text()


def test_atomic_json_retains_previous_artifact_on_encoding_error(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"state": "complete"})
    with pytest.raises(ValueError, match="Out of range"):
        atomic_json(path, {"state": float("nan")})
    assert json.loads(path.read_text()) == {"state": "complete"}
    assert content_id({"a": 1, "b": 2}) == content_id({"b": 2, "a": 1})
