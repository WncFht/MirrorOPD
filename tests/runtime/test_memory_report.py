"""Keep inherited CUDA counters and incomplete spans out of phase-peak claims."""

import runpy
from pathlib import Path

import pytest

REPORT = runpy.run_path(str(Path(__file__).resolve().parents[2] / "benchmarks/memory_report.py"))


def event(kind, sequence, *, scope="phase_peak"):
    return {
        "event": kind,
        "producer_id": "worker-0-test",
        "source": "worker-0",
        "sequence": sequence,
        "time_utc": f"2026-10-17T00:00:0{sequence}+00:00",
        "payload": {
            "phase": "worker.train",
            "duration_seconds": 1,
            "memory_scope": scope,
            "memory": {
                "rss_bytes": 20,
                "pss_bytes": 10,
                "process_peak_allocated_bytes": 100,
                "process_peak_reserved_bytes": 200,
            },
        },
    }


def test_inherited_counter_is_not_reported_as_a_local_peak():
    spans = REPORT["phase_spans"](
        [event("phase.started", 1), event("phase.completed", 2, scope="process_peak")]
    )
    result = REPORT["summarize_phases"](spans, [])
    assert result[0]["cuda_phase_peak_allocated_bytes"] is None
    assert result[0]["sampled_device_peak_bytes"] is None


def test_incomplete_and_failed_spans_are_rejected():
    for records in (
        [event("phase.started", 1)],
        [event("phase.started", 1), event("phase.failed", 2)],
    ):
        with pytest.raises(ValueError, match=r"memory (phase|spans)"):
            REPORT["phase_spans"](records)


def test_device_counter_matches_the_worker_and_interval():
    spans = REPORT["phase_spans"]([event("phase.started", 1), event("phase.completed", 2)])
    start = REPORT["timestamp"](spans[0][0])
    samples = [
        {
            "time_unix": start + delta,
            "devices": {"0": {"used_bytes": value}, "1": {"used_bytes": 999}},
        }
        for delta, value in [(-1, 888), (0.5, 300), (2, 777)]
    ]
    result = REPORT["summarize_phases"](spans, samples)[0]
    assert result["cuda_phase_peak_allocated_bytes"] == 100
    assert result["sampled_device_peak_bytes"] == 300
