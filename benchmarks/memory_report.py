"""Recompute memory tables from complete worker spans and sampled NVML records."""

import argparse
import json
import math
import struct
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def phase_spans(records):
    pending = defaultdict(list)
    spans = []
    for row in sorted(records, key=lambda value: (value["producer_id"], value["sequence"])):
        if row["event"] not in {"phase.started", "phase.completed", "phase.failed"}:
            continue
        key = (row["producer_id"], row["payload"]["phase"])
        if row["event"] == "phase.started":
            pending[key].append(row)
        else:
            if row["event"] == "phase.failed" or not pending[key]:
                raise ValueError("Failed or unmatched memory phase")
            spans.append((pending[key].pop(), row))
    if any(pending.values()):
        raise ValueError("Incomplete memory spans cannot define phase peaks")
    return spans


def timestamp(record):
    return datetime.fromisoformat(record["time_utc"]).timestamp()


def summarize_phases(spans, samples):
    groups = defaultdict(list)
    for start, end in spans:
        groups[(end["source"], end["payload"]["phase"])].append((start, end))
    output = []
    for (source, phase), values in sorted(groups.items()):
        memory = [end["payload"]["memory"] for _, end in values]
        cuda = [
            end["payload"]["memory"]
            for _, end in values
            if end["payload"].get("memory_scope") == "phase_peak"
            and "process_peak_allocated_bytes" in end["payload"]["memory"]
        ]
        board_ids = [source.removeprefix("worker-")] if source.startswith("worker-") else None
        observed = [
            device["used_bytes"]
            for start, end in values
            for sample in samples
            if timestamp(start) <= sample["time_unix"] <= timestamp(end)
            for gpu, device in sample["devices"].items()
            if board_ids is None or gpu in board_ids
        ]
        output.append(
            {
                "source": source,
                "phase": phase,
                "count": len(values),
                "duration_seconds": [end["payload"]["duration_seconds"] for _, end in values],
                "cuda_phase_peak_allocated_bytes": max(
                    (value["process_peak_allocated_bytes"] for value in cuda), default=None
                ),
                "cuda_phase_peak_reserved_bytes": max(
                    (value["process_peak_reserved_bytes"] for value in cuda), default=None
                ),
                "cuda_end_allocated_bytes": [value.get("allocated_bytes") for value in memory],
                "max_boundary_rss_bytes": max(value["rss_bytes"] for value in memory),
                "max_boundary_pss_bytes": max(value["pss_bytes"] for value in memory),
                "sampled_device_peak_bytes": max(observed, default=None),
                "device_samples": len(observed),
            }
        )
    return output


def model_stats(directory):
    count = 0
    sizes = defaultdict(int)
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        for name, value in header.items():
            if name != "__metadata__":
                count += math.prod(value["shape"])
                sizes[value["dtype"]] += value["data_offsets"][1] - value["data_offsets"][0]
    if not count:
        raise ValueError("Model storage accounting requires safetensors headers")
    config = json.loads((directory / "config.json").read_text())
    hidden = config["hidden_size"]
    return {
        "state_elements": count,
        "stored_bytes_by_dtype": dict(sizes),
        "hidden_size": hidden,
        "padded_vocab_size": config["vocab_size"],
        "layers": config["num_hidden_layers"],
        "kv_heads": config["num_key_value_heads"],
        "head_dimension": hidden // config["num_attention_heads"],
    }


def main():
    from mirroropd.runtime.artifacts import file_digest
    from mirroropd.runtime.audit import audit_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.input / "native/status.json").read_text())
    if status.get("state") != "completed" or len(status.get("completed", [])) != 2:
        raise ValueError("The paired native measurement is incomplete")
    cases = []
    for mode in ("eager", "chunked"):
        directory = args.input / "native" / mode
        verification = audit_run(directory)
        records = [
            json.loads(line)
            for path in (directory / "events").glob("*.jsonl")
            for line in path.open()
        ]
        nvml = args.input / "native" / f"{mode}-nvml.jsonl"
        samples = [json.loads(line) for line in nvml.open()]
        if len(samples) < 2:
            raise ValueError("Missing whole-device observations")
        spans = phase_spans(records)
        driver_evaluations = sorted(
            timestamp(start)
            for start, end in spans
            if end["source"] == "driver" and end["payload"]["phase"] == "evaluation"
        )
        optimization_end = max(
            timestamp(end)
            for _, end in spans
            if end["source"] == "driver" and end["payload"]["phase"] == "optimization"
        )
        intervals = []
        for name, begin, stop in (
            ("startup_to_initial_evaluation", samples[0]["time_unix"], driver_evaluations[0]),
            ("after_optimization_to_final_evaluation", optimization_end, driver_evaluations[-1]),
        ):
            observed = [
                value["used_bytes"]
                for row in samples
                if begin <= row["time_unix"] <= stop
                for value in row["devices"].values()
            ]
            intervals.append(
                {
                    "interval": name,
                    "duration_seconds": stop - begin,
                    "sampled_device_peak_bytes": max(observed, default=None),
                }
            )
        job = json.loads((directory / "job.json").read_text())
        anchor = [row["payload"] for row in records if row["event"] == "anchor.created"]
        cases.append(
            {
                "projection": mode,
                "verification": verification,
                "job": job,
                "inputs": json.loads((directory / "inputs.json").read_text()),
                "phases": summarize_phases(spans, samples),
                "unspanned_intervals": intervals,
                "anchor_cache": anchor,
                "device_peak_bytes": max(
                    value["used_bytes"] for row in samples for value in row["devices"].values()
                ),
                "nvml_records": len(samples),
                "nvml_sha256": file_digest(nvml),
            }
        )
    if cases[0]["verification"]["source_id"] != cases[1]["verification"]["source_id"]:
        raise ValueError("Paired executions have different production source identities")
    conditions = []
    for case in cases:
        condition = json.loads(json.dumps(case["job"]))
        condition.pop("name")
        condition["memory"].pop("projection")
        conditions.append(condition)
    if conditions[0] != conditions[1]:
        raise ValueError("Paired scientific conditions differ beyond projection/name")
    for role in ("student", "teacher"):
        if (
            cases[0]["inputs"]["models"][role]["identity"]
            != cases[1]["inputs"]["models"][role]["identity"]
        ):
            raise ValueError("Paired model artifacts differ")
    models = {
        role: model_stats(Path(cases[0]["job"]["models"][role])) for role in ("student", "teacher")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in ("eager", "chunked"):
        path = args.input / f"head-{mode}" / "summary.json"
        (args.output / f"head-{mode}.json").write_bytes(path.read_bytes())
    result = {
        "scope": "One paired short real-model run per mode; stage peaks are maxima, not means",
        "models": models,
        "cases": cases,
        "limitations": [
            "NVML is sampled every 250 ms and is a lower bound on instantaneous board peak.",
            "RSS and PSS are boundary readings, not host-memory phase peaks.",
            "Nested projection spans inherit a counter; only outer worker spans define CUDA phase peaks.",
            "Whole-device usage includes other CUDA contexts and allocations outside PyTorch.",
            "The synthetic prompts and short responses do not evaluate task quality.",
            "BF16 cross-kernel runs are not promised to be bitwise-identical training trajectories.",
        ],
    }
    (args.output / "native-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "audited_cases": len(cases)}))


if __name__ == "__main__":
    main()
