"""Summarize resource telemetry from a fixed math-run snapshot."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw", type=Path, required=True)
args = parser.parse_args()
root = args.raw
out = {}
for p in sorted(root.iterdir()):
    if not p.is_dir():
        continue
    outer = [json.loads(x) for x in (p / "outer.jsonl").read_text().splitlines()]
    inner = [json.loads(x) for x in (p / "inner.jsonl").read_text().splitlines()]
    train = [x["data"] for x in outer if x["step"] > 0]
    stats = {
        "complete_outer": len(train),
        "inner_updates": len(inner),
        "evaluation_outer": [
            x["step"] for x in outer if "evaluation/generated_examples" in x["data"]
        ],
    }
    for key in sorted(set().union(*(x.keys() for x in train))):
        if key.startswith(("timing_s/", "perf/", "actor/perf/")):
            vals = [x[key] for x in train if isinstance(x.get(key), (int, float))]
            if vals:
                stats[key] = {
                    "mean": float(np.mean(vals)),
                    "sum": float(np.sum(vals)),
                    "min": min(vals),
                    "max": max(vals),
                    "n": len(vals),
                }
    gpu = defaultdict(list)
    with (p / "gpu.csv").open() as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            gpu[int(row["index"])].append(
                {
                    "time": datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                    .replace(tzinfo=timezone.utc)
                    .timestamp(),
                    "memory": float(row["memory.used [MiB]"].split()[0]),
                    "utilization": float(row["utilization.gpu [%]"].split()[0]),
                    "power": float(row["power.draw [W]"].split()[0]),
                }
            )
    stats["gpu"] = {
        i: {
            "samples": len(rows),
            "peak_mib": max(x["memory"] for x in rows),
            "mean_utilization": np.mean([x["utilization"] for x in rows]),
            "mean_power_w": np.mean([x["power"] for x in rows]),
            "elapsed_s": rows[-1]["time"] - rows[0]["time"],
        }
        for i, rows in gpu.items()
    }
    host = [json.loads(x) for x in (p / "host-memory.jsonl").read_text().splitlines()]
    bystage = defaultdict(list)
    for x in host:
        bystage[x["stage"]].append(x)
    stats["host_stages"] = {
        s: {
            "samples": len(xs),
            "pids": len({x["pid"] for x in xs}),
            "max_rss_gib": max(x["before"]["rss_bytes"] for x in xs) / 2**30,
            "max_pss_gib": max(x["before"]["pss_bytes"] for x in xs) / 2**30,
            "max_pinned_gib": max(
                x["before"]["pinned"].get("allocated_bytes.current", 0) for x in xs
            )
            / 2**30,
            "max_gpu_allocated_gib": max(x["before"]["gpu_allocated"] for x in xs) / 2**30,
            "max_gpu_reserved_gib": max(x["before"]["gpu_reserved"] for x in xs) / 2**30,
        }
        for s, xs in bystage.items()
    }
    out[p.name] = stats
(root / "resource-statistics.json").write_text(json.dumps(out, indent=2))
for name, s in out.items():
    print(name, "outer", s["complete_outer"], "inner", s["inner_updates"], "gpu", s["gpu"])
    for k in [
        "timing_s/gen",
        "timing_s/old_log_prob",
        "timing_s/mirror_teacher_cache",
        "timing_s/update_actor",
        "timing_s/update_weights",
        "timing_s/step",
        "timing_s/testing",
        "perf/throughput",
        "actor/perf/cpu_memory_used_gb",
        "actor/perf/max_memory_reserved_gb",
    ]:
        print(k, s.get(k))
    print("host", s["host_stages"])
