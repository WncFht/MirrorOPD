"""Summarize measured MDPO logs without confusing canaries with benchmarks."""

import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report(run, output):
    inner = read_rows(run / "inner.jsonl")
    outer = read_rows(run / "outer.jsonl")
    evaluation = [row for row in outer if "evaluation/generated_examples" in row["data"]]
    first = [row for row in inner if row["inner_step"] == 0]
    norms = [row["metrics"]["grad_norm"] for row in inner]
    training = [row["data"] for row in outer if "timing_s/step" in row["data"]]
    with (run / "gpu.csv").open() as stream:
        memory = [float(row[" memory.used [MiB]"].split()[0]) for row in csv.DictReader(stream)]
    summary = {
        "run_name": run.name,
        "exit_code": int((run / "exit-code.txt").read_text())
        if (run / "exit-code.txt").exists()
        else None,
        "completed_outer_calls": len(first),
        "inner_updates": len(inner),
        "actual_clip_thresholds": sorted({row["clip_grad"] for row in inner}),
        "grad_norm_min": min(norms),
        "grad_norm_max": max(norms),
        "clipped_fraction": sum(row["metrics"]["grad_was_clipped"] for row in inner) / len(inner),
        "sync_student_center_max_abs": max(
            abs(row["metrics"]["mirror/student_center/mean"]) for row in first
        ),
        "sync_sampled_log_ratio_rms_max": max(
            row["metrics"]["mirror/sampled_log_ratio/rms"] for row in first
        ),
        "gpu_peak_mib": max(memory),
        "importance_ess_fraction_min": min(
            row["metrics"]["mirror/importance_ess_fraction"] for row in inner
        ),
        "outer_seconds_mean": sum(row["timing_s/step"] for row in training) / len(training),
        "actor_peak_reserved_gib": max(
            row["actor/perf/max_memory_reserved_gb"] for row in training
        ),
        "evaluation": [
            {
                "outer_step": row["step"],
                **{
                    key: value
                    for key, value in row["data"].items()
                    if key.startswith(("val-core/", "val-aux/", "evaluation/"))
                },
            }
            for row in evaluation
        ],
        "interpretation": "Execution evidence; small test subsets are not GSM8K benchmarks.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "outer_call",
        "inner_step",
        "grad_norm",
        "grad_clip_scale",
        "mirror/student_center/mean",
        "mirror/teacher_center/mean",
        "mirror/sampled_log_ratio/rms",
        "mirror/importance_ess_fraction",
    ]
    with (output / "inner-dynamics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in inner:
            writer.writerow(
                {
                    "outer_call": row["outer_call"],
                    "inner_step": row["inner_step"],
                    **{key: row["metrics"][key] for key in fields[2:]},
                }
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
