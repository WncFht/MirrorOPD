"""Aggregate the fixed Pythia pilot; paired bootstrap units are held-out texts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_example(rows, field):
    groups = {}
    for row in rows:
        if field in row:
            groups.setdefault(row["example_index"], []).append(row[field])
    return np.array([np.mean(groups[key]) for key in sorted(groups)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    if (root / "exit-code.txt").read_text().strip() != "0":
        raise ValueError("Training did not finish successfully")
    fields = ("student_teacher_kl", "teacher_student_kl", "student_continuation_nll")
    baseline = read_jsonl(root / "evaluation/step-0/prefixes.jsonl")
    baseline_values = {field: per_example(baseline, field) for field in fields}
    rng = np.random.default_rng(20261008)
    bootstrap_indices = rng.integers(0, 64, (5000, 64))
    evaluations = []
    for step in (0, 8, 16, 24, 32):
        folder = root / f"evaluation/step-{step}"
        rows = read_jsonl(folder / "prefixes.jsonl")
        summary = json.loads((folder / "summary.json").read_text())
        for field in fields:
            values = per_example(rows, field)
            assert len(values) == 64
            # Pair identical prefix positions before grouping, not independent checkpoint draws.
            assert [(r["example_index"], r["prefix_length"]) for r in rows] == [
                (r["example_index"], r["prefix_length"]) for r in baseline
            ]
            deltas = values - baseline_values[field]
            replicates = deltas[bootstrap_indices].mean(axis=1)
            low, high = np.quantile(replicates, (0.025, 0.975))
            evaluations.append(
                {
                    "outer_step": step,
                    "metric": field,
                    "mean": summary["statistics"][field]["mean"],
                    "paired_delta": float(deltas.mean()),
                    "paired_delta_ci_low": float(low),
                    "paired_delta_ci_high": float(high),
                    "examples": 64,
                    "prefixes": summary["statistics"][field]["count"],
                }
            )
    write_csv(root / "checkpoint-metrics.csv", evaluations)
    inner = read_jsonl(root / "inner.jsonl")
    outer = read_jsonl(root / "outer.jsonl")
    assert len(outer) == 32
    assert len(inner) == 128
    assert {(r["outer_call"], r["inner_step"]) for r in inner} == {
        (k, n) for k in range(32) for n in range(4)
    }
    inner_rows = []
    for row in inner:
        metrics = row["metrics"]
        assert row["measurement"] == "pre-update"
        if row["inner_step"] == 0:
            assert metrics["mirror/sampled_log_ratio/rms"] == 0
            assert metrics["mirror/importance_ess_fraction"] == 1
            assert abs(metrics["mirror/student_center/mean"]) < 1e-6
        inner_rows.append(
            {
                "outer_step": row["outer_call"] + 1,
                "inner_step": row["inner_step"],
                "tokens": metrics["mirror/valid_tokens"],
                "log_ratio_rms": metrics["mirror/sampled_log_ratio/rms"],
                "anchor_student_tail_aware_kl": metrics["mirror/student_center/mean"],
                "importance_ess_fraction": metrics["mirror/importance_ess_fraction"],
                "grad_norm_before_clipping": metrics["grad_norm"],
                "score_coefficient_rms": metrics["mirror/score_coefficient/rms"],
                "sampled_teacher_log_ratio_mean": metrics["mirror/sampled_reward/mean"],
                "student_entropy": metrics["mirror/student_entropy/mean"],
            }
        )
    write_csv(root / "inner-metrics.csv", inner_rows)
    gpu = list(csv.DictReader((root / "gpu.csv").open()))
    gpu_peak = max(float(row[" memory.used [MiB]"].split()[0]) for row in gpu)
    total_tokens = sum(row["data"]["actor/mirror/valid_tokens"] for row in outer)
    token_weighted_clip = sum(row["data"]["response_length/clip_ratio"] * 16 for row in outer) / 512
    performance = {
        "outer_updates": len(outer),
        "optimizer_updates": len(inner),
        "prompts": 512,
        "generated_response_tokens": total_tokens,
        "length_cap_fraction_of_trajectories": token_weighted_clip,
        "summed_outer_step_seconds": sum(row["data"]["timing_s/step"] for row in outer),
        "gpu_peak_mib": gpu_peak,
        "all_hard_sync_resets_verified": True,
        "bootstrap": {"unit": "validation text", "draws": 5000, "seed": 20261008},
        "inner_step_token_weighted": {},
    }
    for n in range(4):
        rows = [row for row in inner_rows if row["inner_step"] == n]
        weights = np.array([row["tokens"] for row in rows])
        performance["inner_step_token_weighted"][str(n)] = {
            field: float(np.average([row[field] for row in rows], weights=weights))
            for field in (
                "log_ratio_rms",
                "anchor_student_tail_aware_kl",
                "importance_ess_fraction",
                "score_coefficient_rms",
            )
        }
    (root / "analysis.json").write_text(json.dumps(performance, indent=2) + "\n")
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    main()
