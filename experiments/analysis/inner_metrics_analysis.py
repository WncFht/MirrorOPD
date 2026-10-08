"""Summarize recorded pre-update diagnostics without treating them as targets."""

import argparse
import csv
import json
from pathlib import Path


def summarize(directory):
    rows = []
    summaries = {}
    for mode in ("bpo", "mean", "mdpo"):
        source = directory / f"metrics-{mode}-inner.jsonl"
        records = [json.loads(line) for line in source.read_text().splitlines()]
        groups = {}
        for record in records:
            groups.setdefault(record["outer_call"], []).append(record)
            metrics = record["metrics"]
            rows.append(
                {
                    "mode": mode,
                    "outer_call": record["outer_call"],
                    "inner_step": record["inner_step"],
                    "valid_tokens": metrics["mirror/valid_tokens"],
                    "student_center_mean": metrics["mirror/student_center/mean"],
                    "teacher_center_mean": metrics["mirror/teacher_center/mean"],
                    "coefficient_rms": metrics["mirror/score_coefficient/rms"],
                    "importance_ess_fraction": metrics["mirror/importance_ess_fraction"],
                    "student_entropy_mean": metrics["mirror/student_entropy/mean"],
                    "grad_norm": metrics["grad_norm"],
                }
            )
        summaries[mode] = []
        for outer, group in groups.items():
            assert [r["inner_step"] for r in group] == list(range(len(group)))
            first, last = group[0]["metrics"], group[-1]["metrics"]
            teacher_means = [r["metrics"]["mirror/teacher_center/mean"] for r in group]
            assert max(teacher_means) == min(teacher_means), "outer cache changed"
            assert abs(first["mirror/student_center/mean"]) < 1e-6, "hard sync failed"
            summaries[mode].append(
                {
                    "outer_call": outer,
                    "updates": len(group),
                    "teacher_center_mean": teacher_means[0],
                    "initial_student_center_mean": first["mirror/student_center/mean"],
                    "final_pre_update_student_center_mean": last["mirror/student_center/mean"],
                    "coefficient_rms_first": first["mirror/score_coefficient/rms"],
                    "coefficient_rms_last": last["mirror/score_coefficient/rms"],
                    "grad_norm_first": first["grad_norm"],
                    "grad_norm_last": last["grad_norm"],
                    "min_importance_ess_fraction": min(
                        r["metrics"]["mirror/importance_ess_fraction"] for r in group
                    ),
                }
            )
    with (directory / "inner-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
