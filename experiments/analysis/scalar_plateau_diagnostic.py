"""Retrospective, finite-budget plateau estimates for completed scalar runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(root):
    results = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        config = summary["config"]
        if config["method"] != "constant" or summary["status"] != "complete":
            continue
        directory = summary_path.parent
        audits = [
            json.loads(line) for line in (directory / "audits.jsonl").read_text().splitlines()
        ]
        updates = [
            json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()
        ]
        kl = np.array([row["metrics"]["train"]["teacher_reverse_kl"] for row in audits])
        reference = float(np.median(kl[-6:]))
        cumulative = np.r_[0, np.cumsum([row["applied_lr"] / 1e-6 for row in updates])]
        thresholds = {
            "within_5pct_observed_initial_to_tail_gap": 0.05 * abs(float(kl[0]) - reference),
            "within_1pct_tail_kl": 0.01 * reference,
        }
        estimates = {}
        for label, tolerance in thresholds.items():
            index = next(
                (i for i in range(len(kl) - 2) if np.all(np.abs(kl[i:] - reference) <= tolerance)),
                None,
            )
            step = None if index is None else audits[index]["update"]
            estimates[label] = {
                "tolerance": tolerance,
                "step": step,
                "u": None if step is None else float(cumulative[step]),
                "persistent_audit_points": None if index is None else len(kl) - index,
            }
        results.append(
            {
                "id": directory.name,
                "seed": config["seed"],
                "lr": config["lr"],
                "updates": len(updates),
                "tail_reference": reference,
                "initial_kl": float(kl[0]),
                "estimates": estimates,
                "grad_norm_median_per_100_updates": [
                    float(np.median([r["gradient_norm_before_clip"] for r in updates[i : i + 100]]))
                    for i in range(0, len(updates), 100)
                ],
            }
        )
    return {
        "definition": "Retrospective proximity to median of last six train-KL audits; at least three audit points and no later departure within observed budget. Not an asymptotic limit or a noise-floor estimate.",
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed_constant_runs": len(result["runs"])}))


if __name__ == "__main__":
    main()
