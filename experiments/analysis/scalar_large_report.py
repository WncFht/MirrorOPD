"""Predeclared calibration rankings and paired confirmation statistics."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_run(path):
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    points = [json.loads(line) for line in (path / "audits.jsonl").read_text().splitlines()]
    config = summary["config"]
    initial = points[0]["metrics"]["train"]["bpo"]
    values = np.array([point["metrics"]["train"]["bpo"] / initial for point in points])
    updates = np.array([point["update"] for point in points])
    seconds = np.array([point["seconds"] for point in points])
    complete = summary["status"] == "complete" and updates[-1] == config["updates"]
    log_values = np.log(np.maximum(values, 1e-30))
    log_auc = (
        float(np.sum(np.diff(updates) * (log_values[1:] + log_values[:-1]) / 2) / config["updates"])
        if complete
        else None
    )
    row = {
        "id": path.name,
        "seed": config["seed"],
        "method": config["method"],
        "lr": config["lr"],
        "objective": config["objective"],
        "topk": config["topk"],
        "precision": config["precision"],
        "search_every": config["search_every"],
        "monitor_positions": config["monitor_positions"],
        "status": summary["status"],
        "updates": int(updates[-1]),
        "backwards": summary["counts"]["backwards"],
        "forwards": summary["counts"]["forwards"],
        "seconds": summary["seconds"],
        "final_train_ratio": float(values[-1]),
        "validation_bpo": points[-1]["metrics"]["validation"]["bpo"],
        "validation_teacher_kl": points[-1]["metrics"]["validation"]["teacher_reverse_kl"],
        "log_auc": log_auc,
    }
    for threshold in (0.5, 0.2, 0.1, 0.01, 0.001):
        first = next(
            (i for i in range(len(values) - 1) if max(values[i : i + 2]) <= threshold), None
        )
        last_bad = max((i for i, value in enumerate(values) if value > threshold), default=-1)
        stable = last_bad + 1
        row[f"two_audits_{threshold}"] = int(updates[first]) if first is not None else None
        row[f"persistent_{threshold}"] = int(updates[stable]) if stable < len(updates) else None
    return {"summary": summary, "points": points, "row": row, "ratios": values, "seconds": seconds}


def rows_at(root):
    if not root.exists():
        return []
    return [
        result
        for path in sorted(root.iterdir())
        if path.is_dir() and (result := read_run(path)) is not None
    ]


def select_calibration(root, output):
    runs = rows_at(root)
    constants = sorted(
        (r for r in runs if r["row"]["method"] == "constant" and r["row"]["status"] == "complete"),
        key=lambda r: r["row"]["log_auc"],
    )
    scalars = sorted(
        (
            r
            for r in runs
            if r["row"]["method"] in ("search", "feedback") and r["row"]["status"] == "complete"
        ),
        key=lambda r: r["row"]["log_auc"],
    )
    if len(constants) < 2 or not scalars:
        raise ValueError("calibration has too few completed baseline/candidate arms")
    feedback = min(
        (r for r in scalars if r["row"]["method"] == "feedback"), key=lambda r: r["row"]["log_auc"]
    )
    selection = {
        "criterion": "normalized full-BPO log curve integral over 240 updates",
        "baseline_configs": [r["summary"]["config"] for r in constants[:2]],
        "primary_scalar_config": scalars[0]["summary"]["config"],
        "feedback_target": feedback["summary"]["config"]["target_rate"],
        "rankings": [
            r["row"]
            for r in sorted(
                runs,
                key=lambda r: r["row"]["log_auc"] if r["row"]["log_auc"] is not None else math.inf,
            )
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    return selection


def exact_sign_test(improvements):
    nonzero = [value for value in improvements if value != 0]
    wins = sum(value > 0 for value in nonzero)
    n = len(nonzero)
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0


def summarize(root, output, selection_path):
    runs = rows_at(root)
    output.mkdir(parents=True, exist_ok=True)
    if not runs:
        return
    with (output / "runs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]["row"]))
        writer.writeheader()
        writer.writerows(r["row"] for r in runs)
    if selection_path is None or not selection_path.exists():
        return
    selection = json.loads(selection_path.read_text())
    baseline_lr = selection["baseline_configs"][0]["lr"]
    candidate = selection["primary_scalar_config"]
    comparisons = []
    for seed in (42, 142, 242, 342, 442):
        same_seed = [r for r in runs if r["row"]["seed"] == seed]
        baseline = next(
            (
                r
                for r in same_seed
                if r["row"]["method"] == "constant" and r["row"]["lr"] == baseline_lr
            ),
            None,
        )
        scalar = next(
            (
                r
                for r in same_seed
                if r["summary"]["config"]["method"] == candidate["method"]
                and r["summary"]["config"]["search_every"] == candidate["search_every"]
                and r["summary"]["config"]["target_rate"] == candidate["target_rate"]
            ),
            None,
        )
        if baseline is None or scalar is None:
            continue
        common_time = min(baseline["seconds"][-1], scalar["seconds"][-1])
        baseline_time = float(
            np.exp(np.interp(common_time, baseline["seconds"], np.log(baseline["ratios"])))
        )
        scalar_time = float(
            np.exp(np.interp(common_time, scalar["seconds"], np.log(scalar["ratios"])))
        )
        comparisons.append(
            {
                "seed": seed,
                "baseline": baseline["row"]["id"],
                "scalar": scalar["row"]["id"],
                "final_ratio_gain": 1 - scalar["ratios"][-1] / baseline["ratios"][-1],
                "matched_time_gain": 1 - scalar_time / baseline_time,
                "log_auc_gain": baseline["row"]["log_auc"] - scalar["row"]["log_auc"],
                "validation_bpo_ratio": scalar["row"]["validation_bpo"]
                / baseline["row"]["validation_bpo"],
                "validation_teacher_kl_ratio": scalar["row"]["validation_teacher_kl"]
                / baseline["row"]["validation_teacher_kl"],
            }
        )
    result = {"paired_seeds": len(comparisons), "comparisons": comparisons}
    if len(comparisons) == 5:
        gains = [r["final_ratio_gain"] for r in comparisons]
        times = [r["matched_time_gain"] for r in comparisons]
        result.update(
            median_final_gain=float(np.median(gains)),
            median_matched_time_gain=float(np.median(times)),
            one_sided_sign_p=exact_sign_test(gains),
            alignment_success=bool(
                all(v > 0 for v in gains)
                and np.median(gains) >= 0.1
                and max(r["validation_bpo_ratio"] for r in comparisons) <= 1.05
                and max(r["validation_teacher_kl_ratio"] for r in comparisons) <= 1.05
            ),
            time_success=bool(all(v > 0 for v in times) and np.median(times) >= 0.1),
        )
    (output / "paired-confirmation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.select:
        select_calibration(args.root, args.output)
    else:
        summarize(args.root, args.output, args.selection)


if __name__ == "__main__":
    main()
