"""Preregistered long-absorption and learning-rate time-collapse statistics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from training.oneshot_suite import experiment_plan, selected_experiment_plan

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def history(path, index):
    """Keep the latest record for each update while retaining raw retry ledgers."""
    if not path.exists():
        return []
    # The observer reads a growing JSONL file. A newline commits a record;
    # the writer may still be completing the final line during this read.
    records = [
        json.loads(line)
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]
    by_update = {record[index]: record for record in records}
    return [by_update[update] for update in sorted(by_update)]


def log_curve(times, values, grid):
    """Interpolate positive measured distances without extrapolating."""
    if np.any(np.asarray(values) <= 0):
        raise ValueError("Log-distance interpolation requires positive distances")
    if min(grid) < min(times) or max(grid) > max(times):
        raise ValueError("Time-collapse statistics cannot extrapolate")
    return np.interp(grid, times, np.log(values))


def collapse_statistics(arms):
    """Compare the same numeric grid under raw and effective-time coordinates."""
    grid = np.arange(30.0, 150.1, 1.0)
    raw = np.stack([log_curve(arm["steps"], arm["distance"], grid) for arm in arms])
    effective = np.stack([log_curve(arm["effective_time"], arm["distance"], grid) for arm in arms])
    raw_dispersion = float(np.sqrt(np.mean(np.var(raw, axis=0))))
    effective_dispersion = float(np.sqrt(np.mean(np.var(effective, axis=0))))
    reduction = 1 - effective_dispersion / raw_dispersion if raw_dispersion else 0.0
    full_grid = np.arange(30.0, 300.1, 1.0)
    full_effective = np.stack(
        [log_curve(arm["effective_time"], arm["distance"], full_grid) for arm in arms]
    )
    pair_spread = np.exp(np.max(full_effective, axis=0) - np.min(full_effective, axis=0)) - 1
    return {
        "grid_start": 30,
        "grid_end": 150,
        "raw_log_dispersion": raw_dispersion,
        "effective_log_dispersion": effective_dispersion,
        "dispersion_reduction": reduction,
        "preregistered_30_percent_reduction_passed": reduction >= 0.3,
        "effective_geometric_max_min_ratio": float(
            np.exp(np.mean(np.max(effective, axis=0) - np.min(effective, axis=0)))
        ),
        "full_effective_grid_start": 30,
        "full_effective_grid_end": 300,
        "effective_pair_relative_spread_max": float(pair_spread.max()),
        "effective_pair_relative_spread_median": float(np.median(pair_spread)),
        "preregistered_20_percent_pair_spread_passed": bool(np.all(pair_spread <= 0.2)),
    }


def arm_diagnostics(arm, suite):
    """Distinguish anchor displacement, current teacher distance, and window target error."""
    config = arm["config"]
    records = arm["records"]
    native = arm["fixed_native"]
    result = {
        "initial_native_fixed": native[0]["metrics"],
        "final_native_fixed": native[-1]["metrics"],
        "initial_fp32_readout": arm["fixed_fp32"][0]["metrics"],
        "final_fp32_readout": arm["fixed_fp32"][-1]["metrics"],
        "first_step_anchor_kl": records[1]["metrics"]["student_center/mean"],
        "final_preupdate_anchor_kl": records[-1]["metrics"]["student_center/mean"],
        "clip_count": sum(record["metrics"]["grad_clip_scale"] < 1 for record in records),
        "importance_ess_fraction_min": min(
            record["metrics"]["importance_ess_fraction"] for record in records
        ),
        "importance_ratio_used_in_gradient": config["method"] == "mdpo",
        "first_update_inspected_fp32_change_fraction": records[0]["metrics"][
            "inspected_fp32_changed_fraction"
        ],
        "first_update_inspected_bf16_change_fraction": records[0]["metrics"][
            "inspected_bf16_changed_fraction"
        ],
    }
    if config["anchor_interval"]:
        windows = history(suite / config["id"] / "inner-window-final.jsonl", "update")
        assert [record["update"] for record in windows] == list(
            range(config["anchor_interval"], config["updates"] + 1, config["anchor_interval"])
        )
        result["window_targets_after_full_inner_updates"] = [
            {
                "update": record["update"],
                "current_teacher_kl": record["metrics"]["student_teacher_full_kl"],
                "current_window_target_kl": record["metrics"]["mirror_target_full_kl"],
                "current_window_anchor_kl": record["metrics"]["anchor_student_full_kl"],
                "bpo_target_variance": record["metrics"]["bpo_target_variance"],
            }
            for record in windows
        ]
    return result


def audit_evaluation(directory):
    """Recompute accuracy from all paired question outputs, beyond completion markers."""
    rows = [
        json.loads(line)
        for rank in range(4)
        for line in (directory / f"rank-{rank}.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 570
    assert len({row["uid"] for row in rows}) == 570
    summary = json.loads((directory / "summary.json").read_text())
    for task, expected in (("MATH-500", 500), ("AMC23", 40), ("AIME25", 30)):
        selected = [row for row in rows if row["task"] == task]
        assert len(selected) == expected
        assert summary[task]["correct"] == sum(row["correct"] for row in selected)
        assert summary[task]["accuracy"] == sum(row["correct"] for row in selected) / expected
    return summary


def paired_benchmark_changes(initial, current):
    """Count gains and losses on the same questions without a significance claim."""
    baseline = {
        row["uid"]: row
        for rank in range(4)
        for row in [
            json.loads(line) for line in (initial / f"rank-{rank}.jsonl").read_text().splitlines()
        ]
    }
    rows = [
        json.loads(line)
        for rank in range(4)
        for line in (current / f"rank-{rank}.jsonl").read_text().splitlines()
    ]
    assert {row["uid"] for row in rows} == baseline.keys()
    assert all(
        row["ground_truth"] == baseline[row["uid"]]["ground_truth"]
        and row["prompt"] == baseline[row["uid"]]["prompt"]
        for row in rows
    )
    result = {}
    for task in ("MATH-500", "AMC23", "AIME25"):
        selected = [row for row in rows if row["task"] == task]
        gained = sum(row["correct"] and not baseline[row["uid"]]["correct"] for row in selected)
        lost = sum(not row["correct"] and baseline[row["uid"]]["correct"] for row in selected)
        result[task] = {
            "questions": len(selected),
            "gained_correct": gained,
            "lost_correct": lost,
            "net_correct_change": gained - lost,
            "change_percentage_points": 100 * (gained - lost) / len(selected),
        }
    return result


def audit_completed_arm(suite, config):
    """Require a complete optimizer ledger and final distance before scoring an arm."""
    directory = suite / config["id"]
    records = history(directory / "updates.jsonl", "update_before")
    if not (directory / "completion.json").exists():
        return None
    attempts = sorted((suite / "attempts").glob(config["id"] + "-*"))
    if not attempts or not (attempts[-1] / "exit.json").exists():
        return None
    if json.loads((attempts[-1] / "exit.json").read_text())["exit_code"] != 0:
        return None
    assert [record["update_before"] for record in records] == list(range(config["updates"]))
    final = history(directory / "native-final.jsonl", "update")[-1]
    assert final["update"] == config["updates"]
    steps = [record["update_before"] for record in records] + [final["update"]]
    distance = [record["metrics"]["current_advantage_abs/mean"] for record in records] + [
        final["metrics"]["action_abs_gap"]
    ]
    return {
        "config": config,
        "is_complete": True,
        "logged_updates": len(records),
        "steps": steps,
        "effective_time": [config["lr"] / 1e-6 * step for step in steps],
        "distance": distance,
        "records": records,
        "fixed_native": history(directory / "fixed-native.jsonl", "update"),
        "fixed_fp32": history(directory / "diagnostics.jsonl", "update"),
    }


def observed_curve(suite, config):
    """Plot committed measurements from a running arm without scoring its outcome."""
    records = history(suite / config["id"] / "updates.jsonl", "update_before")
    if not records:
        return None
    return {
        "config": config,
        "is_complete": False,
        "logged_updates": len(records),
        "steps": [record["update_before"] for record in records],
        "effective_time": [record["effective_time_before"] for record in records],
        "distance": [record["metrics"]["current_advantage_abs/mean"] for record in records],
    }


def plot_curves(arms, output):
    """Show each precision/rollout group with all seeds on both time axes."""
    colors = {5e-7: "#2166ac", 1e-6: "#4d9221", 2e-6: "#b2182b"}
    for rollout, precision in (("frozen", "bf16"), ("frozen", "fp32"), ("online", "bf16")):
        selected = [
            arm
            for arm in arms
            if arm["config"]["rollout"] == rollout
            and arm["config"]["precision"] == precision
            and arm["config"]["method"] == "opd"
            and arm["config"]["lr"] > 0
        ]
        if not selected:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for arm in selected:
            config = arm["config"]
            for axis, coordinate in zip(axes, ("steps", "effective_time"), strict=True):
                axis.plot(
                    arm[coordinate],
                    arm["distance"],
                    color=colors[config["lr"]],
                    alpha=0.7,
                    linewidth=1.1,
                    label=f"lr={config['lr']:g}, seed={config['seed']} ({arm['logged_updates']}/{config['updates']} updates logged)",
                )
                axis.set_yscale("log")
                axis.set_xlim(left=0)
                axis.grid(True, alpha=0.2)
                axis.set_ylabel("Native mean absolute teacher log-prob gap")
        axes[0].set_xlabel("Optimizer updates")
        axes[1].set_xlabel("Effective time: updates * lr / 1e-6")
        axes[1].set_xlim(0, 300)
        axes[0].legend(fontsize=7)
        partial = any(not arm["is_complete"] for arm in selected)
        figure.suptitle(
            f"{rollout} OPD / {precision}" + (" / partial observations" if partial else "")
        )
        figure.savefig(output / f"{rollout}-{precision}.png", dpi=180)
        figure.savefig(
            output / f"{rollout}-{precision}.pdf", metadata={"CreationDate": None, "ModDate": None}
        )
        plt.close(figure)


def plot_precision_readouts(suite, config, output):
    """Compare two readouts of identical weights on identical diagnostic positions."""
    directory = suite / config["id"]
    native = {row["update"]: row for row in history(directory / "fixed-native.jsonl", "update")}
    fp32 = {row["update"]: row for row in history(directory / "diagnostics.jsonl", "update")}
    shared = sorted(native.keys() & fp32.keys())
    if len(shared) < 2 or config["precision"] != "bf16" or config["lr"] == 0:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for records, label, color in (
        (native, "Training-precision policy", "#2166ac"),
        (fp32, "FP32 readout of the same weights", "#b2182b"),
    ):
        for axis, metric in zip(
            axes, ("student_teacher_full_kl", "fixed_action_abs_gap"), strict=True
        ):
            axis.plot(
                shared,
                [records[step]["metrics"][metric] for step in shared],
                color=color,
                label=label,
            )
            axis.set_yscale("log")
            axis.set_xlim(left=0)
            axis.set_xlabel("Optimizer updates")
            axis.grid(True, alpha=0.2)
    axes[0].set_ylabel("Full-vocabulary student-to-teacher KL")
    axes[1].set_ylabel("Mean absolute gap on fixed sampled actions")
    axes[0].legend(fontsize=8)
    figure.suptitle(
        config["id"] + (" / partial observations" if shared[-1] < config["updates"] else "")
    )
    figure.savefig(output / (config["id"] + "-readouts.png"), dpi=180)
    figure.savefig(
        output / (config["id"] + "-readouts.pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def main():
    """Summarize completed arms; incomplete conditions remain explicitly pending."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_plan = experiment_plan()
    plan = selected_experiment_plan(args.suite)
    selection_path = args.suite / "execution-selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    arms = [arm for config in plan if (arm := audit_completed_arm(args.suite, config)) is not None]
    lookup = {arm["config"]["id"]: arm for arm in arms}
    report = {
        "requested_arms": len(plan),
        "original_requested_arms": len(original_plan),
        "execution_selection": selection,
        "excluded_arms": [c["id"] for c in original_plan if c["id"] not in {a["id"] for a in plan}],
        "retained_partial_runs": {
            identifier: {
                "logged_updates": len(
                    history(args.suite / identifier / "updates.jsonl", "update_before")
                ),
                "state": "stopped-by-user",
            }
            for identifier in selection.get("retained_partial_ids", [])
        },
        "completed_arms": len(arms),
        "requested_optimizer_updates": sum(config["updates"] for config in plan),
        "completed_optimizer_updates": sum(arm["config"]["updates"] for arm in arms),
        "long_absorption": {},
        "zero_lr_controls": {
            identifier: {
                "anchor_kl_mean_abs_max": max(
                    abs(record["metrics"]["student_center/mean"]) for record in arm["records"]
                ),
                "anchor_kl_rms_max": max(
                    record["metrics"]["student_center/rms"] for record in arm["records"]
                ),
                "inspected_parameter_change_fraction_max": max(
                    record["metrics"]["inspected_fp32_changed_fraction"]
                    for record in arm["records"]
                ),
                "native_distance_change": arm["distance"][-1] - arm["distance"][0],
                "passed": all(
                    abs(record["metrics"]["student_center/mean"]) < 1e-6
                    and record["metrics"]["student_center/rms"] < 1e-5
                    for record in arm["records"]
                ),
            }
            for identifier, arm in lookup.items()
            if arm["config"]["lr"] == 0
        },
        "time_collapse": {},
        "evaluations": {},
        "paired_benchmark_changes": {},
        "arm_diagnostics": {
            identifier: arm_diagnostics(arm, args.suite) for identifier, arm in lookup.items()
        },
        "running_update_counts": {
            config["id"]: len(history(args.suite / config["id"] / "updates.jsonl", "update_before"))
            for config in plan
            if config["id"] not in lookup
        },
        "pending": [config["id"] for config in plan if config["id"] not in lookup],
    }
    for seed in (42, 142, 242):
        identifier = f"frozen-opd-bf16-lr1em6-s{seed}"
        if identifier in lookup:
            arm = lookup[identifier]
            values = np.exp(log_curve(arm["steps"], arm["distance"], [100, 200, 300]))
            first, middle, final = values
            report["long_absorption"][str(seed)] = {
                "d100": float(first),
                "d200": float(middle),
                "d300": float(final),
                "d200_over_d100": float(middle / first),
                "d300_over_d100": float(final / first),
                "passed": bool(middle / first <= 0.8 and final / first <= 0.7),
            }
        for rollout, precision in (("frozen", "bf16"), ("frozen", "fp32"), ("online", "bf16")):
            group = [
                lookup[identifier]
                for tag in ("5em7", "1em6", "2em6")
                if (identifier := f"{rollout}-opd-{precision}-lr{tag}-s{seed}") in lookup
            ]
            if len(group) == 3:
                report["time_collapse"][f"{rollout}-{precision}-s{seed}"] = collapse_statistics(
                    group
                )
    selected_seeds = {str(config["seed"]) for config in plan}
    report["multi_seed_validation_requested"] = len(selected_seeds) > 1
    report["long_absorption_active_seeds_passed"] = (
        all(report["long_absorption"][seed]["passed"] for seed in selected_seeds)
        if selected_seeds <= report["long_absorption"].keys()
        else None
    )
    report["long_absorption_at_least_two_seeds_passed"] = (
        sum(value["passed"] for value in report["long_absorption"].values()) >= 2
        if len(report["long_absorption"]) == 3
        else None
    )
    for path in sorted((args.suite / "evaluation").glob("*/summary.json")):
        completion = json.loads((path.parent / "completion.json").read_text())
        assert completion["unique_questions"] == 570
        report["evaluations"][path.parent.name] = audit_evaluation(path.parent)
    for config in plan:
        initial = args.suite / "evaluation" / f"initial-s{config['seed']}"
        for step in config["checkpoint_steps"]:
            identifier = config["id"] + f"-update{step}"
            if identifier in report["evaluations"]:
                report["paired_benchmark_changes"][identifier] = paired_benchmark_changes(
                    initial, args.suite / "evaluation" / identifier
                )
    expected_evaluations = (
        {f"initial-s{config['seed']}" for config in plan}
        | {"teacher"}
        | {
            config["id"] + f"-update{step}"
            for config in plan
            for step in config["checkpoint_steps"]
        }
    )
    report["pending_evaluations"] = sorted(expected_evaluations - report["evaluations"].keys())
    report["all_evaluations_complete"] = not report["pending_evaluations"]
    report["all_training_complete"] = len(arms) == len(plan)
    observed = [
        curve for config in plan if (curve := observed_curve(args.suite, config)) is not None
    ]
    report["observed_optimizer_updates"] = sum(len(curve["steps"]) for curve in observed)
    with (args.output / "native-distance.csv").open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["arm", "update", "effective_time", "native_action_abs_gap"])
        for arm in arms:
            writer.writerows(
                (arm["config"]["id"], step, effective, distance)
                for step, effective, distance in zip(
                    arm["steps"], arm["effective_time"], arm["distance"], strict=True
                )
            )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_curves([lookup.get(curve["config"]["id"], curve) for curve in observed], args.output)
    for config in original_plan:
        if config in plan or config["id"] in selection.get("retained_partial_ids", []):
            plot_precision_readouts(args.suite, config, args.output)
    print(
        json.dumps(
            {"completed": len(arms), "requested": len(plan), "pending": len(report["pending"])}
        )
    )


if __name__ == "__main__":
    main()
