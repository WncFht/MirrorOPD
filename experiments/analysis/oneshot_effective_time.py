"""Preregistered long-absorption and learning-rate time-collapse statistics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from training.oneshot_suite import experiment_plan

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def history(path, index):
    """Keep the latest record for each update while retaining raw retry ledgers."""
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines()]
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
    }


def audit_completed_arm(suite, config):
    """Require a complete optimizer ledger and final distance before scoring an arm."""
    directory = suite / config["id"]
    records = history(directory / "updates.jsonl", "update_before")
    if not (directory / "completion.json").exists():
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
        "steps": steps,
        "effective_time": [config["lr"] / 1e-6 * step for step in steps],
        "distance": distance,
        "records": records,
        "fixed_native": history(directory / "fixed-native.jsonl", "update"),
        "fixed_fp32": history(directory / "diagnostics.jsonl", "update"),
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
                    label=f"lr={config['lr']:g}, seed={config['seed']}",
                )
                axis.set_yscale("log")
                axis.set_xlim(left=0)
                axis.grid(True, alpha=0.2)
                axis.set_ylabel("Native mean absolute teacher log-prob gap")
        axes[0].set_xlabel("Optimizer updates")
        axes[1].set_xlabel("Effective time: updates * lr / 1e-6")
        axes[1].set_xlim(0, 300)
        axes[0].legend(fontsize=7)
        figure.suptitle(f"{rollout} OPD / {precision}")
        figure.savefig(output / f"{rollout}-{precision}.png", dpi=180)
        figure.savefig(output / f"{rollout}-{precision}.pdf")
        plt.close(figure)


def main():
    """Summarize completed arms; incomplete conditions remain explicitly pending."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = experiment_plan()
    arms = [arm for config in plan if (arm := audit_completed_arm(args.suite, config)) is not None]
    lookup = {arm["config"]["id"]: arm for arm in arms}
    report = {
        "requested_arms": len(plan),
        "completed_arms": len(arms),
        "requested_optimizer_updates": sum(config["updates"] for config in plan),
        "completed_optimizer_updates": sum(arm["config"]["updates"] for arm in arms),
        "long_absorption": {},
        "time_collapse": {},
        "evaluations": {},
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
                "passed": middle / first <= 0.8 and final / first <= 0.7,
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
    report["long_absorption_at_least_two_seeds_passed"] = (
        sum(value["passed"] for value in report["long_absorption"].values()) >= 2
        if len(report["long_absorption"]) == 3
        else None
    )
    for path in sorted((args.suite / "evaluation").glob("*/summary.json")):
        completion = json.loads((path.parent / "completion.json").read_text())
        assert completion["unique_questions"] == 570
        report["evaluations"][path.parent.name] = json.loads(path.read_text())
    report["all_training_complete"] = len(arms) == len(plan)
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
    plot_curves(arms, args.output)
    print(
        json.dumps(
            {"completed": len(arms), "requested": len(plan), "pending": len(report["pending"])}
        )
    )


if __name__ == "__main__":
    main()
