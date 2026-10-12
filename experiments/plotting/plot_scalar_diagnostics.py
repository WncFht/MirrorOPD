"""Local views and explicitly descriptive alignment of scalar BPO runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def name(run):
    config = run["config"]
    if config["method"] == "constant":
        return f"constant {config['lr']:.2g}"
    if config["method"] == "search":
        return f"search every {config['search_every']}"
    return f"feedback {config['target_rate']:.2g}"


def median(values, window=20):
    return np.array([np.median(values[max(0, i - window + 1) : i + 1]) for i in range(len(values))])


def save(fig, output, stem):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def plot(source, output, anchor_step=60, anchor_u=30):
    records = json.loads(source.read_text())["runs"]
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(16, 12), constrained_layout=True)
    windows = [(0, 600), (0, 100), (500, 600)]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    statistics = []
    for index, run in enumerate(records):
        color = colors[index % len(colors)]
        updates = run["lr_updates"]
        steps = np.array([p["update"] for p in updates])
        rates = np.array([p["lr"] for p in updates])
        norms = np.array([p["grad_norm_before_clip"] for p in updates])
        smooth = median(norms)
        audit_steps = np.array([p["update"] for p in run["audit_points"]])
        for column, (low, high) in enumerate(windows):
            audit_mask = (audit_steps >= low) & (audit_steps <= high)
            update_mask = (steps >= low) & (steps <= high)
            for row, key in enumerate(("train_kl", "validation_kl")):
                values = np.array([p[key] for p in run["audit_points"]])
                axes[row, column].plot(
                    audit_steps[audit_mask],
                    values[audit_mask],
                    color=color,
                    marker=".",
                    label=name(run),
                )
            axes[2, column].plot(
                steps[update_mask], norms[update_mask], color=color, alpha=0.18, linewidth=0.6
            )
            axes[2, column].plot(
                steps[update_mask], smooth[update_mask], color=color, linewidth=1.3
            )
            axes[3, column].plot(steps[update_mask], rates[update_mask], color=color, linewidth=0.8)
        statistics.append(
            {
                "id": run["id"],
                "windows": [
                    {
                        "start": low,
                        "end": high,
                        "grad_norm_median": float(
                            np.median(norms[(steps > low) & (steps <= high)])
                        ),
                        "clipped_updates": sum(
                            p["clipped"] for p in updates if low < p["update"] <= high
                        ),
                    }
                    for low, high in ((0, 30), (30, 60), (60, 100), (500, 600))
                ],
            }
        )
    for column, (low, high) in enumerate(windows):
        axes[0, column].set_title(f"Updates {low}--{high}")
        axes[2, column].set_yscale("log")
        axes[2, column].axhline(2, color="black", linestyle=":", linewidth=0.9)
        axes[3, column].set_yscale("symlog", linthresh=1e-10)
        if low <= anchor_step <= high:
            for row in range(4):
                axes[row, column].axvline(anchor_step, color="gray", linestyle="--", linewidth=0.8)
        for row in range(4):
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_xlim(low, high)
        axes[3, column].set_xlabel("Optimizer updates")
    for row, label in enumerate(
        (
            "Train KL(p || teacher)",
            "Held-out KL(p || teacher)",
            "Pre-clip grad norm\nraw + trailing-20 median",
            "Actual applied LR (zero retained)",
        )
    ):
        axes[row, 0].set_ylabel(label)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Seed 42, FP32 sampled BPO, frozen 64 x 256-token pool, minibatch 8\nIndependent 4096-position KL audits; dashed vertical line: descriptive step-60 anchor"
    )
    save(fig, output, "kl-grad-lr-local-seed-42")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for run in records:
        if run["config"]["method"] != "constant":
            continue
        updates = run["lr_updates"]
        steps = np.array([p["update"] for p in updates])
        norms = median(np.array([p["grad_norm_before_clip"] for p in updates]))
        us = np.r_[0, np.cumsum([p["lr"] / 1e-6 for p in updates])]
        audits = run["audit_points"]
        audit_steps = np.array([p["update"] for p in audits])
        values = np.array([p["train_kl"] for p in audits])
        audit_u = us[audit_steps]
        at_step = float(np.interp(anchor_step, audit_steps, values))
        at_u = float(np.interp(anchor_u, audit_u, values))
        for column, x, mask, y in (
            (0, audit_steps, audit_steps >= anchor_step, values),
            (1, audit_steps - anchor_step, audit_steps >= anchor_step, values / at_step),
            (2, audit_u - anchor_u, audit_u >= anchor_u, values / at_u),
        ):
            axes[0, column].plot(x[mask], y[mask], marker=".", label=name(run))
        for column, x, mask in (
            (0, steps, steps >= anchor_step),
            (1, steps - anchor_step, steps >= anchor_step),
            (2, us[steps] - anchor_u, us[steps] >= anchor_u),
        ):
            axes[1, column].plot(x[mask], norms[mask], label=name(run))
    titles = (
        "Absolute KL after step 60",
        "KL / KL(step 60)",
        "KL / KL(u=30): common effective-time anchor",
    )
    labels = ("Optimizer updates", "Updates since step 60", "Effective time since u=30")
    for column in range(3):
        axes[0, column].set_title(titles[column])
        axes[1, column].set_xlabel(labels[column])
        axes[1, column].set_yscale("log")
        for row in range(2):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Train KL(p || teacher)")
    axes[0, 1].set_ylabel("Normalized KL")
    axes[1, 0].set_ylabel("Pre-clip grad norm, trailing-20 median")
    axes[0, 0].legend()
    fig.suptitle(
        "Constant LR comparison; descriptive anchors follow prior d30/d60 diagnostics\nNormalization does not remove Adam moment history or establish causal control"
    )
    save(fig, output, "constant-kl-grad-aligned-seed-42")
    (output / "alignment-statistics.json").write_text(
        json.dumps({"anchor_step": anchor_step, "anchor_u": anchor_u, "runs": statistics}, indent=2)
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
