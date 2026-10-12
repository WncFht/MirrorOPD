"""Plot absolute teacher reverse KL and actual scalar learning rates."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root, output, seed):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    records = []
    for directory in sorted(root.iterdir()):
        summary = directory / "summary.json"
        if not summary.exists():
            continue
        config = json.loads(summary.read_text())["config"]
        if config["seed"] != seed:
            continue
        audits_path, updates_path = directory / "audits.jsonl", directory / "updates.jsonl"
        audits = [json.loads(line) for line in audits_path.read_text().splitlines()]
        updates = [json.loads(line) for line in updates_path.read_text().splitlines()]
        method = config["method"]
        name = (
            f"constant {config['lr']:.2g}"
            if method == "constant"
            else f"search every {config['search_every']}"
            if method == "search"
            else f"feedback target {config['target_rate']:.2g}"
        )
        line = axes[0].plot(
            [row["update"] for row in audits],
            [row["metrics"]["train"]["teacher_reverse_kl"] for row in audits],
            label=name,
            marker=".",
            markersize=3,
        )[0]
        color = line.get_color()
        axes[1].plot(
            [row["update"] for row in audits],
            [row["metrics"]["validation"]["teacher_reverse_kl"] for row in audits],
            color=color,
            marker=".",
            markersize=3,
        )
        axes[2].plot(
            [row["update_after"] for row in updates],
            [row["applied_lr"] for row in updates],
            color=color,
            linewidth=0.8,
        )
        records.append(
            {
                "id": directory.name,
                "config": config,
                "sources": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (summary, audits_path, updates_path)
                },
                "final_train_kl": audits[-1]["metrics"]["train"]["teacher_reverse_kl"],
                "final_validation_kl": audits[-1]["metrics"]["validation"]["teacher_reverse_kl"],
                "zero_lr_updates": sum(row["applied_lr"] == 0 for row in updates),
                "audit_points": [
                    {
                        "update": row["update"],
                        "train_kl": row["metrics"]["train"]["teacher_reverse_kl"],
                        "validation_kl": row["metrics"]["validation"]["teacher_reverse_kl"],
                    }
                    for row in audits
                ],
                "lr_updates": [
                    {
                        "update": row["update_after"],
                        "lr": row["applied_lr"],
                        "grad_norm_before_clip": row["gradient_norm_before_clip"],
                        "clipped": row["clipped"],
                    }
                    for row in updates
                ],
            }
        )
    if not records:
        raise ValueError("no completed runs for requested seed")
    axes[0].set_ylabel("Train d = KL(student || teacher)")
    axes[1].set_ylabel("Held-out d = KL(student || teacher)")
    axes[2].set_ylabel("Actual applied learning rate")
    axes[2].set_yscale("symlog", linthresh=1e-10)
    axes[2].set_xlabel("Optimizer updates (includes rejected parameter updates)")
    axes[0].legend(fontsize=9, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
    config = records[0]["config"]
    batch_label = (
        "full pool" if config.get("full_batch") else f"minibatch {config['batch_sequences']}"
    )
    fig.suptitle(
        f"Frozen sampled-token BPO, seed {seed}; {config['precision'].upper()}, "
        f"{config['train_sequences']} x 256-token pool, {batch_label}\n"
        "KL: full vocabulary at 4096 fixed positions per split; LR: every update"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"teacher-kl-and-lr-seed-{seed}.{suffix}", dpi=180)
    plt.close(fig)
    (output / f"teacher-kl-and-lr-seed-{seed}.json").write_text(
        json.dumps({"kl_direction": "student || teacher", "runs": records}, indent=2) + "\n"
    )
    print(json.dumps({"runs": len(records), "output": str(output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plot(args.root, args.output, args.seed)


if __name__ == "__main__":
    main()
