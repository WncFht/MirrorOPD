"""Summarize measured host-memory profiles and render reproducible figures."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.style))
    import fvstyle as fv

    fv.apply_rc("paper")
    gib = 1024**3
    system = rows(args.root / "system.jsonl")
    phases = rows(args.root / "original-pipeline/host-memory.jsonl")
    start, end = phases[0]["time"] - 5, phases[-1]["time"] + 15
    subset = [r for r in system if start <= r["time"] <= end]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    times = [(r["time"] - start) / 60 for r in subset]
    axes[0].plot(
        times,
        [(r["virtual"]["total"] - r["virtual"]["available"]) / gib for r in subset],
        label="System total - available",
        color=fv.HUES["vteal"],
    )
    axes[0].plot(
        times,
        [r["swap"]["used"] / gib for r in subset],
        label="Swap used",
        color=fv.HUES["vorange"],
    )
    worker_pid = phases[0]["pid"]
    axes[1].plot(
        times,
        [
            sum(p["pss"] or 0 for p in r["processes"] if p["pid"] == worker_pid) / gib
            for r in subset
        ],
        label="Actor/teacher worker PSS",
        color=fv.HUES["vteal"],
    )
    labels = set()
    for r in phases:
        if r["stage"] not in ("before_eval_enter", "before_rollout_wake"):
            continue
        x = (r["time"] - start) / 60
        label = r["stage"] if r["stage"] not in labels else None
        axes[1].axvline(x, color=fv.NEUTRALS["baseline"], linestyle="--", alpha=0.6, label=label)
        labels.add(r["stage"])
    for ax in axes:
        ax.set_ylabel("GiB")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    axes[1].set_xlabel("Minutes from initial weight sync")
    fig.text(
        0.5,
        0.01,
        "Measured B64/J16 original CPUOffload pipeline; 0.5 s system sampling",
        ha="center",
        fontsize=8,
        color=fv.NEUTRALS["baseline"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.root / "original-memory-timeline.png", dpi=180)
    fig.savefig(args.root / "original-memory-timeline.pdf")
    summary = {
        "system_samples": len(subset),
        "system_peak_total_minus_available_gib": max(
            (r["virtual"]["total"] - r["virtual"]["available"]) / gib for r in subset
        ),
        "swap_initial_gib": subset[0]["swap"]["used"] / gib,
        "swap_peak_gib": max(r["swap"]["used"] / gib for r in subset),
    }
    isolated = {}
    for folder in args.root.glob("isolation-*"):
        if not (folder / "stages.jsonl").exists():
            continue
        records = rows(folder / "stages.jsonl")
        isolated[folder.name] = {
            "rss_peak_gib": max(r["rss"] for r in records) / gib,
            "pinned_peak_gib": max(r["pinned"]["reserved_bytes.current"] for r in records) / gib,
            "gradient_bytes": max(r.get("cpu_gradient_bytes", 0) for r in records),
        }
    summary["isolated"] = isolated
    explicit_path = args.root / "explicit-pipeline/host-memory.jsonl"
    if explicit_path.exists():
        explicit = rows(explicit_path)
        system_explicit = rows(args.root / "explicit-system.jsonl")
        es = explicit[0]["time"] - 5
        ee = explicit[-1]["time"] + 15
        ex = [r for r in system_explicit if es <= r["time"] <= ee]
        summary["explicit_pipeline"] = {
            "system_samples": len(ex),
            "system_peak_total_minus_available_gib": max(
                (r["virtual"]["total"] - r["virtual"]["available"]) / gib for r in ex
            ),
            "swap_initial_gib": ex[0]["swap"]["used"] / gib,
            "swap_peak_gib": max(r["swap"]["used"] / gib for r in ex),
            "worker_stage_rss_peak_gib": max(r["before"]["rss_bytes"] for r in explicit) / gib,
        }
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            times,
            [(r["virtual"]["total"] - r["virtual"]["available"]) / gib for r in subset],
            label="Original CPUOffload (1 outer)",
            color=fv.HUES["vcoral"],
        )
        ax.plot(
            [(r["time"] - es) / 60 for r in ex],
            [(r["virtual"]["total"] - r["virtual"]["available"]) / gib for r in ex],
            label="Explicit inference offload (2 outer)",
            color=fv.HUES["vteal"],
        )
        ax.set(xlabel="Minutes from initial weight sync", ylabel="System total - available (GiB)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        fig.text(
            0.5,
            0.01,
            "Measured B64/J16; different rollout RNG histories; capacity comparison",
            ha="center",
            fontsize=8,
            color=fv.NEUTRALS["baseline"],
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(args.root / "offload-memory-comparison.png", dpi=180)
        fig.savefig(args.root / "offload-memory-comparison.pdf")
    bg_ids = {
        p["pid"]
        for r in subset
        for p in r["processes"]
        if p["name"] in ("devin-2api", "claude", "MainThread", "codex")
    }
    summary["observed_background_pss_gib"] = {
        str(pid): [
            min(p["pss"] for r in subset for p in r["processes"] if p["pid"] == pid) / gib,
            max(p["pss"] for r in subset for p in r["processes"] if p["pid"] == pid) / gib,
        ]
        for pid in bg_ids
        if any(p["pid"] == pid for r in subset for p in r["processes"])
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
