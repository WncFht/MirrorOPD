"""Render measured sync/optimization and test diagnostics using figure-viz.

Semantic ledger: each inner point is a full-batch pre-update measurement;
vertical guides are hard-sync boundaries. Test dots are actual greedy test
evaluations, with their sample counts stated in the caption, not training
rewards. Geometry ledger: inner x counts optimizer updates, test x counts
completed outer updates; KL is nats and accuracy is a fraction. Norm and
center have separate axes because they measure distinct quantities.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / ".agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv


def finish(fig, output, name, caption):
    fig.text(0.5, 0.015, caption, ha="center", color=fv.HUES["vgray"], fontsize="small")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    problems = fv.audit(fig)
    (output / f"{name}-audit.json").write_text(json.dumps(problems, indent=2) + "\n")
    errors = [p for p in problems if fv.RENDER_RULES.get(p.split()[0], "error") == "error"]
    if errors:
        raise ValueError(f"Figure audit errors: {errors}")
    fig.savefig(output / f"{name}.png", dpi=220)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis
    summary = json.loads((root / "analysis.json").read_text())
    with (root / "inner-dynamics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fv.apply_rc("paper")
    color = fv.HUES["vteal"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    x = list(range(len(rows)))
    for ax, field, label in zip(
        axes,
        ("grad_norm", "mirror/student_center/mean"),
        ("Pre-clip gradient norm", "Tail-aware student KL, nats"),
        strict=True,
    ):
        ax.plot(x, [float(row[field]) for row in rows], color=color)
        for index, row in enumerate(rows):
            if int(row["inner_step"]) == 0:
                ax.axvline(index, color=fv.HUES["vgray"], linewidth=0.5, linestyle=":")
        ax.set_xlabel("Before optimizer update (zero-based)")
        ax.set_ylabel(label)
        fv.apply_grid(ax, "y-only")
    axes[0].set_yscale("log")
    axes[0].axhline(summary["actual_clip_thresholds"][0], color=fv.NEUTRALS["vink"], linestyle="--")
    finish(fig, root, "inner-dynamics", "Dotted: hard sync. Dashed: clip threshold. Single run.")

    evaluations = summary["evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    x = [row["outer_step"] for row in evaluations]
    for field, label, series_color in (
        ("val-core/openai/gsm8k/acc/mean@1", "Flexible", color),
        ("val-aux/openai/gsm8k/acc_explicit/mean@1", "Explicit", fv.HUES["vorange"]),
    ):
        axes[0].plot(x, [row[field] for row in evaluations], "o-", label=label, color=series_color)
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.30), ncol=2, frameon=False)
    axes[1].plot(
        x,
        [row["val-aux/openai/gsm8k/budget_exhausted/mean@1"] for row in evaluations],
        "o-",
        color=color,
    )
    axes[1].set_ylabel("Budget-exhausted fraction")
    for ax in axes:
        ax.set_xlabel("Completed outer updates")
        visible_max = max(max(line.get_ydata()) for line in ax.lines)
        ax.set_ylim(-0.005, max(0.12, visible_max * 1.15))
        stride = max(1, (len(x) + 4) // 5)
        ax.set_xticks(x[::stride])
        fv.apply_grid(ax, "y-only")
    count = sorted({row["evaluation/generated_examples"] for row in evaluations})
    finish(
        fig,
        root,
        "test-dynamics",
        f"Greedy test; examples per evaluation: {count}. Single training run.",
    )


if __name__ == "__main__":
    main()
