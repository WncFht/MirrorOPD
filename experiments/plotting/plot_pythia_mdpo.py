"""Plot measured Pythia dynamics with the figure-viz paper theme.

Semantic ledger: one line is one learning rate or eta setting in validation
plots, and one pre-update inner index in individual training plots. Eta scan
inner plots average each pre-update index across outer batches by token count.
Validation dots use identical 64
texts, 127 prefixes/text; NLL uses 64 continuation tokens/text. No bands imply
multiple training seeds. Geometry ledger: x is completed outer updates in
validation, current outer batch in inner diagnostics; y units are nats,
dimensionless ESS fraction, or log-ratio RMS as labeled. Each panel has a
distinct physical quantity; colors retain the same meaning across panels.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / ".agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def finish(fig, output, name):
    fig.tight_layout(rect=[0, 0.2 if len(fig.legends[0].texts) > 4 else 0.1, 1, 1])
    problems = fv.audit(fig)
    (output / f"{name}-audit.json").write_text(json.dumps(problems, indent=2) + "\n")
    for problem in problems:
        print(problem)
    errors = [p for p in problems if fv.RENDER_RULES.get(p.split()[0], "error") == "error"]
    if errors:
        raise ValueError(f"Figure audit errors: {errors}")
    fig.savefig(output / f"{name}.png", dpi=220)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--eta-scan", type=Path, help="root containing the five fixed eta runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fv.apply_rc("paper")
    runs = [(args.run, "LR 1e-5", fv.HUES["vteal"])]
    if args.control:
        runs.append((args.control, "LR 1e-6", fv.HUES["vorange"]))
    if args.eta_scan:
        runs = [
            (args.eta_scan / name, label, color)
            for name, label, color in (
                ("pythia-mdpo-eta-0p03", "eta=0.03", fv.HUES["vteal"]),
                ("pythia-mdpo-low-lr", "eta=0.1", fv.HUES["vorange"]),
                ("pythia-mdpo-eta-0p3", "eta=0.3", fv.HUES["vcoral"]),
                ("pythia-mdpo-eta-1", "eta=1", fv.HUES["vviolet"]),
                ("pythia-mdpo-eta-inf", "eta=infinity", fv.NEUTRALS["vink"]),
            )
        ]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    for run, label, color in runs:
        rows = read_csv(run / "checkpoint-metrics.csv")
        for ax, field in zip(axes, ("student_teacher_kl", "student_continuation_nll"), strict=True):
            selected = [row for row in rows if row["metric"] == field]
            ax.plot(
                [int(row["outer_step"]) for row in selected],
                [float(row["mean"]) for row in selected],
                "o-",
                label=label,
                color=color,
            )
            ax.set_xlabel("Completed outer updates")
            ax.set_xticks([0, 8, 16, 24, 32])
    axes[0].set_title("(a) Fixed-prefix full-vocab KL")
    axes[0].set_ylabel("KL(Student || Teacher), nats")
    axes[1].set_title("(b) Held-out continuation NLL")
    axes[1].set_ylabel("NLL, nats/token")
    for ax in axes:
        fv.apply_grid(ax, "y-only")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=3 if args.eta_scan else len(runs),
    )
    if args.eta_scan:
        fig.set_size_inches(5.5, 2.7)
    finish(fig, args.output, "validation-dynamics")

    if args.eta_scan:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.7))
        for run, label, color in runs:
            with (run / "analysis.json").open() as stream:
                stats = json.load(stream)["inner_step_token_weighted"]
            for ax, field in zip(axes, ("log_ratio_rms", "importance_ess_fraction"), strict=True):
                ax.plot(
                    range(1, 5),
                    [stats[str(n)][field] for n in range(4)],
                    "o-",
                    label=label,
                    color=color,
                )
                ax.set_xlabel("Before inner update")
                ax.set_xticks([1, 2, 3, 4])
                fv.apply_grid(ax, "y-only")
        axes[0].set_ylabel("Sampled log-ratio RMS")
        axes[1].set_ylabel("ESS / valid response tokens")
        axes[0].set_title("(a) Departure from frozen anchor")
        axes[1].set_title("(b) Importance weight efficiency")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3)
        finish(fig, args.output, "eta-inner-comparison")
        return

    inner = read_csv(args.run / "inner-metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    colors = fv.series_colors("paper", 4)
    for n, color in enumerate(colors):
        selected = [row for row in inner if int(row["inner_step"]) == n]
        for ax, field in zip(axes, ("log_ratio_rms", "importance_ess_fraction"), strict=True):
            ax.plot(
                [int(row["outer_step"]) for row in selected],
                [float(row[field]) for row in selected],
                label=f"Pre inner {n + 1}",
                color=color,
                linewidth=0.8,
            )
            ax.set_xlabel("Outer batch")
            ax.set_xticks([1, 8, 16, 24, 32])
    axes[0].set_title("(a) Departure from frozen anchor")
    axes[0].set_ylabel("Sampled log-ratio RMS")
    axes[1].set_title("(b) Importance weight efficiency")
    axes[1].set_ylabel("ESS / valid response tokens")
    for ax in axes:
        fv.apply_grid(ax, "y-only")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4)
    finish(fig, args.output, "inner-dynamics")


if __name__ == "__main__":
    main()
