"""Plot verified evidence for the unified distillation report with figure-viz.

Semantic ledger: a series is one optimizer rule, rollout mode, or parent stage;
points are audited finite-budget aggregates, never independent repetitions.
Quadratic/categorical panels use medians across the five original seeds;
confirmation points are paired response pools. The historical figure has only
saved table checkpoints, with no interpolation presented as measured history.
Geometric ledger: x is the declared case, actual LR, parent update or update
count; y is the stated loss ratio, norm or distance. Equal categories have equal
spacing. Log scales are used only for positive observations; horizontal unity
is a neutral comparison reference, not an observed series. All figures use the
skill palette, a bottom evidence caption, and the complete fv.audit pipeline.
"""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw


def rows(path):
    return list(csv.DictReader(read(path).decode().splitlines()))


def save(fv, fig, output, name, caption, values):
    fig.subplots_adjust(bottom=0.38, top=0.93, left=0.13, right=0.97, wspace=0.47)
    fig.text(0.5, 0.025, caption, ha="center", color=fv.NEUTRALS["baseline"], fontsize="small")
    fig.canvas.draw()
    problems = fv.audit(fig)
    errors = [p for p in problems if fv.problem_severity(p) == "error"]
    record = {
        "figure": name,
        "caption": caption,
        "semantic_and_geometric_ledger": __doc__,
        "values": values,
        "source_sha256": dict(SOURCES),
        "audit": {"ran": True, "errors": errors, "messages": problems},
    }
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    for line in fv.format_audit_lines(problems):
        print(f"{name}: {line}")
    if errors:
        raise ValueError(f"{name}: figure audit failed")
    for ext in ("png", "pdf"):
        metadata = {"Creator": "figure-viz; plot_distillation_study.py"}
        if ext == "pdf":
            metadata |= {"CreationDate": None, "ModDate": None}
        fig.savefig(output / f"{name}.{ext}", dpi=200, metadata=metadata)
    plt.close(fig)


def quadratic(fv, output):
    data = rows(ROOT / "results/scalar-schedule/quadratic-aggregate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.3), sharey=True)
    values = []
    for ax, optimizer in zip(axes, ("gd", "adam"), strict=True):
        cases = [(k, rotated) for k in (10, 1000, 100000) for rotated in (False, True)]
        for method, color in (
            ("constant", "vteal"),
            ("search", "vorange"),
            ("feedback", "vviolet"),
        ):
            points = []
            for case in cases:
                group = [
                    r
                    for r in data
                    if r["optimizer"] == optimizer
                    and tuple(json.loads(r["case"])) == case
                    and r["method"] == method
                    and int(r["hits_0.01"]) == 5
                ]
                steps = min((float(r["median_steps_0.01_successful"]) for r in group), default=None)
                points.append(steps if steps is not None else float("nan"))
                values.append(
                    {
                        "optimizer": optimizer,
                        "case": case,
                        "method": method,
                        "median_steps_all_five_successful": steps,
                    }
                )
            ax.plot(range(6), points, marker="o", label=method, color=fv.HUES[color])
        ax.set(
            yscale="log",
            xlabel="Condition number / coordinates",
            title="GD" if optimizer == "gd" else "AdamW",
        )
        ax.set_xticks(range(6), ["10\nA", "10\nR", "1k\nA", "1k\nR", "100k\nA", "100k\nR"])
    axes[0].set_ylabel("Median updates to 1% loss")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=3,
        frameon=False,
    )
    save(
        fv,
        fig,
        output,
        "quadratic-rule-comparison",
        "Five-seed medians; gaps: not all reached; A: aligned, R: rotated",
        values,
    )


def categorical(fv, output):
    data = rows(ROOT / "results/scalar-schedule/categorical-aggregate.csv")
    cases = [
        case for case in dict.fromkeys(r["case"] for r in data) if json.loads(case)[0] != "exact"
    ]
    values = []
    for case in cases:
        group = [r for r in data if r["case"] == case]
        values.append(
            {
                "case": json.loads(case),
                "constant": min(
                    float(r["median_final_ratio"]) for r in group if r["method"] == "constant"
                ),
                "search": next(
                    float(r["median_final_ratio"]) for r in group if r["method"] == "search"
                ),
            }
        )
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    labels = []
    for value in values:
        case = value["case"]
        labels.append(("L" if case[0] == "sampled" else "N") + f"/{case[1]}/{case[2]}")
    for method, hue in (("constant", "vteal"), ("search", "vorange")):
        ax.plot(
            range(len(values)),
            [r[method] for r in values],
            marker="o",
            color=fv.HUES[hue],
            label=method,
        )
    ax.set(
        yscale="log",
        xlabel="Update field / actions / center support",
        ylabel="Final full BPO / initial",
    )
    ax.set_xticks(range(len(values)), labels, rotation=0, ha="center")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        frameon=False,
    )
    save(
        fv,
        fig,
        output,
        "categorical-field-comparison",
        "Five-seed medians; L: linear, N: nonlinear; fixed actions; update 500",
        values,
    )


def center(fv, output):
    data = json.loads(read(ROOT / "results/scalar-large/center-bias-s1042.json"))
    means = data["means"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1))
    for ax, key, ylabel in zip(
        axes,
        ("teacher_kl_missing", "target_logit_field_norm"),
        ("Missing teacher center", "Target logit-field norm"),
        strict=True,
    ):
        ax.plot(
            [16, 64], [means[f"k{k}_{key}"] for k in (16, 64)], marker="o", color=fv.HUES["vteal"]
        )
        ax.set(xscale="log", yscale="log", xlabel="Head support k", ylabel=ylabel)
        ax.set_xticks([16, 64], ["16", "64"])
        ax.minorticks_off()
    save(
        fv,
        fig,
        output,
        "center-target-bias",
        "40960 fixed prefixes; exact Gibbs target; means, not parameter gradients",
        data,
    )


def confirmation(fv, output):
    data = rows(ROOT / "results/scalar-large/confirm-snapshot-24/runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5))
    values = []
    baselines = {int(r["seed"]): r for r in data if r["id"].startswith("confirm-constant1-")}
    for method, prefix, hue in (
        ("search 1", "confirm-search-every1-", "vorange"),
        ("search 4", "confirm-search-every4-", "vcoral"),
        ("feedback", "confirm-feedback-", "vviolet"),
    ):
        selected = sorted(
            [r for r in data if r["id"].startswith(prefix)], key=lambda r: int(r["seed"])
        )
        assert len(selected) == (4 if method == "feedback" else 5)
        xs, loss, seconds = [], [], []
        for r in selected:
            seed = int(r["seed"])
            base = baselines[seed]
            xs.append(seed)
            loss.append(float(np.exp(float(r["log_auc"]) - float(base["log_auc"]))))
            seconds.append(float(r["seconds"]) / float(base["seconds"]))
            values.append(
                {
                    "method": method,
                    "seed": seed,
                    "geometric_loss_ratio": loss[-1],
                    "seconds_ratio": seconds[-1],
                }
            )
        for ax, yy in zip(axes, (loss, seconds), strict=True):
            ax.plot(xs, yy, marker="o", label=method, color=fv.HUES[hue])
            ax.axhline(1, color=fv.NEUTRALS["baseline"], linestyle=":")
            ax.set(xlabel="Response-pool seed", xticks=[42, 142, 242, 342, 442])
    axes[0].set_ylabel("Geometric full-BPO / constant")
    axes[1].set_ylabel("Elapsed time / constant")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=3,
        frameon=False,
    )
    save(
        fv,
        fig,
        output,
        "paired-confirmation",
        "600 updates; locked constant 5e-7; feedback seed442 cancelled",
        values,
    )


def horizon(fv, output):
    source = json.loads(
        read(
            ROOT
            / "results/scalar-large/fullpool-lr-study/fullstage/figures/same-state-lr-response.json"
        )
    )
    data = rows(ROOT / "results/scalar-large/fullpool-lr-study/fullstage/runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.4))
    for stage, hue in zip(source["stages"], ("vteal", "vorange", "vviolet"), strict=True):
        update = stage["parent_update"]
        label = f"parent {update}"
        xy = np.asarray(stage["one_step_gain_percent"])
        axes[0].plot(xy[:, 0], xy[:, 1], marker="o", color=fv.HUES[hue], label=label)
        group = sorted(
            [r for r in data if r["id"].startswith(f"fullstage-parent{update}-")],
            key=lambda r: float(r["lr"]),
        )
        axes[1].plot(
            [float(r["lr"]) / 1e-6 for r in group],
            [float(r["final_train_ratio"]) for r in group],
            marker="o",
            color=fv.HUES[hue],
            label=label,
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("LR / 1e-6")
        ax.set_xticks([0.125, 0.5, 2, 8], ["0.125", "0.5", "2", "8"])
    axes[0].set_ylabel("One-update full-BPO decrease (%)")
    axes[0].axhline(0, color=fv.NEUTRALS["baseline"], linestyle=":")
    axes[1].set(yscale="log", ylabel="After 40 updates: Q / parent Q")
    axes[1].axhline(1, color=fv.NEUTRALS["baseline"], linestyle=":")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=3,
        frameon=False,
    )
    save(
        fv,
        fig,
        output,
        "same-state-horizon",
        "Same parameters and Adam moments within each parent; 64 full trajectories",
        source["stages"],
    )


def historical(fv, output):
    path = (
        ROOT
        / "data/distillation-report-sources-20261015/2026-10-13-OneShot与镜像蒸馏实验报告.md.gz"
    )
    source = gzip.decompress(read(path)).decode()
    lines = source[source.index("| frozen `ceejzu26`") :].splitlines()
    values = []
    for line in lines:
        if not line.startswith("|"):
            break
        parts = [p.strip() for p in line.strip("|").split("|")]
        values.append(
            {
                "mode": parts[0].split()[0],
                "step": int(parts[1]),
                "absolute": float(parts[2]),
                "signed": float(parts[3]),
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.3))
    for mode, hue in (("frozen", "vteal"), ("online", "vorange")):
        rr = [r for r in values if r["mode"] == mode]
        for ax, key in zip(axes, ("absolute", "signed"), strict=True):
            ax.plot(
                [r["step"] for r in rr],
                [r[key] for r in rr],
                marker="o",
                color=fv.HUES[hue],
                label=mode,
            )
            ax.set(yscale="log", xlabel="Logged historical step")
    axes[0].set_ylabel("Mean absolute action gap")
    axes[1].set_ylabel("Negative mean advantage")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=2,
        frameon=False,
    )
    save(
        fv,
        fig,
        output,
        "historical-fixed-online",
        "Saved historical checkpoints; signed and absolute gaps use different definitions",
        values,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-viz", type=Path, default=ROOT.parent / ".agents/skills/figure-viz"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/distillation-study")
    args = parser.parse_args()
    style = args.figure_viz / "assets/mpl/fvstyle.py"
    sys.path.insert(0, str(style.parent))
    spec = importlib.util.spec_from_file_location("fvstyle", style)
    fv = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fv
    spec.loader.exec_module(fv)
    SOURCES["figure-viz/fvstyle.py"] = hashlib.sha256(style.read_bytes()).hexdigest()
    fv.apply_rc("paper")
    args.output.mkdir(parents=True, exist_ok=True)
    style_sha = SOURCES["figure-viz/fvstyle.py"]
    for function in (historical, quadratic, categorical, center, confirmation, horizon):
        SOURCES.clear()
        SOURCES["figure-viz/fvstyle.py"] = style_sha
        read(Path(__file__))
        function(fv, args.output)


if __name__ == "__main__":
    main()
