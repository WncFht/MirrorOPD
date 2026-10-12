"""One curated outer/inner review for mathematical distillation experiments."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from experiments.analysis.math_run_records import (
    BENCHMARKS,
    COLORS,
    RUNS,
    load_run,
    metrics,
    read_rows,
    write_csv,
)

ETA = 0.3  # The three frozen formal runs share this verified setting.
TIMING_KEYS = ["gen", "old_log_prob", "mirror_teacher_cache", "update_actor", "update_weights"]
FIGURE_SPECS = [
    ("01-quality-budgets", "outer", "mean correctness", "rollouts, updates, training minutes"),
    (
        "02-outer-learning",
        "outer",
        "teacher TA KL, sampled KL, verifier, entropy, length, clipping",
        "frozen k",
    ),
    (
        "03-outer-optimization",
        "outer",
        "coefficient RMS, gradient, anchor KL, ESS, overlap",
        "frozen k",
    ),
    (
        "04-inner-progress",
        "inner",
        "coefficient RMS, gradient, anchor KL, objective estimate",
        "pre-update n",
    ),
    ("05-inner-heatmaps", "inner", "RMS and gradient relative to own n=0; anchor KL", "(k,n)"),
    ("06-is-ess", "outer and inner", "ESS deficit, ratio extrema, weight std and U RMS", "k and n"),
    (
        "07-generation-budget",
        "outer",
        "test response tokens and budget exhaustion",
        "completed outer",
    ),
    (
        "08-resources",
        "resources",
        "exclusive top-level seconds, training time, device memory/utilization",
        "k and wall time",
    ),
    ("09-log-semantics", "logging", "SDK steps and duplicate-x center sources", "actual W&B axes"),
    (
        "10-reference-protocol",
        "evaluation",
        "historical student/teacher mean@16",
        "separate T=.7,p=.9,L=31744 protocol",
    ),
]


def finish(fig, axes, output, name):
    """Publish a fixed PNG/PDF pair with a common scientific plotting style."""
    for ax in np.asarray(axes).ravel():
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=190)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def line(ax, x, y, method, **kwargs):
    """Keep method colors stable across all curated panels."""
    ax.plot(x, y, color=COLORS[method], label=method, **kwargs)


def evaluation_points(run, benchmark):
    """Pair a complete evaluation with its actual optimizer and training budget."""
    points = []
    for row in run["evaluation"]:
        k, data, repeats = row["step"], row["data"], run["R"]
        points.append(
            {
                "outer": k,
                "updates": k * run["N"],
                "training_minutes": run["times"][k] / 60,
                "accuracy": data[f"val-core/{benchmark}/acc/mean@{repeats}"],
                "tokens": data[f"val-aux/{benchmark}/response_tokens/mean@{repeats}"],
                "exhausted": data[f"val-aux/{benchmark}/budget_exhausted/mean@{repeats}"],
            }
        )
    return points


def gpu_series(directory):
    """Align consecutive four-device monitoring samples, including startup/tests."""
    with (directory / "gpu.csv").open() as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    per_device = [[r for r in rows if int(r["index"]) == gpu] for gpu in range(4)]
    assert {int(r["index"]) for r in rows} == {0, 1, 2, 3}
    count = min(map(len, per_device))
    start = datetime.strptime(per_device[0][0]["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
        tzinfo=timezone.utc
    )
    times = [
        (
            datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            - start
        ).total_seconds()
        / 60
        for r in per_device[0][:count]
    ]
    memory = np.asarray(
        [[float(r["memory.used [MiB]"].split()[0]) / 1024 for r in rs[:count]] for rs in per_device]
    )
    utilization = np.asarray(
        [[float(r["utilization.gpu [%]"].split()[0]) for r in rs[:count]] for rs in per_device]
    )
    return np.asarray(times), memory.max(axis=0), utilization.mean(axis=0)


def summarize(runs):
    """Keep tables independent of the selected figure layout."""
    summary = {"eta": ETA, "methods": {}, "protocol": next(iter(runs.values()))["protocol"]}
    for method, run in runs.items():
        inner = run["inner"]
        ess = metrics(inner, "importance_ess_fraction")
        mean, std = metrics(inner, "importance_ratio/mean"), metrics(inner, "importance_ratio/std")
        np.testing.assert_allclose(ess, 1 / (1 + (std / mean) ** 2), atol=1e-12)
        grad = np.asarray([r["metrics"]["grad_norm"] for r in inner])
        times, memory, utilization = gpu_series(run["directory"])
        groups = run["groups"]
        rms = np.asarray([metrics(g, "score_coefficient/rms") for g in groups])
        norms = np.asarray([[r["metrics"]["grad_norm"] for r in g] for g in groups])
        item = {
            "run": run["name"],
            "N": run["N"],
            "R": run["R"],
            "completed_outer": run["completed"],
            "updates": len(inner),
            "pending_inner_excluded": run["pending_inner"],
            "evaluation_outer": [r["step"] for r in run["evaluation"]],
            "training_seconds": float(run["times"][-1]),
            "ess_min": float(ess.min()),
            "ess_mean": float(ess.mean()),
            "ratio_max": float(metrics(inner, "importance_ratio/max").max()),
            "ratio_min": float(metrics(inner, "importance_ratio/min").min()),
            "gradient_min": float(grad.min()),
            "gradient_max": float(grad.max()),
            "clipped_updates": int(sum(r["metrics"]["grad_was_clipped"] for r in inner)),
            "device_peak_gib": float(memory.max()),
            "device_mean_utilization": float(utilization.mean()),
            "monitor_minutes": float(times[-1]),
            "c_rms_last_over_first": (rms[:, -1] / rms[:, 0]).tolist(),
            "grad_last_over_first": (norms[:, -1] / norms[:, 0]).tolist(),
            "benchmarks": {b: evaluation_points(run, b) for b in BENCHMARKS},
            "mean_phase_seconds": {
                key: float(np.mean([r["data"]["timing_s/" + key] for r in run["training"]]))
                for key in TIMING_KEYS
            },
        }
        for key in ["teacher_center/mean", "sampled_reward/mean", "student_entropy/mean"]:
            values = metrics(run["first"], key)
            item[key] = {
                "first": float(values[0]),
                "last": float(values[-1]),
                "first10": float(values[:10].mean()),
                "last10": float(values[-10:].mean()),
            }
        summary["methods"][method] = item
    return summary


def render_review(raw, bpo_raw, output, *, include_wandb=True, reference_raw=None):
    """Generate ten complementary figure groups, with one common source snapshot."""
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    runs = {m: load_run(bpo_raw if m == "BPO" else raw, name) for m, name in RUNS.items()}
    assert len({r["protocol"] for r in runs.values()}) == 1, (
        "Different evaluation protocols require separate panels"
    )
    common = min(r["completed"] for r in runs.values())
    summary = summarize(runs)
    summary["common_outer_window"] = common
    selected = {0, min(4, common - 1), min(9, common - 1), min(11, common - 1), common - 1}

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for col, benchmark in enumerate(BENCHMARKS):
        for method, run in runs.items():
            points = evaluation_points(run, benchmark)
            for row, key, label in [
                (0, "outer", "Completed fresh rollouts"),
                (1, "updates", "Completed optimizer updates"),
                (2, "training_minutes", "Training minutes; validation excluded"),
            ]:
                line(
                    axes[row, col],
                    [p[key] for p in points],
                    [100 * p["accuracy"] for p in points],
                    method,
                    marker="o",
                    markersize=3,
                )
                axes[row, col].set(
                    title=benchmark, xlabel=label, ylabel=f"mean@{run['R']} correctness (%)"
                )
    fig.suptitle(f"{summary['protocol']}; BPO completed outers={runs['BPO']['completed']}")
    finish(fig, axes, figures, "01-quality-budgets")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for method, run in runs.items():
        x = np.arange(run["completed"])
        ys = [
            metrics(run["first"], "teacher_center/mean"),
            -metrics(run["first"], "sampled_reward/mean"),
            np.asarray([r["data"]["critic/score/mean"] for r in run["training"]]),
            metrics(run["first"], "student_entropy/mean"),
            np.asarray([r["data"]["response_length/mean"] for r in run["training"]]),
            np.asarray([r["data"]["response_length/clip_ratio"] for r in run["training"]]),
        ]
        labels = [
            r"$\overline{\widehat{\mathcal{K}}_k^{T,\mathrm{TA}}}$ (nats)",
            r"$-\bar r_k$ (nats)",
            r"$B^{-1}\sum_i\mathrm{score}_i$",
            r"$\overline{H(q)}$ (nats)",
            r"$B^{-1}\sum_i H_i$ (tokens)",
            r"$B^{-1}\sum_i\mathbf{1}\{H_i=7168\}$",
        ]
        for ax, y, label in zip(axes.flat, ys, labels, strict=True):
            line(ax, x, y, method)
            ax.set(xlabel="Frozen rollout k; before its updates", ylabel=label)
    finish(fig, axes, figures, "02-outer-learning")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for method, run in runs.items():
        x = np.arange(run["completed"])
        for col, key in enumerate(["score_coefficient/rms", "grad_norm"]):
            if key == "grad_norm":
                first = [r["metrics"][key] for r in run["first"]]
                last = [r["metrics"][key] for r in run["last"]]
            else:
                first, last = metrics(run["first"], key), metrics(run["last"], key)
            axes[0, col].plot(x, first, "--", color=COLORS[method], label=f"{method}: n=0")
            if run["N"] > 1:
                axes[0, col].plot(
                    x, last, color=COLORS[method], label=f"{method}: n={run['N'] - 1}"
                )
        ratio = np.asarray(summary["methods"][method]["c_rms_last_over_first"])
        line(axes[0, 2], x, ratio, method)
        line(axes[1, 0], x, metrics(run["last"], "student_center/mean"), method)
        ess = np.asarray([metrics(g, "importance_ess_fraction").min() for g in run["groups"]])
        line(axes[1, 1], x, 100 * (1 - ess), method)
        native_pairs = ["student_teacher", "student_anchor", "anchor_teacher"]
        if "mirror/overlap_student_teacher_intersection_fraction/mean" in run["last"][0]["metrics"]:
            for pair, style in zip(native_pairs, ["-", "--", ":"], strict=True):
                axes[1, 2].plot(
                    x,
                    metrics(run["last"], f"overlap_{pair}_intersection_fraction/mean"),
                    color=COLORS[method],
                    linestyle=style,
                    label=f"{method}: {pair}",
                )
        else:
            line(axes[1, 2], x, metrics(run["last"], "support_overlap/mean"), method)
    labels = [
        r"$\sqrt{\overline{c^2}}$",
        r"$\Vert g\Vert_2$; before clip",
        r"$\mathrm{RMS}(c_{k,N-1})/\mathrm{RMS}(c_{k,0})$",
        r"$\overline{\widehat{\mathcal{K}}_{k,N-1}^{S,\mathrm{TA}}}$ (nats)",
        r"$100(1-\min_n f_{k,n})$ (%)",
        r"$\overline{|\mathrm{Top}_J(p)\cap\mathrm{Top}_J(q)|/J}$",
    ]
    for ax, label in zip(axes.flat, labels, strict=True):
        ax.set(xlabel="Frozen rollout k", ylabel=label)
    if all(
        "mirror/overlap_student_teacher_intersection_fraction/mean" not in r["last"][0]["metrics"]
        for r in runs.values()
    ):
        axes[1, 2].set_title("Legacy S-anchor only; S-teacher not recorded")
    finish(fig, axes, figures, "03-outer-optimization")

    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    for row, (method, run) in enumerate(runs.items()):
        for k in sorted({*selected, run["completed"] - 1}):
            group = run["groups"][k]
            rms = metrics(group, "score_coefficient/rms")
            obj = (
                metrics(group, "teacher_center/mean")
                if method == "OPD"
                else metrics(group, "score_coefficient/mean")
                if method == "MDPO"
                else ETA / 2 * rms**2
            )
            ys = [
                rms,
                [r["metrics"]["grad_norm"] for r in group],
                metrics(group, "student_center/mean"),
                obj,
            ]
            for col, y in enumerate(ys):
                axes[row, col].plot(np.arange(run["N"]), y, "o-", markersize=3, label=f"k={k}")
                axes[row, col].set(title=method, xlabel="Pre-update inner n")
                if run["N"] == 1:
                    axes[row, col].set_xticks([0])
        axes[row, 0].set_ylabel(r"$\sqrt{\overline{c^2}}$")
        axes[row, 1].set_ylabel(r"$\Vert g\Vert_2$")
        axes[row, 2].set_ylabel(r"$\overline{\widehat{\mathcal{K}}^S}$ (nats)")
        if run["N"] == 1:
            axes[row, 2].set_ylim(-1e-6, 1e-6)
            axes[row, 2].text(
                0.05, 0.85, r"$\mathcal{K}^S\simeq0$ at n=0", transform=axes[row, 2].transAxes
            )
        axes[row, 3].set_ylabel(
            {
                "OPD": r"TA teacher KL (only pre-update)",
                "MDPO": r"$\widehat L_{\mathrm{MDPO,IS}}=\bar c$",
                "BPO": r"TA residual proxy: $\eta\overline{c^2}/2$",
            }[method]
        )
    finish(fig, axes, figures, "04-inner-progress")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, method in enumerate(["MDPO", "BPO"]):
        run = runs[method]
        rms = np.asarray([metrics(g, "score_coefficient/rms") for g in run["groups"]])
        grad = np.asarray([[r["metrics"]["grad_norm"] for r in g] for g in run["groups"]])
        center = np.asarray([metrics(g, "student_center/mean") for g in run["groups"]])
        for col, matrix in enumerate([rms / rms[:, :1], grad / grad[:, :1], center]):
            options = (
                {
                    "norm": LogNorm(
                        vmin=min(0.1, float(matrix.min())), vmax=max(1.4, float(matrix.max()))
                    )
                }
                if col < 2
                else {"vmin": 0}
            )
            im = axes[row, col].imshow(
                matrix, origin="lower", aspect="auto", cmap="viridis", **options
            )
            axes[row, col].set(
                title=method
                + ": "
                + ["RMS(c) / own n=0", "gradient / own n=0", "anchor-student TA KL (nats)"][col],
                xlabel="Pre-update n",
                ylabel="Frozen rollout k",
            )
            fig.colorbar(im, ax=axes[row, col])
    finish(fig, axes, figures, "05-inner-heatmaps")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for method, run in runs.items():
        x = np.arange(run["completed"])
        values = np.asarray([metrics(g, "importance_ess_fraction") for g in run["groups"]])
        line(axes[0, 0], x, 100 * (1 - values.min(axis=1)), method)
        line(
            axes[0, 1], x, [metrics(g, "importance_ratio/max").max() for g in run["groups"]], method
        )
        line(
            axes[0, 2], x, [metrics(g, "importance_ratio/min").min() for g in run["groups"]], method
        )
        for col, key in enumerate(
            ["importance_ess_fraction", "importance_ratio/std", "sampled_log_ratio/rms"]
        ):
            matrix = np.asarray([metrics(g, key) for g in run["groups"][:common]])
            line(
                axes[1, col],
                np.arange(run["N"]),
                matrix.mean(axis=0),
                method,
                marker="o",
                markersize=3,
            )
            axes[1, col].fill_between(
                np.arange(run["N"]),
                matrix.min(axis=0),
                matrix.max(axis=0),
                color=COLORS[method],
                alpha=0.13,
            )
            if run["N"] == 1:
                axes[1, col].set_xticks([0])
    labels = [
        r"$100(1-\min_n f_{k,n})$ (%)",
        r"$\max_{n,j}\omega_j$",
        r"$\min_{n,j}\omega_j$",
        r"$f=\mathrm{ESS}/M_k$",
        r"$\sqrt{\overline{\omega^2}-\bar\omega^2}$",
        r"$\sqrt{\overline{U^2}}$ (nats)",
    ]
    for i, (ax, label) in enumerate(zip(axes.flat, labels, strict=True)):
        ax.set(
            xlabel="Frozen rollout k" if i < 3 else f"Inner n; common k=0..{common - 1}",
            ylabel=label,
        )
    axes[0, 2].set_yscale("log")
    finish(fig, axes, figures, "06-is-ess")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col, benchmark in enumerate(BENCHMARKS):
        for method, run in runs.items():
            points = evaluation_points(run, benchmark)
            for row, key, scale, label in [
                (0, "tokens", 1, r"$D^{-1}\sum_i H_i$ (tokens)"),
                (1, "exhausted", 100, "Budget-exhausted responses (%)"),
            ]:
                line(
                    axes[row, col],
                    [p["outer"] for p in points],
                    [p[key] * scale for p in points],
                    method,
                    marker="o",
                    markersize=3,
                )
                axes[row, col].set(title=benchmark, xlabel="Completed outer", ylabel=label)
    finish(fig, axes, figures, "07-generation-budget")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    bottom = np.zeros(len(runs))
    for key in [*TIMING_KEYS, "other"]:
        vals = []
        for method, run in runs.items():
            if key == "other":
                v = np.mean(
                    [
                        r["data"]["timing_s/step"]
                        - sum(r["data"]["timing_s/" + k] for k in TIMING_KEYS)
                        for r in run["training"]
                    ]
                )
            else:
                v = summary["methods"][method]["mean_phase_seconds"][key]
            vals.append(v)
        axes[0, 0].bar(list(runs), vals, bottom=bottom, label=key)
        bottom += vals
    axes[0, 0].set(
        ylabel="Mean seconds / completed outer", title="Exclusive top-level training timing"
    )
    for method, run in runs.items():
        line(axes[0, 1], np.arange(run["completed"] + 1), run["times"] / 60, method)
        time, mem, util = gpu_series(run["directory"])
        line(axes[1, 0], time, mem, method, linewidth=0.7)
        line(axes[1, 1], time, util, method, linewidth=0.7)
    axes[0, 1].set(xlabel="Completed outer", ylabel="Cumulative training minutes")
    axes[1, 0].set(
        xlabel="Monitor wall minutes; includes validation",
        ylabel="Max of four device memories (GiB)",
    )
    axes[1, 1].set(
        xlabel="Monitor wall minutes; includes validation",
        ylabel="Mean of four GPU utilization (%)",
    )
    finish(fig, axes, figures, "08-resources")

    if include_wandb:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for method, run in runs.items():
            rows = read_rows(run["directory"] / "wandb-history.jsonl")
            quality_key = "01_quality/val-core/MATH-500/acc/mean@1"
            qrows = [r for r in rows if r.get(quality_key) is not None]
            line(
                axes[0],
                [r["_step"] for r in qrows],
                [100 * r[quality_key] for r in qrows],
                method,
                marker="o",
                markersize=3,
            )
            center_key = "04_kl_centers/outer_student_center/mean"
            crows = [r for r in rows if r.get(center_key) is not None]
            line(
                axes[1],
                [r["00_axes/outer_rollout_step"] for r in crows],
                [r[center_key] for r in crows],
                method,
            )
            first = [r for r in crows if r["00_axes/outer_rollout_step"] == 0]
            axes[2].plot(
                np.arange(len(first)),
                [r[center_key] for r in first],
                "o-",
                color=COLORS[method],
                label=method,
            )
        axes[0].set(
            xlabel="W&B SDK _step",
            ylabel="MATH-500 correctness (%)",
            title="Log calls are not optimizer updates",
        )
        axes[1].set(
            xlabel="W&B rollout axis",
            ylabel="Mixed-source student center",
            title="Actual duplicate-x history",
        )
        axes[2].set(
            xlabel="Writes at x=0: snapshot, average",
            ylabel="Student center",
            title="Same field, different statistics",
        )
        finish(fig, axes, figures, "09-log-semantics")

    if reference_raw is not None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        refs = [
            json.loads((reference_raw / f"baseline-{rid}" / "wandb-metadata.json").read_text())
            for rid in ["6uorrk2k", "u87buult"]
        ]
        for col, benchmark in enumerate(BENCHMARKS):
            vals = [
                100 * ref["scalar_summary"][f"val-core/{benchmark}/acc/mean@16"] for ref in refs
            ]
            axes[col].bar(["Initial student", "Teacher"], vals, color=["#0072B2", "#CC79A7"])
            axes[col].set(title=benchmark, ylabel="mean@16 correctness (%)", ylim=(0, 100))
        fig.suptitle(
            "Historical reference ONLY: T=0.7, top-p=0.9, response budget=31744; 570 questions"
        )
        finish(fig, axes, figures, "10-reference-protocol")

    write_csv(
        tables / "inner.csv",
        [
            {"method": m, "k": r["outer_call"], "n": r["inner_step"], **r["metrics"]}
            for m, run in runs.items()
            for r in run["inner"]
        ],
    )
    write_csv(
        tables / "outer.csv",
        [
            {"method": m, "completed_outer": r["step"], **r["data"]}
            for m, run in runs.items()
            for r in run["training"]
        ],
    )
    write_csv(
        tables / "evaluation.csv",
        [
            {"method": m, "benchmark": b, **p}
            for m, run in runs.items()
            for b in BENCHMARKS
            for p in evaluation_points(run, b)
        ],
    )
    (tables / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = {
        str(run["directory"] / name): hashlib.sha256(
            (run["directory"] / name).read_bytes()
        ).hexdigest()
        for run in runs.values()
        for name in ["inner.jsonl", "outer.jsonl", "gpu.csv"]
    }
    (tables / "source-sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    manifest = [
        {
            "id": name,
            "scope": scope,
            "quantities": quantities,
            "axes": axes,
            "generated": (figures / f"{name}.png").exists(),
        }
        for name, scope, quantities, axes in FIGURE_SPECS
    ]
    (output / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main():
    """Build the canonical fixed review without modifying archived snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--bpo-raw", type=Path)
    parser.add_argument("--reference-raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render_review(
        args.raw, args.bpo_raw or args.raw, args.output, reference_raw=args.reference_raw
    )
    print(
        json.dumps(
            {
                m: {
                    k: v
                    for k, v in r.items()
                    if k
                    in ["completed_outer", "updates", "evaluation_outer", "ess_min", "gradient_max"]
                }
                for m, r in result["methods"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
