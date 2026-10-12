"""Reproduce time-aligned math-policy audit from immutable JSONL snapshots."""

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

RUNS = {
    "OPD": "dapo-deepseek-opd-o50-j1",
    "MDPO": "dapo-deepseek-mdpo-o50-j16",
    "BPO": "dapo-deepseek-bpo-o50-j16",
}
COLORS = {"OPD": "#0072B2", "MDPO": "#D55E00", "BPO": "#009E73"}
BENCHMARKS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}
METRIC_LABELS = {
    "teacher_center/mean": r"$\overline{\widehat{\mathcal{K}}^{T,\mathrm{TA}}}$ (nats)",
    "student_center/mean": r"$\overline{\widehat{\mathcal{K}}^{S,\mathrm{TA}}}$ (nats)",
    "importance_ess_fraction": r"$f=\mathrm{ESS}/M_k$",
    "importance_ratio/std": r"$\sqrt{\overline{\omega^2}-\bar{\omega}^2}$",
    "importance_ratio/max": r"$\max_j\omega_j$",
    "importance_ratio/min": r"$\min_j\omega_j$",
    "importance_ratio/mean": r"$\bar{\omega}$",
    "sampled_log_ratio/rms": r"$\sqrt{\overline{U^2}}$ (nats)",
    "score_coefficient/rms": r"$\sqrt{\overline{c^2}}$",
    "score_coefficient/mean": r"$\bar{c}$",
    "student_entropy/mean": r"$\overline{H(p)}$ (nats)",
    "support_overlap/mean": r"$|\mathrm{Top}_J(p)\cap\mathcal{H}_k|/J$",
    "anchor_head_mass/mean": r"$\overline{\sum_{a\in\mathcal{H}_k}q_a}$",
    "student_head_mass/mean": r"$\overline{\sum_{a\in\mathcal{H}_k}p_a}$",
    "grad_norm": r"$\Vert g\Vert_2$ (before clipping)",
    "critic/score/mean": r"$100 B^{-1}\sum_i\mathrm{score}_i$ (%)",
    "response_length/mean": r"$B^{-1}\sum_i H_i$ (tokens)",
    "response_length/clip_ratio": r"$100 B^{-1}\sum_i\mathbf{1}\{H_i=7168\}$ (%)",
    "perf/total_num_tokens": r"$\sum_i(P_i+H_i)$ (tokens)",
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finish(fig, axes, output, name):
    for ax in np.asarray(axes).ravel():
        ax.grid(alpha=0.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def line(ax, x, y, method, **kwargs):
    ax.plot(x, y, label=method, color=COLORS[method], **kwargs)


def metrics(rows, key):
    return np.asarray([r["metrics"]["mirror/" + key] for r in rows])


def load_run(root, name):
    directory = root / name
    inner = read_rows(directory / "inner.jsonl")
    outer = read_rows(directory / "outer.jsonl")
    training = [r for r in outer if "timing_s/step" in r["data"]]
    completed = len(training)
    # Ongoing runs may have finished inner updates without an outer commit.
    inner = [r for r in inner if r["outer_call"] < completed]
    assert [r["step"] for r in training] == list(range(1, completed + 1))
    epochs = inner[0]["epochs"]
    assert [(r["outer_call"], r["inner_step"]) for r in inner] == [
        (k, n) for k in range(completed) for n in range(epochs)
    ]
    times = np.concatenate(([0.0], np.cumsum([r["data"]["timing_s/step"] for r in training])))
    evaluation = [r for r in outer if "evaluation/generated_examples" in r["data"]]
    assert all(r["data"]["evaluation/generated_examples"] == 570 for r in evaluation)
    groups = [inner[k * epochs : (k + 1) * epochs] for k in range(completed)]
    return {
        "inner": inner,
        "training": training,
        "evaluation": evaluation,
        "groups": groups,
        "first": [g[0] for g in groups],
        "last": [g[-1] for g in groups],
        "N": epochs,
        "completed": completed,
        "times": times,
        "name": name,
        "directory": directory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bpo-raw", type=Path, help="Newer BPO snapshot root; reuse completed runs"
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    roots = {m: args.bpo_raw if m == "BPO" and args.bpo_raw else args.raw for m in RUNS}
    runs = {m: load_run(roots[m], n) for m, n in RUNS.items()}
    summary = {}
    inner_csv, eval_csv, outer_csv = [], [], []
    for method, run in runs.items():
        inner = run["inner"]
        ess = metrics(inner, "importance_ess_fraction")
        mean = metrics(inner, "importance_ratio/mean")
        std = metrics(inner, "importance_ratio/std")
        np.testing.assert_allclose(ess, 1 / (1 + (std / mean) ** 2), atol=1e-12)
        grad = np.asarray([r["metrics"]["grad_norm"] for r in inner])
        summary[method] = {
            "run": run["name"],
            "completed_outer": run["completed"],
            "N": run["N"],
            "completed_updates": len(inner),
            "training_seconds": float(run["times"][-1]),
            "ess_min": float(ess.min()),
            "ess_mean": float(ess.mean()),
            "ratio_max": float(metrics(inner, "importance_ratio/max").max()),
            "ratio_min": float(metrics(inner, "importance_ratio/min").min()),
            "ratio_mean_range": [float(mean.min()), float(mean.max())],
            "U_rms_max": float(metrics(inner, "sampled_log_ratio/rms").max()),
            "grad_range": [float(grad.min()), float(grad.max())],
            "clip_fraction": float(np.mean(grad > 2)),
            "benchmark": {},
            "windows": {},
        }
        inner_csv.extend(
            {"method": method, "k": r["outer_call"], "n": r["inner_step"], **r["metrics"]}
            for r in inner
        )
        outer_csv.extend(
            {"method": method, "outer": r["step"], **r["data"]} for r in run["training"]
        )
        for benchmark, size in BENCHMARKS.items():
            points = []
            for r in run["evaluation"]:
                data, k = r["data"], r["step"]
                point = {
                    "method": method,
                    "benchmark": benchmark,
                    "outer": k,
                    "updates": k * run["N"],
                    "training_seconds": float(run["times"][k]),
                    "accuracy": data[f"val-core/{benchmark}/acc/mean@1"],
                    "correct": round(size * data[f"val-core/{benchmark}/acc/mean@1"]),
                    "budget_exhausted": data[f"val-aux/{benchmark}/budget_exhausted/mean@1"],
                    "tokens": data[f"val-aux/{benchmark}/response_tokens/mean@1"],
                }
                eval_csv.append(point)
                points.append(point)
            summary[method]["benchmark"][benchmark] = points
        for key in ["teacher_center/mean", "sampled_reward/mean", "student_entropy/mean"]:
            vals = metrics(run["first"], key)
            summary[method]["windows"][key] = {
                "first_up_to_10": float(vals[:10].mean()),
                "last_up_to_10": float(vals[-10:].mean()),
                "initial": float(vals[0]),
                "latest": float(vals[-1]),
            }

    # Three budgets: fresh rollout, optimizer updates, and measured training time.
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for col, benchmark in enumerate(BENCHMARKS):
        for method in runs:
            points = summary[method]["benchmark"][benchmark]
            for row, (xkey, label) in enumerate(
                [
                    ("outer", "Completed fresh rollouts k"),
                    ("updates", "Optimizer updates"),
                    ("training_seconds", "Training time excluding validation (min)"),
                ]
            ):
                scale = 60 if xkey == "training_seconds" else 1
                line(
                    axes[row, col],
                    [p[xkey] / scale for p in points],
                    [100 * p["accuracy"] for p in points],
                    method,
                    marker="o",
                    markersize=3,
                )
                axes[row, col].set(xlabel=label, ylabel="Greedy accuracy (%)", title=benchmark)
    finish(fig, axes, output, "01-quality-budgets")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for col, benchmark in enumerate(BENCHMARKS):
        for method in runs:
            points = summary[method]["benchmark"][benchmark]
            line(
                axes[col],
                [p["outer"] for p in points],
                [100 * (p["accuracy"] - points[0]["accuracy"]) for p in points],
                method,
                marker="o",
                markersize=3,
            )
        axes[col].axhline(0, color="gray", linewidth=0.8)
        axes[col].set(
            title=benchmark,
            xlabel="Completed fresh rollouts",
            ylabel="Change from own initial (pp)",
        )
    finish(fig, axes, output, "02-quality-deltas")

    keys = ["teacher_center/mean", "sampled_reward/mean", "student_entropy/mean"]
    titles = [
        r"Frozen $\widehat{\mathcal{K}}_k^T$ (nats)",
        r"Frozen $-\overline{r}_k$ (nats)",
        r"$\overline{H}(\pi_k)$ (nats)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for method, run in runs.items():
        for col, key in enumerate(keys):
            y = metrics(run["first"], key) * (-1 if col == 1 else 1)
            k = np.arange(run["completed"])
            line(axes[0, col], k, y, method)
            line(axes[1, col], run["times"][:-1] / 60, y, method)
            axes[0, col].set(xlabel="Pre-update frozen rollout k", ylabel=titles[col])
            axes[1, col].set(xlabel="Training time before rollout (min)", ylabel=titles[col])
    finish(fig, axes, output, "03-frozen-alignment")

    keys = [
        "importance_ess_fraction",
        "importance_ratio/std",
        "importance_ratio/max",
        "importance_ratio/min",
        "importance_ratio/mean",
        "sampled_log_ratio/rms",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for method, run in runs.items():
        for ax, key in zip(axes.flat, keys, strict=True):
            vals = np.asarray([metrics(g, key) for g in run["groups"]])
            x = np.arange(run["completed"])
            if key == "importance_ess_fraction":
                line(ax, x, 100 * (1 - vals.min(axis=1)), method)
                ax.set_ylabel(r"Worst within-outer $100(1-ESS/M)$ (%)")
            else:
                line(ax, x, vals[:, -1], method)
                ax.set_ylabel(METRIC_LABELS[key] + " (last pre-update)")
            ax.set_xlabel("Frozen rollout k")
            if key == "importance_ratio/min":
                ax.set_yscale("log")
    finish(fig, axes, output, "04-is-ess-by-outer")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for method, run in runs.items():
        for ax, key in zip(axes.flat, keys, strict=True):
            matrix = np.asarray([metrics(g, key) for g in run["groups"]])
            avg = matrix.mean(axis=0)
            line(ax, np.arange(run["N"]), avg, method, marker="o", markersize=3)
            ax.fill_between(
                np.arange(run["N"]),
                matrix.min(axis=0),
                matrix.max(axis=0),
                color=COLORS[method],
                alpha=0.13,
            )
            ax.set(xlabel="Within-outer pre-update n", ylabel=METRIC_LABELS[key])
    finish(fig, axes, output, "05-is-ess-inner-profile")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row, method in enumerate(["MDPO", "BPO"]):
        run = runs[method]
        for col, key in enumerate(["importance_ess_fraction", "importance_ratio/max"]):
            matrix = np.asarray([metrics(g, key) for g in run["groups"]])
            if col == 0:
                matrix = 100 * (1 - matrix)
            im = axes[row, col].imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
            axes[row, col].set(
                title=method + (": ESS deficit (%)" if col == 0 else ": max action ratio"),
                xlabel="Pre-update n",
                ylabel="Frozen rollout k",
            )
            fig.colorbar(im, ax=axes[row, col])
    finish(fig, axes, output, "06-is-ess-heatmaps")

    keys = [
        "student_center/mean",
        "sampled_log_ratio/rms",
        "score_coefficient/rms",
        "student_entropy/mean",
        "support_overlap/mean",
        "importance_ess_fraction",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for method, run in runs.items():
        first = run["groups"][0]
        for ax, key in zip(axes.flat, keys, strict=True):
            line(ax, np.arange(run["N"]), metrics(first, key), method, marker="o", markersize=3)
            ax.set(xlabel="First outer pre-update n", ylabel=METRIC_LABELS[key])
    finish(fig, axes, output, "07-first-outer")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for method, run in runs.items():
        x = np.arange(run["completed"])
        for col, key in enumerate(["grad_norm", "score_coefficient/rms", "score_coefficient/mean"]):
            matrix = np.asarray(
                [
                    [
                        r["metrics"]["grad_norm"]
                        if key == "grad_norm"
                        else r["metrics"]["mirror/" + key]
                        for r in g
                    ]
                    for g in run["groups"]
                ]
            )
            line(axes[0, col], x, matrix[:, 0], method)
            line(axes[1, col], x, matrix[:, -1], method)
            axes[0, col].set(xlabel="Frozen rollout k (first inner)", ylabel=METRIC_LABELS[key])
            axes[1, col].set(xlabel="Frozen rollout k (last pre-update)", ylabel=METRIC_LABELS[key])
    axes[0, 0].axhline(2, color="black", linestyle="--", label="clip=2")
    finish(fig, axes, output, "08-optimization")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col, benchmark in enumerate(BENCHMARKS):
        for method in runs:
            points = summary[method]["benchmark"][benchmark]
            for row, key in enumerate(["tokens", "budget_exhausted"]):
                line(
                    axes[row, col],
                    [p["outer"] for p in points],
                    [p[key] * (100 if row else 1) for p in points],
                    method,
                    marker="o",
                    markersize=3,
                )
                axes[row, col].set(
                    title=benchmark,
                    xlabel="Completed fresh rollouts",
                    ylabel="Budget exhausted (%)" if row else "Mean generated tokens",
                )
    finish(fig, axes, output, "09-eval-length-budget")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for method, run in runs.items():
        data = [r["data"] for r in run["training"]]
        for ax, key, scale in zip(
            axes.flat,
            [
                "critic/score/mean",
                "response_length/mean",
                "response_length/clip_ratio",
                "perf/total_num_tokens",
            ],
            [100, 1, 100, 1],
            strict=True,
        ):
            line(ax, np.arange(run["completed"]), [d[key] * scale for d in data], method)
            ax.set(xlabel="Rollout generated before updates (k)", ylabel=METRIC_LABELS[key])
    finish(fig, axes, output, "10-training-rollouts")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    keys = [
        "anchor_head_mass/mean",
        "student_head_mass/mean",
        "support_overlap/mean",
        "student_center/mean",
    ]
    for method, run in runs.items():
        for ax, key in zip(axes.flat, keys, strict=True):
            line(ax, np.arange(run["completed"]), metrics(run["last"], key), method)
            ax.set(xlabel="Frozen rollout k (last pre-update)", ylabel=METRIC_LABELS[key])
    finish(fig, axes, output, "11-support-centers")

    timing_keys = ["gen", "old_log_prob", "mirror_teacher_cache", "update_actor", "update_weights"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bottom = np.zeros(3)
    for key in [*timing_keys, "other"]:
        vals = []
        for method, run in runs.items():
            rows = [r["data"] for r in run["training"]]
            if key == "other":
                values = [
                    d["timing_s/step"] - sum(d["timing_s/" + k] for k in timing_keys) for d in rows
                ]
            else:
                values = [d["timing_s/" + key] for d in rows]
            value = float(np.mean(values))
            summary[method].setdefault("mean_seconds", {})[key] = value
            vals.append(value)
        axes[0].bar(list(runs), vals, bottom=bottom, label=key)
        bottom += vals
    axes[0].set(
        ylabel="Mean seconds / outer", title="Exclusive top-level timing; validation excluded"
    )
    for method, run in runs.items():
        line(axes[1], np.arange(1, run["completed"] + 1), run["times"][1:] / 60, method)
    axes[1].set(xlabel="Completed outer", ylabel="Cumulative training minutes")
    finish(fig, axes, output, "12-timing")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col, (method, run) in enumerate(runs.items()):
        with (run["directory"] / "gpu.csv").open() as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        # A common fixed offset only defines elapsed time, not a UTC claim.
        start = datetime.strptime(rows[0]["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
            tzinfo=timezone.utc
        )
        for gpu in range(4):
            values = [r for r in rows if int(r["index"]) == gpu]
            time = [
                (
                    datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
                        tzinfo=timezone.utc
                    )
                    - start
                ).total_seconds()
                / 60
                for r in values
            ]
            for ax, key, scale in [
                (axes[0, col], "memory.used [MiB]", 1024),
                (axes[1, col], "utilization.gpu [%]", 1),
            ]:
                y = [float(r[key].split()[0]) / scale for r in values]
                ax.plot(time, y, linewidth=0.6, alpha=0.65, label=f"GPU {gpu}")
                ax.set(xlabel="Monitor elapsed wall time (min)", title=method)
        axes[0, col].set_ylabel("Device used memory (GiB)")
        axes[1, col].set_ylabel("GPU utilization (%)")
    finish(fig, axes, output, "13-gpu-walltime")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for method, run in runs.items():
        k = np.arange(run["completed"])
        snapshot = metrics(run["first"], "student_center/mean")
        aggregate = [r["data"]["actor/mirror/student_center/mean"] for r in run["training"]]
        line(axes[0], k, snapshot, method, linestyle="--")
        axes[0].plot(k, aggregate, color=COLORS[method], label=method + " outer inner-average")
        line(axes[1], k, metrics(run["first"], "teacher_center/mean"), method)
    axes[0].set(
        xlabel="Frozen rollout k",
        ylabel="Student center (nats)",
        title="Dashed: n=0; solid: aggregation over n",
    )
    axes[1].set(
        xlabel="Frozen rollout k", ylabel="Teacher center (nats)", title="Clean n=0 snapshot only"
    )
    finish(fig, axes, output, "14-snapshot-vs-aggregate")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for method, run in runs.items():
        history = read_rows(run["directory"] / "wandb-history.jsonl")
        quality_key = "01_quality/val-core/MATH-500/acc/mean@1"
        qrows = [r for r in history if r.get(quality_key) is not None]
        line(
            axes[0],
            [r["_step"] for r in qrows],
            [100 * r[quality_key] for r in qrows],
            method,
            marker="o",
            markersize=3,
        )
        center_key = "04_kl_centers/outer_student_center/mean"
        crows = [r for r in history if r.get(center_key) is not None]
        line(
            axes[1],
            [r["00_axes/outer_rollout_step"] for r in crows],
            [r[center_key] for r in crows],
            method,
            marker=".",
            markersize=3,
        )
        cfirst = [r for r in crows if r["00_axes/outer_rollout_step"] == 0]
        axes[2].plot(
            np.arange(len(cfirst)),
            [r[center_key] for r in cfirst],
            "o-",
            color=COLORS[method],
            label=method,
        )
    axes[0].set(
        xlabel="W&B internal _step (log calls)",
        ylabel="MATH-500 accuracy (%)",
        title="Misleading default axis",
    )
    axes[1].set(
        xlabel="W&B outer_rollout_step",
        ylabel="Student center",
        title="Actual duplicate-x online history",
    )
    axes[2].set(
        xlabel="Writes at rollout x=0: snapshot, aggregate",
        ylabel="Student center",
        title="Two online values at x=0",
    )
    finish(fig, axes, output, "15-wandb-collision")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for method, run in runs.items():
        for col, key in enumerate(
            ["teacher_center/mean", "sampled_reward/mean", "student_entropy/mean"]
        ):
            vals = metrics(run["first"], key) * (-1 if col == 1 else 1)
            line(axes[col], np.arange(run["completed"]) * run["N"], vals, method)
            axes[col].set(xlabel="Optimizer updates before frozen rollout", ylabel=titles[col])
    finish(fig, axes, output, "16-frozen-by-updates")

    write_csv(output / "inner.csv", inner_csv)
    write_csv(output / "evaluation.csv", eval_csv)
    write_csv(output / "outer.csv", outer_csv)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = {}
    for method, name in RUNS.items():
        for filename in ["inner.jsonl", "outer.jsonl", "gpu.csv"]:
            path = roots[method] / name / filename
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "source-sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(
        json.dumps(
            {
                m: {k: v for k, v in s.items() if k not in ["benchmark", "windows"]}
                for m, s in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
