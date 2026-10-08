"""Report a completed budget probe without claiming inner convergence.

Semantic ledger: each row is a pre-update measurement on a full rollout
batch; each line is one outer call. Tokens within a trajectory are dependent.
Geometry ledger: x is inner update index; y is ESS fraction, log-ratio RMS,
Tail-aware anchor KL in nats, or clipping-before gradient norm. No error bands
are shown because this is a single trajectory of optimization, not seed repeats.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / ".agents/skills/figure-viz/assets/mpl"))
import fvstyle as fv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.run / "exit-code.txt").read_text().strip() != "0":
        raise ValueError("Probe did not complete successfully")
    rows = [json.loads(x) for x in (args.run / "inner.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("Missing inner measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = (
        ("mirror/importance_ess_fraction", "Token IS ESS fraction"),
        ("mirror/sampled_log_ratio/rms", "Sampled log-ratio RMS"),
        ("mirror/student_center/mean", "Anchor KL, Tail-aware (nats)"),
        ("grad_norm", "Gradient norm before clipping"),
    )
    fv.apply_rc("paper")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    outer_calls = sorted({r["outer_call"] for r in rows})
    for ax, (field, label) in zip(axes.flat, fields, strict=True):
        for outer in outer_calls:
            selected = [r for r in rows if r["outer_call"] == outer]
            ax.plot(
                [r["inner_step"] for r in selected],
                [r["metrics"][field] for r in selected],
                label=f"Outer {outer}",
                color=fv.series_colors("paper", len(outer_calls))[outer_calls.index(outer)],
            )
        ax.set_xlabel("Pre-update inner index")
        ax.set_ylabel(label)
        ax.legend()
    audit = fv.audit(fig)
    (args.output / "figure-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fig.savefig(args.output / "inner-dynamics.png", dpi=180)
    fig.savefig(args.output / "inner-dynamics.pdf")
    plt.close(fig)
    with (args.run / "gpu.csv").open() as stream:
        gpu = list(csv.DictReader(stream, skipinitialspace=True))
    memory_key = next(k for k in gpu[0] if k.startswith("memory.used"))
    summary = {
        "run": args.run.name,
        "inner_updates": len(rows),
        "outer_calls": len(outer_calls),
        "gpu_sampled_peak_mib": max(float(r[memory_key].split()[0]) for r in gpu),
        "minimum_ess_fraction": min(r["metrics"][fields[0][0]] for r in rows),
        "maximum_importance_ratio": max(r["metrics"]["mirror/importance_ratio/max"] for r in rows),
        "first_pre_update": rows[0],
        "last_pre_update": rows[-1],
        "limitation": "No post-final-update oracle; diagnostics do not prove convergence.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if "pre_update" not in k}, indent=2))


if __name__ == "__main__":
    main()
