"""Aggregate the controlled eta scan, reusing the fixed-prefix pilot statistics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pythia_mdpo_report import read_jsonl, write_csv

RUNS = (
    ("0.03", "pythia-mdpo-eta-0p03"),
    ("0.1", "pythia-mdpo-low-lr"),
    ("0.3", "pythia-mdpo-eta-0p3"),
    ("1", "pythia-mdpo-eta-1"),
    ("infinity", "pythia-mdpo-eta-inf"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/training"))
    parser.add_argument("--output", type=Path, default=Path("results/training/pythia-eta-scan"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = []
    dynamics_rows = []
    for eta, name in RUNS:
        root = args.results / name
        overrides = (root / "overrides.txt").read_text().splitlines()
        assert "actor_rollout_ref.actor.optim.lr=1e-6" in overrides
        expected_eta = ".inf" if eta == "infinity" else eta
        assert f"distillation.distillation_loss.mirror_eta={expected_eta}" in overrides
        with (root / "checkpoint-metrics.csv").open() as stream:
            checkpoint_rows.extend({"eta": eta, **row} for row in csv.DictReader(stream))
        for n in range(4):
            records = [r for r in read_jsonl(root / "inner.jsonl") if r["inner_step"] == n]
            assert len(records) == 32
            weights = [r["metrics"]["mirror/valid_tokens"] for r in records]
            row = {"eta": eta, "inner_step": n}
            for key in (
                "mirror/sampled_log_ratio/rms",
                "mirror/importance_ess_fraction",
                "mirror/student_center/mean",
                "mirror/score_coefficient/rms",
                "grad_norm",
            ):
                values = np.array([r["metrics"][key] for r in records])
                row[key] = float(np.average(values, weights=weights))
                row[key + "/max"] = float(values.max())
            dynamics_rows.append(row)
    write_csv(args.output / "checkpoint-comparison.csv", checkpoint_rows)
    write_csv(args.output / "inner-comparison.csv", dynamics_rows)
    (args.output / "setting.json").write_text(
        json.dumps(
            {
                "runs": dict(RUNS),
                "lr": 1e-6,
                "eta_infinity": "zero anchor-KL coefficient; frozen reward retained",
                "validation_unit": "64 held-out texts; 8128 identical prefixes; 4096 NLL tokens",
                "training_seeds": 1,
                "bootstrap_unit": "text, paired to initialization",
            },
            indent=2,
        )
        + "\n"
    )
    print("Verified fixed LR and eta overrides; wrote checkpoint and inner comparisons")


if __name__ == "__main__":
    main()
