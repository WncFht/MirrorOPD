"""Audit archived One-Shot and MirrorOPD histories without online credentials."""

import csv
import gzip
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path


def main():
    """Recompute all inner records, source hashes, and W&B archive parity."""
    root = Path(__file__).resolve().parents[2]
    output = root / "results/oneshot-inner-audit"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "data-manifest.json").read_text())["files"]
    report = {"mirror": {}, "oneshot": {}}
    table = []
    for method in ("mdpo", "bpo"):
        name = f"dapo-deepseek-{method}-o50-j16"
        folder = root / "data/training/math-4gpu-final-20261012" / name
        with gzip.open(folder / "inner.jsonl.gz", "rt") as handle:
            records = [json.loads(line) for line in handle]
        assert len(records) == 800
        assert {(r["outer_call"], r["inner_step"]) for r in records} == {
            (o, i) for o in range(50) for i in range(16)
        }
        assert all(r["measurement"] == "pre-update" for r in records)
        archive_hashes = {}
        for file in ("inner.jsonl", "source-provenance.json", "source-commit.txt"):
            archive = folder / (file + ".gz")
            with gzip.open(archive, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            entry = next(
                item for item in manifest if item["archive"] == str(archive.relative_to(root))
            )
            assert digest == entry["sha256"]
            archive_hashes[file] = digest
        with gzip.open(folder / "source-commit.txt.gz", "rt") as handle:
            commit = handle.read().strip()
        with gzip.open(folder / "source-provenance.json.gz", "rt") as handle:
            provenance = json.load(handle)
        source_matches = {}
        for file in (
            "training/losses.py",
            "training/cache.py",
            "training/metrics.py",
            "training/tracking.py",
            "integrations/verl/fsdp.py",
        ):
            blob = subprocess.check_output(  # noqa: S603 -- repository provenance, no shell
                [shutil.which("git"), "-C", str(root), "show", f"{commit}:src/mirroropd/{file}"]
            )
            source_matches[file] = (
                hashlib.sha256(blob).hexdigest()
                == provenance["source_sha256"]["first_party/src/mirroropd/" + file]
            )
        assert all(source_matches.values())
        online = (
            root
            / (
                "data/training/math-policy-review-20261012"
                if method == "bpo"
                else "data/training/math-policy-audit-20261012"
            )
            / name
            / "wandb-history.jsonl.gz"
        )
        with gzip.open(online, "rt") as handle:
            history = [json.loads(line) for line in handle]
        online_inner = {
            int(r["00_axes/inner_update"]): r
            for r in history
            if r.get("04_kl_centers/inner_student_center/mean") is not None
        }
        errors = []
        for index, record in enumerate(records):
            if index in online_inner:
                errors.append(
                    abs(
                        record["metrics"]["mirror/student_center/mean"]
                        - online_inner[index]["04_kl_centers/inner_student_center/mean"]
                    )
                )
            metrics = record["metrics"]
            table.append(
                {
                    "method": method,
                    "outer_call": record["outer_call"],
                    "inner_step": record["inner_step"],
                    "updates_before_measurement": index,
                    "anchor_student_tail_kl": metrics["mirror/student_center/mean"],
                    "anchor_teacher_tail_kl": metrics["mirror/teacher_center/mean"],
                    "frozen_reward_mean": metrics["mirror/sampled_reward/mean"],
                    "current_signed_gap_on_anchor_actions": metrics["mirror/sampled_log_ratio/mean"]
                    - metrics["mirror/sampled_reward/mean"],
                    "grad_norm": metrics["grad_norm"],
                    "ess_fraction": metrics["mirror/importance_ess_fraction"],
                }
            )
        frozen = all(
            all(
                len({r["metrics"][key] for r in records if r["outer_call"] == o}) == 1
                for o in range(50)
            )
            for key in (
                "mirror/teacher_center/mean",
                "mirror/sampled_reward/mean",
                "mirror/anchor_head_mass/mean",
            )
        )
        report["mirror"][method] = {
            "records": len(records),
            "commit": commit,
            "archive_sha256": archive_hashes,
            "source_matches": source_matches,
            "frozen_cache_fields_constant": frozen,
            "archived_online_inner_rows": len(online_inner),
            "online_parity_rows": len(errors),
            "online_student_center_max_error": max(errors),
            "inner_student_center_statistics": {
                str(i): {"min": min(values), "mean": statistics.mean(values), "max": max(values)}
                for i in (0, 1, 2, 15)
                for values in [
                    [
                        r["metrics"]["mirror/student_center/mean"]
                        for r in records
                        if r["inner_step"] == i
                    ]
                ]
            },
        }
    for run_id in ("ceejzu26", "5aub1k4h", "mdxt04lm"):
        folder = root.parent / "MD-OPD/plots/runs"
        with (folder / f"{run_id}__history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        report["oneshot"][run_id] = {
            "rows": len(rows),
            "sha256": hashlib.sha256((folder / f"{run_id}__history.csv").read_bytes()).hexdigest(),
            "selected_steps": [
                {
                    key: row.get(key)
                    for key in (
                        "_step",
                        "opd/adv_abs_mean",
                        "opd/adv_mean",
                        "opd/token_reward_mean",
                        "opd/top_k_overlap_ratio",
                    )
                }
                for row in rows
                if int(float(row["_step"])) in (1, 30, 50, 100, 200, 279, 300)
            ],
        }
    with (output / "inner-history.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: {
                    "rows": value["records"],
                    "online_parity": value["online_parity_rows"],
                    "online_max_error": value["online_student_center_max_error"],
                }
                for key, value in report["mirror"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
