"""Archive and aggregate actual production-estimator validation records."""

import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def cpu_gradient_records():
    import torch

    from mirroropd.training.losses import full_vocab_mirror_loss

    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64)
    z = torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    rows = []
    for method in ("bpo", "mdpo"):
        logp = z.log_softmax(-1)
        raw = (0.7 * q.sqrt() + 0.3 * teacher.sqrt()).square()
        target = (raw / raw.sum()).log()
        if method == "bpo":
            residual = logp - target
            residual = residual - (q * residual).sum()
            reference = (q * residual.square()).sum() / 0.6
        else:
            reference = (logp.exp() * (logp - target)).sum() / 0.3
        expected = torch.autograd.grad(reference, z)[0]
        loss = full_vocab_mirror_loss(z, q.log(), teacher.log(), method, 0.3, 0.5)
        actual = torch.autograd.grad(loss, z)[0]
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)
        rows.append(
            {
                "method": method,
                "production": actual.tolist(),
                "reference": expected.tolist(),
                "max_absolute_error": float((actual - expected).abs().max()),
                "kappa": 0.5,
                "eta": 0.3,
            }
        )
    return rows


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "runs/power-implementation-20261015"
    data = root / "data/power-implementation-20261015"
    output = root / "results/power-implementation-20261015"
    data.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    records = []
    provenance = []
    training = []
    native = []
    archives = []
    for directory in sorted(source.iterdir()):
        if not directory.is_dir() or not (directory / "completion.jsonl").exists():
            continue
        completion = read_rows(directory / "completion.jsonl")[-1]
        if completion.get("state") != "completed":
            continue
        destination = data / directory.name
        destination.mkdir(exist_ok=True)
        for path in sorted(directory.iterdir()):
            if path.suffix not in (".json", ".jsonl", ".patch", ".yaml"):
                continue
            if path.suffix == ".patch":
                raw = path.read_bytes()
                target = destination / (path.name + ".gz")
                with (
                    target.open("wb") as stream,
                    gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed,
                ):
                    compressed.write(raw)
                if gzip.decompress(target.read_bytes()) != raw:
                    raise RuntimeError("Backend patch archive failed its bytewise roundtrip")
                (destination / path.name).unlink(missing_ok=True)
                archives.append(
                    {
                        "path": f"runs/power-implementation-20261015/backend-patches/{directory.name}.patch",
                        "archive": str(target.relative_to(root)),
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                provenance.append(
                    {
                        "path": str(target.relative_to(root)),
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                )
                continue
            target = destination / path.name
            shutil.copy2(path, target)
            provenance.append(
                {
                    "path": str(target.relative_to(root)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        attempts = sorted((source / "attempts").glob(directory.name + "-*"))
        snapshot = directory / "source"
        if attempts:
            snapshot = attempts[-1] / "source"
            for relative in ("config.json", "source/source-sha256.json", "exit.json"):
                path = attempts[-1] / relative
                if path.exists():
                    target = destination / path.name
                    if path.name == "source-sha256.json":
                        hashes = json.loads(path.read_text())
                        target.write_text(
                            json.dumps(
                                {
                                    "files": [
                                        {"path": name, "sha256": digest}
                                        for name, digest in sorted(hashes.items())
                                    ]
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                    else:
                        shutil.copy2(path, target)
                    provenance.append(
                        {
                            "path": str(target.relative_to(root)),
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        }
                    )
            for path in (attempts[-1] / "owners").glob("rank-*.json"):
                target = destination / ("gpu-owner-" + path.name)
                shutil.copy2(path, target)
                provenance.append(
                    {
                        "path": str(target.relative_to(root)),
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                )
        if snapshot.exists():
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as bundle:
                for path in sorted(snapshot.rglob("*")):
                    if path.is_file() and path.suffix in (".py", ".json"):
                        raw = path.read_bytes()
                        info = tarfile.TarInfo(str(path.relative_to(snapshot)))
                        info.size, info.mode = len(raw), 0o644
                        bundle.addfile(info, io.BytesIO(raw))
            raw = buffer.getvalue()
            target = destination / "source.tar.gz"
            with (
                target.open("wb") as stream,
                gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed,
            ):
                compressed.write(raw)
            if gzip.decompress(target.read_bytes()) != raw:
                raise RuntimeError("Source archive failed its bytewise roundtrip")
            archives.append(
                {
                    "path": f"runs/power-implementation-20261015/source-archives/{directory.name}.tar",
                    "archive": str(target.relative_to(root)),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            provenance.append(
                {
                    "path": str(target.relative_to(root)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        for path in directory.glob("audit-rank-*.json"):
            records.extend(json.loads(path.read_text()))
        if completion["phase"] == "train":
            measurement = read_rows(directory / "measurements.jsonl")
            updates = read_rows(directory / "updates.jsonl")
            if len(updates) != completion["updates"]:
                raise RuntimeError(f"Incomplete optimizer-update ledger: {directory.name}")
            if [row["update"] for row in updates] != list(range(1, completion["updates"] + 1)):
                raise RuntimeError(f"Duplicate or missing optimizer updates: {directory.name}")
            training.append(
                {
                    "id": directory.name,
                    "updates": len(updates),
                    "seconds": completion["seconds"],
                    "initial": measurement[0],
                    "final": measurement[-1],
                    "peak_allocated_gib": max(row["peak_allocated_gib"] for row in updates),
                    "gradient_checks": read_rows(directory / "parameter-gradient-check.jsonl"),
                    "anchor_refresh": read_rows(directory / "anchor-refresh.jsonl"),
                    "checkpoint_roundtrip": read_rows(directory / "checkpoint-roundtrip.jsonl"),
                }
            )
        elif completion["phase"] == "verl":
            rows = read_rows(directory / "inner.jsonl")
            console = (directory / "console.log").read_text(errors="replace")
            versions = [
                int(value)
                for value in re.findall(r"mirror/rollout_weight_version:([0-9]+)", console)
            ]
            if versions != [0, 1]:
                raise RuntimeError("Native rollout weight versions are not the two fresh policies")
            probability_differences = [
                float(value)
                for value in re.findall(r"training/rollout_probs_diff_max:([0-9.eE+-]+)", console)
            ]
            rollout_audit = {
                "weight_versions": versions,
                "rollout_actor_probability_max_differences": probability_differences,
                "scope": "Version freshness and numerical engine differences are separate checks",
            }
            target = destination / "rollout-validation.json"
            target.write_text(json.dumps(rollout_audit, indent=2) + "\n")
            provenance.append(
                {
                    "path": str(target.relative_to(root)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
            if len(rows) != 4:
                raise RuntimeError("Native verl optimizer ledger is incomplete")
            for row in rows:
                metrics = row["metrics"]
                if completion["estimator"] == "full_vocab":
                    if (
                        "mirror/full_loss/mean" not in metrics
                        or "mirror/importance_ratio/mean" in metrics
                    ):
                        raise RuntimeError(
                            "Native complete-action mode returned sampled diagnostics"
                        )
                elif (
                    "mirror/score_coefficient/mean" not in metrics
                    or "mirror/full_loss/mean" in metrics
                ):
                    raise RuntimeError("Native sampled mode did not execute its score path")
            windows = []
            for outer in (0, 1):
                part = [row for row in rows if row["outer_call"] == outer]
                if [row["inner_step"] for row in part] != [0, 1]:
                    raise RuntimeError("Native verl inner-step order is invalid")
                teachers = [row["metrics"]["mirror/teacher_center/mean"] for row in part]
                students = [row["metrics"]["mirror/student_center/mean"] for row in part]
                if teachers[0] != teachers[1] or abs(students[0]) > 1e-4:
                    raise RuntimeError("Native teacher freezing or fresh anchor check failed")
                windows.append(
                    {
                        "outer": outer,
                        "teacher_center": teachers[0],
                        "student_centers": students,
                        "valid_tokens": [row["metrics"]["mirror/valid_tokens"] for row in part],
                    }
                )
            native.append(
                {
                    "id": directory.name,
                    "completion": completion,
                    "windows": windows,
                    "rollout": rollout_audit,
                }
            )
    audit = []
    if records:
        for method, topj in sorted({(r["method"], r["topj"]) for r in records}):
            part = [r for r in records if r["method"] == method and r["topj"] == topj]
            weight = sum(r["prefixes"] for r in part)
            item = {
                "method": method,
                "topj": topj,
                "prefixes": weight,
                "responses": len(part),
                "full_gradient_max_error": max(r["full_gradient_max_error"] for r in part),
            }
            for field in ("teacher_center_mae", "student_center_mae", "score_coefficient_mae"):
                item[field] = sum(r[field] * r["prefixes"] for r in part) / weight
            for field in ("compact_cache_bytes", "full_cache_bytes"):
                item[field + "_per_prefix"] = sum(r[field] for r in part) / weight
            audit.append(item)
        with (output / "cache-audit.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audit[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(audit)
    result = {
        "date": "2026-10-15",
        "kappa": 0.5,
        "eta": 0.3,
        "physical_gpus": [0, 1, 2, 3],
        "audit": audit,
        "training": training,
        "native": native,
        "provenance": provenance,
        "scope": "Fixed-parameter correctness and estimator comparison; single t22 prompt; no task-score or optimal-hyperparameter claim",
        "cpu_gradients": cpu_gradient_records(),
        "inputs": [
            {"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
            for name in (
                "data/oneshot-effective-time-20261012/seed-42/manifest.json",
                "../models/manifest.json",
                "../MD-OPD/data/math/DAPO-oneshot/t22.parquet",
                "data/power-implementation-20261015/preregistration.json",
                "data/power-implementation-20261015/runtime.json",
                "data/power-implementation-20261015/model-verification.json",
            )
        ],
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    manifest_path = root / "data-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prefix = "runs/power-implementation-20261015/"
    manifest["files"] = [
        entry for entry in manifest["files"] if not entry["path"].startswith(prefix)
    ] + archives
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=4) + "\n")
    print(
        json.dumps({"audit_cells": len(audit), "completed_training_arms": len(training)}, indent=2)
    )


if __name__ == "__main__":
    main()
