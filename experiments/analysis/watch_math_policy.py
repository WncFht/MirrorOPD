"""Refresh BPO comparison figures after each committed outer; never touch training."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from experiments.analysis.math_training_review import render_review


def snapshot_bpo(source, root):
    """Copy complete appended records into a private plotting snapshot."""
    destination = root / source.name
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"snapshot_unix": time.time(), "files": {}}
    # Read outer first: only its committed prefix determines accepted inner rows.
    for filename in ["outer.jsonl", "inner.jsonl", "gpu.csv"]:
        data = (source / filename).read_bytes()
        data = data.rsplit(b"\n", 1)[0] + b"\n"
        temporary = destination / (filename + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination / filename)
        manifest["files"][filename] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    (root / "snapshot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    """Poll local committed records and stop automatically when the trainer exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous_outer = -1
    while True:
        lines = (args.source / "outer.jsonl").read_text().splitlines(keepends=True)
        rows = [json.loads(s) for s in lines if s.endswith("\n")]
        completed = sum("timing_s/step" in r["data"] for r in rows)
        exit_path = args.source / "exit-code.txt"
        training_exited = exit_path.exists()
        if completed != previous_outer:
            manifest = snapshot_bpo(args.source, args.output / "raw")
            staging = args.output / "rendering"
            summary = render_review(
                args.baseline_raw, args.output / "raw", staging, include_wandb=False
            )
            for area in ["figures", "tables"]:
                target = args.output / area
                target.mkdir(exist_ok=True)
                for artifact in (staging / area).iterdir():
                    artifact.replace(target / artifact.name)
            (staging / "figure-manifest.json").replace(args.output / "figure-manifest.json")
            snapshot_rows = [
                json.loads(s)
                for s in (args.output / "raw" / args.source.name / "outer.jsonl")
                .read_text()
                .splitlines()
            ]
            completed = summary["methods"]["BPO"]["completed_outer"]
            state = {
                "watcher_pid": os.getpid(),
                "updated_unix": time.time(),
                "bpo_completed_outer": completed,
                "bpo_evaluation_outer": [
                    r["step"] for r in snapshot_rows if "evaluation/generated_examples" in r["data"]
                ],
                "training_exited": training_exited,
                "training_exit_code": int(exit_path.read_text()) if training_exited else None,
                "snapshot": manifest,
                "review_summary": summary,
            }
            temporary = args.output / "status.json.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(args.output / "status.json")
            print(
                json.dumps({"updated": completed, "training_exited": training_exited}), flush=True
            )
            previous_outer = completed
        if args.once or training_exited:
            if training_exited:
                state = json.loads((args.output / "status.json").read_text())
                state.update(
                    training_exited=True,
                    training_exit_code=int(exit_path.read_text()),
                    watcher_finished=True,
                )
                (args.output / "status.json").write_text(json.dumps(state, indent=2) + "\n")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
