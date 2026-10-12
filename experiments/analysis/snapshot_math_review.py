"""Freeze an ongoing math run and optionally fetch its complete W&B evidence."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    """Preserve complete appended records and exact source bytes without overwrites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    destination = args.output / args.source.name
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"snapshot_unix": time.time(), "source": str(args.source), "files": {}}
    names = [
        "inner.jsonl",
        "outer.jsonl",
        "gpu.csv",
        "host-memory.jsonl",
        "resolved-config.yaml",
        "setting.json",
        "source-commit.txt",
        "source-provenance.json",
        "model-data-audit.json",
        "wandb-run.json",
    ]
    for name in names:
        data = (args.source / name).read_bytes()
        if name.endswith((".jsonl", ".csv")):
            data = data.rsplit(b"\n", 1)[0] + b"\n"
        (destination / name).write_bytes(data)
    if args.wandb:
        import wandb

        reference = json.loads((destination / "wandb-run.json").read_text())
        run = wandb.Api().run(f"{reference['entity']}/{reference['project']}/{reference['id']}")
        rows = list(run.scan_history(page_size=1000))
        with (destination / "wandb-history.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        metadata = {
            "fetched_unix": time.time(),
            "state": run.state,
            "url": run.url,
            "config": dict(run.config),
            "summary": dict(run.summary),
            "history_rows": len(rows),
        }
        (destination / "wandb-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for path in sorted(destination.iterdir()):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "snapshot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "snapshot_unix": manifest["snapshot_unix"],
                "files": len(manifest["files"]),
                "output": str(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
