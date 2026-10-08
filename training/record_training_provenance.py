"""Record deployed source hashes and runtime versions without collecting environment secrets."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hashes = {}
    for label, root, subdirs in (
        ("first_party", args.repo, ("src", "training")),
        ("framework", args.upstream / "verl", ("verl",)),
    ):
        for subdir in subdirs:
            for path in sorted((root / subdir).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".sh", ".json", ".yaml"):
                    hashes[f"{label}/{path.relative_to(root)}"] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
    versions = {}
    for package in ("torch", "transformers", "vllm", "ray", "verl", "datasets", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source checkout (not installed as distribution)"
    args.output.write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "versions": versions,
                "source_sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
