"""Restore versioned compressed research data, checking the recorded SHA-256."""

import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data-manifest.json").read_text())
    for item in manifest["files"]:
        destination = root / item["path"]
        archive = root / item["archive"]
        if destination.exists():
            if digest(destination) != item["sha256"]:
                raise ValueError(f"Local data differs; preserved without overwrite: {destination}")
            print(f"Verified existing {item['path']}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                with gzip.open(archive, "rb") as stream:
                    shutil.copyfileobj(stream, output)
                output.flush()
                if digest(temporary) != item["sha256"]:
                    raise ValueError(f"Compressed data checksum mismatch: {archive}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Restored {item['path']}")


if __name__ == "__main__":
    main()
