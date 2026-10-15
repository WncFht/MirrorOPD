"""Content-addressed model identities and immutable runtime source snapshots."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from mirroropd.identity import content_id
from mirroropd.runtime.artifacts import atomic_json, file_digest


class FingerprintCache:
    """Persist byte fingerprints; never infer content equality from cached stat fields."""

    def __init__(self, path):
        self.path = Path(path)
        self.values = json.loads(self.path.read_text()) if self.path.exists() else {}

    def digest(self, path):
        path = Path(path).resolve()
        before = path.stat()
        identity = [
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ]
        digest = file_digest(path)
        after = path.stat()
        after_identity = [
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ]
        if identity != after_identity:
            raise ValueError(f"Input artifact changed while being hashed: {path}")
        self.values[str(path)] = {"stat": identity, "sha256": digest}
        return digest

    def save(self):
        atomic_json(self.path, self.values)


def model_artifact(path, cache, *, trust_remote_code):
    path = Path(path).expanduser().resolve()
    if not (path / "config.json").is_file():
        raise ValueError(f"Local Hugging Face model artifact needs config.json: {path}")
    patterns = [
        "*.safetensors",
        "pytorch_model*.bin",
        "config.json",
        "generation_config.json",
        "tokenizer*",
        "vocab*",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "spiece.model",
    ]
    if trust_remote_code:
        patterns.append("*.py")
    files = sorted({file for pattern in patterns for file in path.glob(pattern) if file.is_file()})
    if not any(file.suffix in {".safetensors", ".bin"} for file in files):
        raise ValueError(f"No model weight files in {path}")
    records = [
        {
            "path": str(file.relative_to(path)),
            "bytes": file.stat().st_size,
            "sha256": cache.digest(file),
        }
        for file in files
    ]
    return {"path": str(path), "identity": content_id(records), "files": records}


def runtime_versions():
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in (
        "torch",
        "transformers",
        "vllm",
        "ray",
        "tensordict",
        "numpy",
        "pandas",
        "pyarrow",
        "omegaconf",
        "hydra-core",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot_sources(package_root, backend_root, destination):
    package_root, backend_root, destination = (
        Path(package_root),
        Path(backend_root),
        Path(destination),
    )
    records = []
    for component, root in (("mirroropd", package_root), ("verl", backend_root / "verl/verl")):
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or (
                    path.suffix not in {".py", ".json", ".yaml", ".yml", ".jinja"}
                    and path.name not in {"LICENSE", "NOTICE", "version"}
                )
            ):
                continue
            target = destination / component / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            records.append(
                {
                    "component": component,
                    "path": str(path.relative_to(root)),
                    "sha256": file_digest(target),
                }
            )
    source_id = content_id(records)
    artifact = {
        "schema_version": 2,
        "source_id": source_id,
        "files": records,
        "runtime": runtime_versions(),
    }
    atomic_json(destination / "manifest.json", artifact)
    return artifact


def verify_snapshot(destination):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    for item in manifest["files"]:
        if file_digest(destination / item["component"] / item["path"]) != item["sha256"]:
            raise ValueError("Frozen runtime source changed")
    if content_id(manifest["files"]) != manifest["source_id"]:
        raise ValueError("Runtime source manifest changed")
    return manifest


def verify_backend(root, patch_path, base_commit):
    root = Path(root)
    head = subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if head != base_commit:
        raise ValueError("Backend checkout is not at the pinned base commit")
    actual = subprocess.check_output(["/usr/bin/git", "diff", "--binary", base_commit], cwd=root)  # noqa: S603 -- pinned commit from the repository manifest
    expected = Path(patch_path).read_bytes()
    if actual != expected:
        raise ValueError(
            "Backend working tree does not exactly match the versioned integration patch"
        )
    untracked = subprocess.check_output(
        ["/usr/bin/git", "ls-files", "--others", "--exclude-standard", "--", "verl"],
        cwd=root,
        text=True,
    )
    if untracked.strip():
        raise ValueError("Backend contains unversioned runtime source")
    return {"base_commit": base_commit, "patch_sha256": file_digest(patch_path)}
