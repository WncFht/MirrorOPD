"""Build complete manual-transfer packages without download receipts or destructive merges."""

import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from mirroropd.runtime.artifacts import atomic_json, file_digest
from mirroropd.runtime.provenance import runtime_versions, verify_backend

CODE_DIRECTORIES = {"src", "tests", "scripts", "hope", "docs", ".agents", ".github"}
CODE_FILES = {
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "AGENTS.md",
    "Makefile",
    "LICENSE",
    "requirements-dev.txt",
    "package.json",
    "package-lock.json",
    "ruff.toml",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".prettierignore",
    ".markdownlint-cli2.jsonc",
    ".markdownlint.jsonc",
    ".prettierrc",
    ".gitattributes",
}
OMITTED = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "checkpoints",
    "wandb-cache",
    "wandb-data",
}


def transfer_cache():
    """Keep staging and archive reconstruction on disk rather than /tmp."""
    cache = Path.home() / ".cache/mirroropd/transfers"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def write_bootstrap(output):
    """Ship the same CLI as a stdlib-only zipapp for the first offline unpack."""
    target = output / "bootstrap.pyz"
    package = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "__main__.py", "from mirroropd.cli import main\nraise SystemExit(main())\n"
        )
        for path in sorted(package.rglob("*")):
            if path.is_file() and (
                path.suffix in {".py", ".json", ".patch"} or path.name in {"LICENSE", "NOTICE"}
            ):
                archive.write(path, "mirroropd/" + path.relative_to(package).as_posix())
    return {"path": target.name, "bytes": target.stat().st_size, "sha256": file_digest(target)}


def git(root, *arguments):
    """Run local Git operations with explicit arguments and no shell expansion."""
    return subprocess.check_output(["/usr/bin/git", "-C", str(root), *arguments])  # noqa: S603 -- local source paths and fixed Git subcommands


def stage_code(source, backend, destination):
    """Copy current first-party bytes and reconstruct a self-contained patched backend."""
    source, backend = Path(source).resolve(), Path(backend).resolve()
    integration = source / "src/mirroropd/integrations/verl"
    manifest = json.loads((integration / "backend.json").read_text())
    patch = integration / manifest["patch"]
    verified = verify_backend(backend, patch, manifest["base_commit"])
    files = set()
    for name in CODE_DIRECTORIES | CODE_FILES:
        root = source / name
        if root.is_dir():
            files.update(path for path in root.rglob("*") if path.is_file())
        elif root.is_file():
            files.add(root)
    for original in sorted(files):
        path = original.relative_to(source)
        if any(part in OMITTED for part in path.parts) or any(
            part.startswith(".env") or part == ".netrc" for part in path.parts
        ):
            continue
        if original.is_symlink():
            raise ValueError(f"Code package requires regular source files: {path}")
        target = destination / "MirrorOPD" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    target = destination / "MirrorOPD/upstream/TA-OPD-v2"
    target.parent.mkdir(parents=True, exist_ok=True)
    # A local clone copies objects, not the source repository's config, credentials or hooks.
    git(source, "clone", "--local", "--no-hardlinks", "--no-checkout", str(backend), str(target))
    git(target, "checkout", "--detach", manifest["base_commit"])
    git(target, "apply", str(patch))
    git(target, "remote", "remove", "origin")
    verify_backend(target, patch, manifest["base_commit"])
    atomic_json(destination / "MirrorOPD/environment.json", runtime_versions())
    return {
        "backend": verified,
        "excluded": ["models", "datasets", "dependency wheels", "research evidence"],
    }


def stage_run(run, kind, destination):
    """Snapshot whole evidence files; never silently drop large validation or event files."""
    run = Path(run).expanduser().resolve()
    if not (run / "state.json").is_file():
        raise ValueError(f"Run directory needs state.json: {run}")
    states = {}
    for state in [run / "state.json", *sorted(run.glob("*/state.json"))]:
        states[state.relative_to(run).as_posix()] = json.loads(state.read_text())
    copied = 0
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run)
        if (
            any(part in OMITTED for part in relative.parts)
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        if kind == "wandb" and "wandb" not in relative.parts and path.name != "wandb.json":
            continue
        if path.name.startswith(".env") or path.name == ".netrc":
            continue
        target = destination / "runs" / run.name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    if not copied:
        raise ValueError(f"No {kind} files to package under {run}")
    return {
        "states_at_start": states,
        "live_snapshot": any(
            value["state"] not in {"completed", "prepared", "failed", "interrupted"}
            for value in states.values()
        ),
        "excluded": sorted(OMITTED),
    }


def pack_transfer(kind, *, output, backend=None, run=None, chunk_bytes=10 * 1024**2, source=None):
    """Create a new package directory containing all parts and a checksum manifest."""
    if chunk_bytes < 1:
        raise ValueError("chunk-bytes must be positive")
    if kind not in {"code", "results", "wandb"}:
        raise ValueError("Package kind must be code, results or wandb")
    if (kind == "code" and (backend is None or run is not None)) or (
        kind != "code" and (run is None or backend is not None)
    ):
        raise ValueError("Code packages require --backend; results/wandb packages require --run")
    output = Path(output).expanduser().resolve()
    if run and output.is_relative_to(Path(run).expanduser().resolve()):
        raise ValueError("Package output must be outside the run being copied")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=transfer_cache()) as temporary:
        stage = Path(temporary) / "files"
        stage.mkdir()
        details = (
            stage_code(source or Path(__file__).resolve().parents[3], backend, stage)
            if kind == "code"
            else stage_run(run, kind, stage)
        )
        archive = Path(temporary) / "payload.tar.gz"
        records = []
        with tarfile.open(archive, "w:gz") as stream:
            for path in sorted(stage.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Package contains an unsupported symlink: {path}")
                if not path.is_file():
                    continue
                relative = path.relative_to(stage).as_posix()
                records.append(
                    {"path": relative, "bytes": path.stat().st_size, "sha256": file_digest(path)}
                )
                stream.add(path, arcname=relative, recursive=False)
        parts = []
        with archive.open("rb") as stream:
            index = 0
            while block := stream.read(chunk_bytes):
                part = output / f"payload.tar.gz.part.{index:04d}"
                part.write_bytes(block)
                parts.append({"path": part.name, "bytes": len(block), "sha256": file_digest(part)})
                index += 1
        manifest = {
            "schema_version": 1,
            "kind": kind,
            "parts": parts,
            "files": records,
            "details": details,
            "bootstrap": write_bootstrap(output),
        }
        atomic_json(output / "manifest.json", manifest)
    return {"manifest": str(output / "manifest.json"), "parts": len(parts), "files": len(records)}


def safe_member(name):
    """Allow only canonical relative paths beneath a new extraction directory."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.as_posix() != name:
        raise ValueError(f"Unsafe package path: {name}")
    return path


def unpack_transfer(manifest_path, output):
    """Verify every part and file, then publish into a new directory without deleting inputs."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1 or not manifest["parts"] or not manifest["files"]:
        raise ValueError("Unsupported or empty transfer manifest")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(
            "Unpack requires a new output directory; existing runs are never overwritten"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=transfer_cache()) as temporary:
        archive = Path(temporary) / "payload.tar.gz"
        seen_parts = set()
        with archive.open("wb") as target:
            for record in manifest["parts"]:
                name = record["path"]
                if safe_member(name).name != name or name in seen_parts:
                    raise ValueError("Invalid or duplicated transfer part name")
                seen_parts.add(name)
                part = manifest_path.parent / name
                if part.stat().st_size != record["bytes"] or file_digest(part) != record["sha256"]:
                    raise ValueError(f"Transfer part checksum mismatch: {name}")
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target)
        expected = {record["path"]: record for record in manifest["files"]}
        if len(expected) != len(manifest["files"]):
            raise ValueError("Duplicated file in transfer manifest")
        for name in expected:
            safe_member(name)
        with tempfile.TemporaryDirectory(prefix=".mirror-unpack-", dir=output.parent) as stage:
            seen = set()
            with tarfile.open(archive, "r:gz") as stream:
                for member in stream:
                    safe_member(member.name)
                    if not member.isfile() or member.name not in expected or member.name in seen:
                        raise ValueError(
                            f"Unexpected, duplicated or nonregular package member: {member.name}"
                        )
                    record = expected[member.name]
                    if member.size != record["bytes"]:
                        raise ValueError(f"Package file size mismatch: {member.name}")
                    target = Path(stage) / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
                    target.chmod(member.mode & 0o777)
                    if file_digest(target) != record["sha256"]:
                        raise ValueError(f"Package file checksum mismatch: {member.name}")
                    seen.add(member.name)
            if seen != set(expected):
                raise ValueError("Package is missing files from its manifest")
            # Reserve publication rather than merging with another transfer or an active run.
            output.mkdir()
            for path in Path(stage).iterdir():
                shutil.move(str(path), output / path.name)
    return {
        "directory": str(output),
        "kind": manifest["kind"],
        "files": len(expected),
        "details": manifest["details"],
    }
