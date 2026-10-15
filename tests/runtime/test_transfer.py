"""Manual-transfer round trips retain large evidence and reject incomplete or unsafe packages."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from mirroropd.runtime.artifacts import atomic_json, file_digest
from mirroropd.runtime.transfer import pack_transfer, unpack_transfer


def sample_run(tmp_path):
    run = tmp_path / "runs/test"
    run.mkdir(parents=True)
    atomic_json(run / "state.json", {"state": "completed"})
    (run / "validation").mkdir()
    (run / "validation/0.jsonl").write_bytes(bytes(range(256)) * 4096)
    (run / "checkpoints").mkdir()
    (run / "checkpoints/model.bin").write_bytes(b"excluded")
    (run / "wandb/offline-run-test").mkdir(parents=True)
    (run / "wandb/offline-run-test/run-test.wandb").write_bytes(b"offline")
    (run / ".env").write_text("excluded")
    return run


def test_results_and_wandb_packages_are_complete_repeatable_snapshots(tmp_path):
    run = sample_run(tmp_path)
    first = pack_transfer("results", run=run, output=tmp_path / "package", chunk_bytes=512)
    assert first["parts"] > 1
    restored = tmp_path / "restored"
    unpack_transfer(first["manifest"], restored)
    assert (restored / "runs/test/validation/0.jsonl").read_bytes() == (
        run / "validation/0.jsonl"
    ).read_bytes()
    assert not (restored / "runs/test/checkpoints").exists()
    assert not (restored / "runs/test/.env").exists()
    second = pack_transfer("wandb", run=run, output=tmp_path / "logs")
    unpack_transfer(second["manifest"], tmp_path / "logs-restored")
    assert (tmp_path / "logs-restored/runs/test/wandb/offline-run-test/run-test.wandb").is_file()
    assert Path(first["manifest"]).exists()
    assert Path(second["manifest"]).exists()
    with pytest.raises(ValueError, match="new output"):
        unpack_transfer(first["manifest"], restored)


def test_missing_or_corrupted_parts_never_publish_partial_results(tmp_path):
    packed = pack_transfer("results", run=sample_run(tmp_path), output=tmp_path / "package")
    manifest = Path(packed["manifest"])
    part = manifest.parent / json.loads(manifest.read_text())["parts"][0]["path"]
    contents = part.read_bytes()
    part.unlink()
    with pytest.raises(FileNotFoundError):
        unpack_transfer(manifest, tmp_path / "missing")
    part.write_bytes(b"x" * len(contents))
    with pytest.raises(ValueError, match="checksum"):
        unpack_transfer(manifest, tmp_path / "corrupt")
    assert not (tmp_path / "missing").exists()
    assert not (tmp_path / "corrupt").exists()


def test_first_unpack_needs_only_python_standard_library(tmp_path):
    packed = pack_transfer("results", run=sample_run(tmp_path), output=tmp_path / "package")
    bootstrap = Path(packed["manifest"]).parent / "bootstrap.pyz"
    result = subprocess.run(  # noqa: S603 -- generated CLI zipapp, isolated from all installed site packages
        [
            sys.executable,
            "-S",
            str(bootstrap),
            "unpack",
            packed["manifest"],
            "--output",
            str(tmp_path / "restored"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "restored/runs/test/validation/0.jsonl").is_file()


@pytest.mark.parametrize("link", [False, True])
def test_unsafe_paths_and_symlinks_are_rejected_even_with_matching_part_hash(tmp_path, link):
    archive = tmp_path / "payload.tar.gz.part.0000"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("escape" if link else "../escape")
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = str(tmp_path / "escape")
            stream.addfile(member)
        else:
            member.size = 1
            stream.addfile(member, io.BytesIO(b"x"))
    atomic_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "kind": "results",
            "details": {},
            "parts": [
                {
                    "path": archive.name,
                    "bytes": archive.stat().st_size,
                    "sha256": file_digest(archive),
                }
            ],
            "files": [{"path": "escape", "bytes": 0, "sha256": "unused"}],
        },
    )
    with pytest.raises(ValueError, match=r"Unsafe|nonregular"):
        unpack_transfer(tmp_path / "manifest.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_live_snapshot_is_explicit_and_cannot_package_itself(tmp_path):
    run = sample_run(tmp_path)
    atomic_json(run / "state.json", {"state": "running"})
    result = pack_transfer("results", run=run, output=tmp_path / "package")
    assert json.loads(Path(result["manifest"]).read_text())["details"]["live_snapshot"]
    with pytest.raises(ValueError, match="outside"):
        pack_transfer("results", run=run, output=run / "package")
