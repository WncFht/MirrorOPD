"""Prepare only the pinned native backend; never overwrite another working tree."""

import json
import subprocess
from pathlib import Path

from mirroropd.runtime.provenance import verify_backend


def _git(arguments, *, check=True, **kwargs):
    return subprocess.run(["/usr/bin/git", *arguments], check=check, **kwargs)  # noqa: S603 -- argument vector from the packaged integration manifest


def prepare_backend(destination):
    integration = Path(__file__).parent
    manifest = json.loads((integration / "backend.json").read_text())
    destination = Path(destination).expanduser().resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _git(
            ["clone", "--no-checkout", manifest["url"], str(destination)],
            check=True,
        )
        _git(
            ["checkout", "--detach", manifest["base_commit"]],
            cwd=destination,
            check=True,
        )
    reverse = _git(
        ["apply", "--reverse", "--check", str(integration / manifest["patch"])],
        cwd=destination,
        capture_output=True,
        check=False,
    )
    if reverse.returncode:
        _git(
            ["apply", "--check", str(integration / manifest["patch"])],
            cwd=destination,
            check=True,
        )
        _git(
            ["apply", str(integration / manifest["patch"])],
            cwd=destination,
            check=True,
        )
    return verify_backend(destination, integration / manifest["patch"], manifest["base_commit"])
