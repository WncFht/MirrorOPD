"""Atomic GPU reservation using the shared workspace directory-lease protocol.

The owner schema is compatible with the other experiment controllers. Keep
this self-contained so a fresh scalar checkout does not depend on untracked
One-Shot implementation files. GPU driver checks remain a separate guard.
"""

import json
import os
import time
import uuid
from pathlib import Path


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


class GpuGroupLease:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.owner = {
            "pid": os.getpid(),
            "pid_namespace": str(Path("/proc/self/ns/pid").readlink()),
            "process_identity": process_identity(os.getpid()),
            "nonce": uuid.uuid4().hex,
        }

    def acquire(self):
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.directory.mkdir()
        except FileExistsError:
            path = self.directory / "owner.json"
            if not path.exists():
                return False
            existing = json.loads(path.read_text())
            if (
                existing["pid_namespace"] == self.owner["pid_namespace"]
                and process_identity(existing["pid"]) != existing["process_identity"]
            ):
                path.unlink()
                self.directory.rmdir()
            return False
        self.heartbeat()
        return True

    def heartbeat(self):
        self.owner["heartbeat_unix"] = time.time()
        temporary = self.directory / "owner.tmp"
        temporary.write_text(json.dumps(self.owner) + "\n")
        temporary.replace(self.directory / "owner.json")

    def release(self):
        path = self.directory / "owner.json"
        if path.exists() and json.loads(path.read_text())["nonce"] == self.owner["nonce"]:
            path.unlink()
            self.directory.rmdir()
