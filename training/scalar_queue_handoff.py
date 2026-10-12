"""Drain this session's workers before handing GPU0--3 to the prioritized study."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from training.scalar_gpu_lease import process_identity
from training.scalar_large_suite import PYTHON, ROOT


def children(pid):
    result = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            command = (path.parent / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except FileNotFoundError:
            continue
        if int(fields[1]) == pid and "training/scalar_large_worker.py" in command:
            result.append(
                {"pid": int(path.parent.name), "identity": fields[19], "command": command}
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-pid", type=int, required=True)
    args = parser.parse_args()
    identity = process_identity(args.controller_pid)
    cmdline = Path(f"/proc/{args.controller_pid}/cmdline").read_bytes()
    if identity is None or b"training.scalar_large_suite" not in cmdline:
        raise RuntimeError("handoff requires the verified scalar controller")
    os.kill(args.controller_pid, signal.SIGSTOP)
    try:
        workers = children(args.controller_pid)
        record = {
            "controller_pid": args.controller_pid,
            "controller_identity": identity,
            "workers": workers,
            "next": "training.scalar_full_pool_study",
        }
        (ROOT / "runs/scalar-large/queue-handoff.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        while any(process_identity(w["pid"]) == w["identity"] for w in workers):
            print(
                json.dumps(
                    {
                        "status": "draining_active_workers",
                        "workers": [
                            w["pid"] for w in workers if process_identity(w["pid"]) == w["identity"]
                        ],
                    }
                ),
                flush=True,
            )
            time.sleep(15)
        if process_identity(args.controller_pid) == identity:
            os.kill(args.controller_pid, signal.SIGTERM)
            os.kill(args.controller_pid, signal.SIGCONT)
        while process_identity(args.controller_pid) == identity:
            time.sleep(1)
    except BaseException:
        if process_identity(args.controller_pid) == identity:
            os.kill(args.controller_pid, signal.SIGCONT)
        raise
    # An interrupted heartbeat may leave a temporary file owned by the retired controller.
    lease = ROOT.parent / "runs/gpu-locks/physical-gpu-0-3.lease"
    try:
        owner = json.loads((lease / "owner.json").read_text())
    except FileNotFoundError:
        owner = None
    if (
        owner
        and owner["pid"] == args.controller_pid
        and owner["process_identity"] == identity
        and owner["pid_namespace"] == str(Path("/proc/self/ns/pid").readlink())
    ):
        (lease / "owner.tmp").unlink(missing_ok=True)
    subprocess.run(  # noqa: S603 -- fixed internal experiment argv
        [str(PYTHON), "-u", "-m", "training.scalar_full_pool_study"], cwd=ROOT, check=True
    )


if __name__ == "__main__":
    main()
