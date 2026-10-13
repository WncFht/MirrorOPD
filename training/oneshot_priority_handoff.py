"""Finish the verified current worker, run a prioritized study, then resume OneShot."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from training.gpu_exclusive import process_identity
from training.oneshot_suite import wait_for_idle, write_status


def checked_controller(repo, controller_pid, worker_pid):
    """Require the controller, worker, stage and reservation to agree before pausing."""
    lease = repo.parent / "runs/gpu-locks/physical-gpu-0-3.lease"
    owner = json.loads((lease / "owner.json").read_text())
    current = json.loads((repo / "runs/oneshot-effective-time-20261012/status.json").read_text())
    identity, worker_identity = process_identity(controller_pid), process_identity(worker_pid)
    namespace = str(Path("/proc/self/ns/pid").readlink())
    if (
        identity is None
        or worker_identity is None
        or owner["pid"] != controller_pid
        or owner["process_identity"] != identity
        or owner["pid_namespace"] != namespace
        or current["pid"] != worker_pid
        or current["phase"] != "train"
        or current["state"] != "running"
    ):
        raise RuntimeError("Current OneShot controller/worker/lease identity mismatch")
    command = Path(f"/proc/{controller_pid}/cmdline").read_bytes()
    if b"training.oneshot_suite" not in command:
        raise RuntimeError("The selected controller is not the OneShot suite")
    fields = Path(f"/proc/{worker_pid}/stat").read_text().rsplit(")", 1)[1].split()
    if int(fields[1]) != controller_pid:
        raise RuntimeError("The selected worker is not a child of this controller")
    return owner, current, worker_identity


def remove_retired_lease(repo, owner):
    """Remove a lease only after its recorded controller has exited."""
    directory = repo.parent / "runs/gpu-locks/physical-gpu-0-3.lease"
    path = directory / "owner.json"
    if not path.exists():
        return
    observed = json.loads(path.read_text())
    if observed["nonce"] != owner["nonce"]:
        raise RuntimeError("A different controller owns the reservation")
    if process_identity(owner["pid"]) == owner["process_identity"]:
        raise RuntimeError("Cannot release a live controller's reservation")
    (directory / "owner.tmp").unlink(missing_ok=True)
    path.unlink()
    directory.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--worker-pid", type=int, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.suite.mkdir(parents=True, exist_ok=True)
    status = args.suite / "handoff-status.json"
    owner, current, worker_identity = checked_controller(repo, args.controller_pid, args.worker_pid)
    record = {
        "controller": owner,
        "worker_pid": args.worker_pid,
        "worker_identity": worker_identity,
        "stage": current["stage"],
        "current_output": current["output"],
        "next": "training.power_mean_suite",
        "resume": "training.oneshot_suite",
        "authorized_priority": "after current training worker",
    }
    (args.suite / "handoff.json").write_text(json.dumps(record, indent=2) + "\n")
    retired = False
    os.kill(args.controller_pid, signal.SIGSTOP)
    try:
        # Pausing only the parent leaves the separately launched worker group running.
        reread = json.loads((repo / "runs/oneshot-effective-time-20261012/status.json").read_text())
        if reread["pid"] != args.worker_pid or reread["stage"] != current["stage"]:
            raise RuntimeError("OneShot changed stage during handoff registration")
        while process_identity(args.worker_pid) == worker_identity:
            progress_path = Path(current["output"]) / "progress.json"
            try:
                progress = json.loads(progress_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                progress = {}
            write_status(
                status,
                state="draining_current_training",
                pid=os.getpid(),
                current_stage=current["stage"],
                current_progress=progress,
                ready=(args.suite / "ready.json").exists(),
            )
            time.sleep(15)
        completion = json.loads((Path(current["output"]) / "completion.json").read_text())
        if completion["state"] != "completed":
            raise RuntimeError("Current OneShot worker did not complete successfully")
        while not (args.suite / "ready.json").exists():
            write_status(status, state="waiting_for_validated_study", pid=os.getpid())
            time.sleep(15)
        if process_identity(args.controller_pid) == owner["process_identity"]:
            os.kill(args.controller_pid, signal.SIGTERM)
            os.kill(args.controller_pid, signal.SIGCONT)
        while process_identity(args.controller_pid) == owner["process_identity"]:
            time.sleep(1)
        retired = True
        remove_retired_lease(repo, owner)
        wait_for_idle(status, "power-mean-priority")
        python = str(repo.parent / "envs/opd/bin/python")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repo) + ":" + str(repo / "src")
        write_status(status, state="running_prioritized_study", pid=os.getpid())
        result = subprocess.run(  # noqa: S603 -- fixed internal module and validated suite path
            [python, "-u", "-m", "training.power_mean_suite", "--suite", str(args.suite)],
            cwd=repo,
            env=environment,
            check=False,
        )
        write_status(status, state="prioritized_study_finished", exit_code=result.returncode)
        # Remaining original arms and checkpoint evaluations retain their original order.
        with (args.suite / "resumed-oneshot.log").open("a") as log:
            resumed = subprocess.Popen(  # noqa: S603 -- fixed original experiment controller
                [python, "-u", "-m", "training.oneshot_suite"],
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_status(
            status,
            state="original_queue_resumed",
            study_exit_code=result.returncode,
            resumed_controller_pid=resumed.pid,
        )
    finally:
        if not retired and process_identity(args.controller_pid) == owner["process_identity"]:
            os.kill(args.controller_pid, signal.SIGCONT)


if __name__ == "__main__":
    main()
