"""Own foreground/background controllers and signal only the recorded local process."""

import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import Job, with_algorithm
from mirroropd.runtime.reservation import pid_namespace, process_identity


def controller_identity():
    """Bind a PID to its start time, host and PID namespace."""
    return {
        "controller_pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "hostname": socket.gethostname(),
        "pid_namespace": pid_namespace(),
    }


def local_controller(record):
    """Refuse to interpret another node's PID as a local process."""
    return record["hostname"] == socket.gethostname() and record["pid_namespace"] == pid_namespace()


def run_status(directory):
    """Read recorded state; check liveness only on the controller's node/namespace."""
    directory = Path(directory)
    path = directory / "state.json"
    value = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "state": "starting",
            **json.loads((directory / "controller.json").read_text()),
        }
    )
    if "hostname" not in value:
        # Older states lack a host identity; they cannot prove local process ownership.
        value["process_check"] = "unavailable"
    elif not local_controller(value):
        value["process_check"] = "other_host_or_namespace"
    else:
        value["process_check"] = "local"
        if value["state"] in {"starting", "running", "preparing", "verifying"} and (
            process_identity(value["controller_pid"]) != value["process_identity"]
        ):
            value["state"] = "orphaned"
    return value


def stop_run(directory, *, timeout=30):
    """Request graceful shutdown and wait; never escalate to an unrelated process."""
    import psutil

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Stop timeout must be positive")
    directory = Path(directory)
    record = json.loads((directory / "controller.json").read_text())
    if not local_controller(record):
        raise ValueError(
            f"Stop this run on controller host {record['hostname']} in its PID namespace"
        )
    pid, identity = record["controller_pid"], record["process_identity"]
    if identity is None or process_identity(pid) != identity:
        return run_status(directory)
    try:
        process = psutil.Process(pid)
        if process_identity(pid) == identity:
            process.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:
        return run_status(directory)
    deadline = time.monotonic() + timeout
    while process_identity(pid) == identity:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Controller is still stopping; inspect controller.log and retry status"
            )
        time.sleep(0.1)
    return run_status(directory)


def start_run(
    job,
    *,
    directory,
    backend,
    python,
    algorithms=None,
    lease_root=None,
    borrowed_slot=None,
    prepare_only=False,
):
    """Reserve a new output directory and launch a hangup-immune controller."""
    directory = Path(directory).expanduser().resolve()
    request = {
        "schema_version": 2,
        "job": job.to_dict(),
        "backend": str(Path(backend).expanduser().resolve()),
        "python": str(Path(python).expanduser().absolute()),
        "algorithms": algorithms,
        "lease_root": str(Path(lease_root).expanduser().resolve()) if lease_root else None,
        "borrowed_slot": str(Path(borrowed_slot).expanduser().resolve()) if borrowed_slot else None,
        "prepare_only": prepare_only,
    }
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "request.json", request)
    command = ["nohup", sys.executable, "-u", "-m", "mirroropd.runtime.control", str(directory)]
    with (directory / "controller.log").open("x") as log:
        child = subprocess.Popen(  # noqa: S603 -- fixed controller module with explicit arguments
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = controller_identity() | {
        "controller_pid": child.pid,
        "process_identity": process_identity(child.pid),
        "directory": str(directory),
        "log": str(directory / "controller.log"),
    }
    atomic_json(directory / "controller.json", record)
    return child, record


def wait_run(child, directory):
    """Stream the controller log in foreground and forward termination requests."""
    directory = Path(directory)
    handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"Foreground launcher received signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        with (directory / "controller.log").open() as stream:
            while True:
                sys.stdout.write(stream.read())
                sys.stdout.flush()
                if child.poll() is not None:
                    sys.stdout.write(stream.read())
                    sys.stdout.flush()
                    return child.wait()
                time.sleep(0.1)
    except (KeyboardInterrupt, InterruptedError):
        stop_run(directory)
        return child.wait(timeout=5) or 1
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def controller_main(directory):
    """Execute one job or a sequential suite; persist failures before worker cleanup."""
    directory = Path(directory).resolve()
    owner = controller_identity()
    suite = False
    code = 1

    def state(name, **details):
        atomic_json(
            directory / "state.json",
            {
                "schema_version": 2,
                "run_id": directory.name,
                "state": name,
                **owner,
                "updated_unix": time.time(),
                **details,
            },
        )

    def interrupted(signum, _frame):
        raise InterruptedError(f"Controller received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        state("preparing")
        request = json.loads((directory / "request.json").read_text())
        job = Job.from_dict(request["job"])
        from mirroropd.runtime.runner import run_job

        options = {
            key: request[key]
            for key in (
                "backend",
                "python",
                "lease_root",
                "borrowed_slot",
                "prepare_only",
            )
        }
        algorithms = request["algorithms"]
        suite = algorithms is not None
        if suite:
            for algorithm in algorithms:
                state("running", active_run=algorithm)
                run_job(with_algorithm(job, algorithm), directory=directory / algorithm, **options)
            state("prepared" if request["prepare_only"] else "completed")
        else:
            run_job(job, directory=directory, directory_created=True, **options)
        code = 0
    except BaseException as error:  # noqa: BLE001 -- controller boundary records signals and all execution failures
        if suite or not (directory / "run.json").exists():
            state(
                "interrupted"
                if isinstance(error, (InterruptedError, KeyboardInterrupt))
                else "failed",
                error_type=type(error).__name__,
                error=str(error),
            )
        traceback.print_exc()
    finally:
        atomic_json(directory / "controller-exit.json", {"return_code": code})
    return code


if __name__ == "__main__":
    sys.exit(controller_main(sys.argv[1]))
