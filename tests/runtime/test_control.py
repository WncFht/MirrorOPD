"""Real detached sessions, persistent failure logs and ownership-checked shutdown."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from mirroropd.cli import main
from mirroropd.runtime import control
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import Job


def spec():
    return Job.from_dict(
        {
            "name": "control",
            "models": {"student": "s", "teacher": "t"},
            "data": {"train": ["train"], "validation": ["test"]},
        }
    )


def eventually(predicate):
    deadline = time.monotonic() + 15
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Controller did not reach the expected state")
        time.sleep(0.05)


def test_detached_controller_survives_launcher_exit_and_hangup_then_stops(tmp_path):
    directory = tmp_path / "run"
    # Replace only GPU execution: the real launcher, nohup, session, logs and stop path run.
    worker = """
import json, signal, sys, time
from pathlib import Path
from mirroropd.runtime.control import controller_identity
from mirroropd.runtime.artifacts import atomic_json
root = Path(sys.argv[1])
def state(name):
    atomic_json(root / "state.json", {"state": name, **controller_identity()})
def stop(signum, frame):
    state("interrupted")
    print("stopped", flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
state("running")
print("ready", flush=True)
while True:
    time.sleep(0.05)
"""
    launcher = """
import json, subprocess, sys
from pathlib import Path
from mirroropd.runtime.control import start_run
from mirroropd.runtime.config import Job
original = subprocess.Popen
def launch(argv, **kwargs):
    return original([argv[0], argv[1], "-u", "-c", sys.argv[2], sys.argv[1]], **kwargs)
subprocess.Popen = launch
job = Job.from_dict(json.loads(sys.argv[3]))
child, record = start_run(job, directory=Path(sys.argv[1]), backend=Path("unused"), python=sys.executable)
print(json.dumps(record))
"""
    result = subprocess.run(  # noqa: S603 -- fixed launcher and fake GPU worker source
        [sys.executable, "-c", launcher, str(directory), worker, json.dumps(spec().to_dict())],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    record = json.loads(result.stdout)
    pid = record["controller_pid"]
    try:
        eventually((directory / "state.json").exists)
        assert control.run_status(directory)["state"] == "running"
        assert os.getsid(pid) == pid
        os.kill(pid, signal.SIGHUP)
        time.sleep(0.1)
        assert control.run_status(directory)["state"] == "running"
        assert control.stop_run(directory, timeout=5)["state"] == "interrupted"
        assert (directory / "controller.log").read_text() == "ready\nstopped\n"
        with pytest.raises(FileExistsError):
            control.start_run(spec(), directory=directory, backend="unused", python=sys.executable)
    finally:
        if control.process_identity(pid) == record["process_identity"]:
            os.kill(pid, signal.SIGKILL)


def test_status_and_stop_do_not_interpret_foreign_or_reused_pids(tmp_path):
    record = control.controller_identity() | {"hostname": "another-node", "state": "running"}
    atomic_json(tmp_path / "controller.json", record)
    atomic_json(tmp_path / "state.json", record)
    assert control.run_status(tmp_path)["state"] == "running"
    assert control.run_status(tmp_path)["process_check"] == "other_host_or_namespace"
    with pytest.raises(ValueError, match="controller host"):
        control.stop_run(tmp_path)
    record = control.controller_identity() | {"process_identity": "reused", "state": "running"}
    atomic_json(tmp_path / "controller.json", record)
    atomic_json(tmp_path / "state.json", record)
    assert control.stop_run(tmp_path)["state"] == "orphaned"
    assert control.process_identity(os.getpid()) is not None


@pytest.mark.parametrize("detach", [False, True])
def test_cli_records_real_preparation_failure_and_exit_code(tmp_path, detach, capsys):
    directory = tmp_path / "failed"
    args = [
        "run",
        "--recipe",
        "smoke",
        "--models",
        "deepseek-justrl",
        "--dataset",
        "dapo-math",
        "--var",
        "model_root=/missing",
        "--var",
        "data_root=/missing",
        "--algorithm",
        "on_policy",
        "--backend",
        str(tmp_path / "missing-backend"),
        "--output",
        str(directory),
        "--prepare-only",
    ]
    assert main([*args, "--detach"] if detach else args) == (0 if detach else 1)
    eventually((directory / "controller-exit.json").exists)
    assert control.run_status(directory)["state"] == "failed"
    assert json.loads((directory / "controller-exit.json").read_text())["return_code"] == 1
    assert "Traceback" in (directory / "controller.log").read_text()
    assert "missing-backend" in (directory / "controller.log").read_text()
    if detach:
        assert json.loads(capsys.readouterr().out)["directory"] == str(directory)


def test_backend_interpreter_path_preserves_virtual_environment_symlink(tmp_path, monkeypatch):
    original = subprocess.Popen

    def launch(argv, **kwargs):
        assert argv[0] == "nohup"
        return original([sys.executable, "-c", "pass"], **kwargs)

    monkeypatch.setattr(control.subprocess, "Popen", launch)
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    directory = tmp_path / "run"
    child, _ = control.start_run(spec(), directory=directory, backend="unused", python=executable)
    child.wait(timeout=5)
    assert json.loads((directory / "request.json").read_text())["python"] == str(executable)
