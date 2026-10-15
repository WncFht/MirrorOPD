"""Track and stop only descendants of the job process, with PID reuse protection."""

import os
import signal
import subprocess
import time

from mirroropd.runtime.reservation import process_identity


class ProcessTree:
    def __init__(self, process):
        self.process = process
        self.identities = {process.pid: process_identity(process.pid)}

    def observe(self):
        import psutil

        for pid, identity in list(self.identities.items()):
            if process_identity(pid) != identity:
                continue
            try:
                children = psutil.Process(pid).children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for child in children:
                self.identities[child.pid] = process_identity(child.pid)

    def terminate(self, timeout=20):
        self.observe()
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
        for pid, identity in reversed(list(self.identities.items())):
            if identity is not None and process_identity(pid) == identity:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not any(
                identity is not None and process_identity(pid) == identity
                for pid, identity in self.identities.items()
            ):
                break
            time.sleep(0.1)
        for pid, identity in self.identities.items():
            if identity is not None and process_identity(pid) == identity:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return


def launch(argv, *, cwd, environment, log):
    child = subprocess.Popen(  # noqa: S603 -- explicit argument vector from the validated job compiler
        argv,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ProcessTree(child)
