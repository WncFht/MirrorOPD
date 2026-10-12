"""GPU 0--3 ownership records and conflict detection for serial experiments."""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path


def process_identity(pid):
    """Return a start-time identity only for a live process in this namespace."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


class GpuGroupLease:
    """Atomic-directory reservation; FUSE flock calls can block indefinitely."""

    def __init__(self, directory):
        """Record this controller's namespace and process identity."""
        self.directory = Path(directory)
        self.owner = {
            "pid": os.getpid(),
            "pid_namespace": str(Path("/proc/self/ns/pid").readlink()),
            "process_identity": process_identity(os.getpid()),
            "nonce": uuid.uuid4().hex,
        }

    def acquire(self):
        """Acquire an idle reservation or reclaim a provably dead local owner."""
        try:
            self.directory.mkdir()
        except FileExistsError:
            path = self.directory / "owner.json"
            try:
                existing = json.loads(path.read_text())
            except FileNotFoundError:
                return False
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
        """Refresh the public reservation without a blocking filesystem lock."""
        self.owner["heartbeat_unix"] = time.time()
        temporary = self.directory / "owner.tmp"
        temporary.write_text(json.dumps(self.owner) + "\n")
        temporary.replace(self.directory / "owner.json")

    def release(self):
        """Remove only this controller's reservation."""
        path = self.directory / "owner.json"
        if path.exists() and json.loads(path.read_text())["nonce"] == self.owner["nonce"]:
            path.unlink()
            self.directory.rmdir()


def gpu_processes():
    """Read driver process IDs, including processes outside this PID namespace."""
    inventory = subprocess.check_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = {}
    for line in inventory.splitlines():
        index, uuid, memory = (field.strip() for field in line.split(","))
        devices[int(index)] = {"uuid": uuid, "memory_mib": int(memory), "pids": []}
    processes = subprocess.check_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    by_uuid = {value["uuid"]: value for value in devices.values()}
    for line in processes.splitlines():
        uuid, pid = (field.strip() for field in line.split(","))
        by_uuid[uuid]["pids"].append(int(pid))
    return {index: devices[index] for index in range(4)}


def register_gpu_owner(directory, rank):
    """Register the sole context after a rank initializes its assigned GPU."""
    if os.environ["CUDA_VISIBLE_DEVICES"] not in ("0,1,2,3", str(rank)):
        raise RuntimeError("This experiment is restricted to physical GPUs 0--3")
    device = gpu_processes()[rank]
    if len(device["pids"]) != 1:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"conflict-{rank}.json").write_text(
            json.dumps({"physical_gpu": rank, "driver_pids": device["pids"]}) + "\n"
        )
        raise RuntimeError(f"GPU {rank} is not exclusively owned: {device}")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    record = {
        "rank": rank,
        "physical_gpu": rank,
        "driver_pid": device["pids"][0],
        "local_pid": os.getpid(),
        "registered_unix": time.time(),
    }
    temporary = path / f"rank-{rank}.tmp"
    temporary.write_text(json.dumps(record) + "\n")
    temporary.replace(path / f"rank-{rank}.json")


def foreign_gpu_processes(directory):
    """Distinguish driver PIDs from registered experiment workers."""
    owners = {}
    for path in Path(directory).glob("rank-*.json"):
        record = json.loads(path.read_text())
        owners[record["physical_gpu"]] = {record["driver_pid"]}
    observed = gpu_processes()
    return {
        index: set(value["pids"]) - owners[index]
        for index, value in observed.items()
        if index in owners and set(value["pids"]) - owners[index]
    }
