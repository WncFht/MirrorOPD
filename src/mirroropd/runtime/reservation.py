"""One atomic-directory ownership protocol for GPU jobs and borrowed reservations."""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from mirroropd.runtime.artifacts import atomic_json


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


def process_state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return None


def pid_namespace():
    return str(Path("/proc/self/ns/pid").readlink())


class ResourceBusyError(RuntimeError):
    pass


class DirectoryLease:
    """Reclaim only provably dead local owners; heartbeat age never grants ownership."""

    def __init__(self, directory, run_id):
        self.directory = Path(directory)
        self.owner = {
            "pid": os.getpid(),
            "pid_namespace": pid_namespace(),
            "process_identity": process_identity(os.getpid()),
            "nonce": uuid.uuid4().hex,
            "run_id": run_id,
            "schema_version": 2,
        }

    def acquire(self):
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                self.directory.mkdir()
            except FileExistsError:
                if not self._reclaim_dead():
                    return False
            else:
                atomic_json(
                    self.directory / "owner.json", self.owner | {"heartbeat_unix": time.time()}
                )
                return True
        return False

    def _reclaim_dead(self):
        claim = self.directory / "_reclaim"
        token = None
        try:
            claim.mkdir()
        except (FileExistsError, FileNotFoundError):
            return False
        try:
            token = claim / self.owner["nonce"]
            token.write_text(self.owner["nonce"])
            owner = json.loads((self.directory / "owner.json").read_text())
            dead = (
                owner.get("pid_namespace") == self.owner["pid_namespace"]
                and process_identity(owner["pid"]) != owner["process_identity"]
            )
            only_claim = {path.name for path in self.directory.iterdir()} == {
                "owner.json",
                "_reclaim",
            }
            if not dead or not only_claim:
                return False
            retired = self.directory.with_name(
                self.directory.name + ".retired-" + self.owner["nonce"]
            )
            self.directory.rename(retired)
            claim = retired / "_reclaim"
            token = claim / self.owner["nonce"]
            (retired / "owner.json").unlink()
            token.unlink()
            claim.rmdir()
            retired.rmdir()
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
        else:
            return True
        finally:
            if token is not None and token.exists():
                token.unlink()
                claim.rmdir()

    def heartbeat(self):
        path = self.directory / "owner.json"
        if not path.exists() or json.loads(path.read_text())["nonce"] != self.owner["nonce"]:
            raise ResourceBusyError("Reservation ownership was lost")
        atomic_json(path, self.owner | {"heartbeat_unix": time.time()})

    def release(self):
        path = self.directory / "owner.json"
        if not path.exists() or json.loads(path.read_text())["nonce"] != self.owner["nonce"]:
            return
        for _ in range(20):
            children = {item.name for item in self.directory.iterdir()}
            if children == {"owner.json"}:
                break
            if children != {"owner.json", "_reclaim"}:
                raise ResourceBusyError("Reservation still contains child reservations")
            time.sleep(0.01)
        else:
            raise ResourceBusyError("Reservation has an unfinished recovery claim")
        path.unlink()
        self.directory.rmdir()


def gpu_snapshot(gpu_ids):
    inventory = subprocess.check_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = {}
    for line in inventory.splitlines():
        index, identity, used, total, utilization = (field.strip() for field in line.split(","))
        if int(index) in gpu_ids:
            devices[int(index)] = {
                "uuid": identity,
                "memory_mib": int(used),
                "capacity_mib": int(total),
                "utilization_percent": int(utilization),
                "driver_pids": [],
            }
    if set(devices) != set(gpu_ids):
        raise ValueError("Requested physical GPU IDs are not present")
    by_id = {value["uuid"]: value for value in devices.values()}
    processes = subprocess.check_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in processes.splitlines():
        identity, pid = (field.strip() for field in line.split(","))
        if identity in by_id:
            by_id[identity]["driver_pids"].append(int(pid))
    return devices


class GpuReservation:
    """Acquire every physical GPU atomically as a group or release the partial group."""

    def __init__(self, gpu_ids, directory, run_id, *, borrowed_slot=None, snapshot=gpu_snapshot):
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(index < 0 for index in gpu_ids):
            raise ValueError("GPU IDs must be a nonempty distinct set of physical indices")
        self.gpu_ids = tuple(sorted(gpu_ids))
        self.directory = Path(directory)
        self.run_id = run_id
        self.borrowed_slot = Path(borrowed_slot) if borrowed_slot else None
        self.snapshot = snapshot
        self.leases = []
        self.borrowed_parent = None

    def __enter__(self):
        try:
            self._acquire()
        except BaseException:
            self._release()
            raise
        return self

    def _acquire(self):
        if self.borrowed_slot:
            parent = json.loads((self.borrowed_slot.parent / "owner.json").read_text())
            if (
                parent["pid_namespace"] != pid_namespace()
                or process_identity(parent["pid"]) != parent["process_identity"]
            ):
                raise ResourceBusyError("Borrowed parent owner identity is not live locally")
            if process_state(parent["pid"]) not in {"T", "t"}:
                raise ResourceBusyError("Borrowed parent owner must be paused")
            self.borrowed_parent = parent
        observed = self.snapshot(self.gpu_ids)
        directories = [
            self.directory / (observed[index]["uuid"] + ".lease") for index in self.gpu_ids
        ]
        if self.borrowed_slot:
            directories.insert(0, self.borrowed_slot)
        for directory in directories:
            lease = DirectoryLease(directory, self.run_id)
            if not lease.acquire():
                raise ResourceBusyError(f"Resource reservation is held: {directory}")
            self.leases.append(lease)
        self.assert_idle()

    def assert_idle(self):
        observed = self.snapshot(self.gpu_ids)
        if any(device["driver_pids"] or device["memory_mib"] > 64 for device in observed.values()):
            raise ResourceBusyError(f"Requested GPUs have active contexts: {observed}")
        return observed

    def heartbeat(self):
        if self.borrowed_parent and (
            process_identity(self.borrowed_parent["pid"])
            != self.borrowed_parent["process_identity"]
            or process_state(self.borrowed_parent["pid"]) not in {"T", "t"}
        ):
            raise ResourceBusyError("Borrowed parent is no longer paused with the same identity")
        for lease in self.leases:
            lease.heartbeat()

    def _release(self):
        for lease in reversed(self.leases):
            lease.release()
        self.leases.clear()

    def __exit__(self, *_args):
        self._release()
