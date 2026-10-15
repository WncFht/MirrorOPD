"""Actual process identities, concurrent stale recovery and partial GPU rollback."""

import json
import multiprocessing
import os
import subprocess
import sys
import time

import pytest

from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.reservation import (
    DirectoryLease,
    GpuReservation,
    ResourceBusyError,
    process_identity,
)


def test_live_owner_and_unknown_namespaces_cannot_be_reclaimed(tmp_path):
    lease = DirectoryLease(tmp_path / "lock", "run")
    assert lease.acquire()
    assert not DirectoryLease(lease.directory, "other").acquire()
    lease.heartbeat()
    lease.release()
    lease.directory.mkdir()
    atomic_json(lease.directory / "owner.json", lease.owner | {"pid_namespace": "foreign"})
    assert not DirectoryLease(lease.directory, "other").acquire()


def test_dead_process_identity_can_be_reclaimed_without_age_expiry(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # noqa: S603 -- fixed local interpreter and source
    try:
        identity = process_identity(child.pid)
        lease = DirectoryLease(tmp_path / "lock", "old")
        lease.directory.mkdir()
        atomic_json(
            lease.directory / "owner.json",
            lease.owner | {"pid": child.pid, "process_identity": identity},
        )
        child.terminate()
        child.wait(timeout=10)
        replacement = DirectoryLease(lease.directory, "new")
        assert replacement.acquire()
        assert json.loads((replacement.directory / "owner.json").read_text())["run_id"] == "new"
        replacement.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def _contend(directory, active, peak, gate, errors):
    try:
        gate.wait(timeout=10)
        for _ in range(100):
            lease = DirectoryLease(directory, str(os.getpid()))
            if lease.acquire():
                with active.get_lock():
                    active.value += 1
                    peak.value = max(peak.value, active.value)
                time.sleep(0.02)
                with active.get_lock():
                    active.value -= 1
                lease.release()
                return
            time.sleep(0.005)
        errors.put("Contention did not settle")
    except BaseException as error:  # noqa: BLE001 -- relay every child failure to the parent assertion
        errors.put(repr(error))


def test_concurrent_dead_owner_recovery_never_overlaps_owners(tmp_path):
    lease = DirectoryLease(tmp_path / "lock", "old")
    lease.directory.mkdir()
    atomic_json(
        lease.directory / "owner.json", lease.owner | {"pid": 2**30, "process_identity": "dead"}
    )
    context = multiprocessing.get_context("spawn")
    active, peak = context.Value("i", 0), context.Value("i", 0)
    gate = context.Barrier(4)
    errors = context.Queue()
    children = [
        context.Process(target=_contend, args=(lease.directory, active, peak, gate, errors))
        for _ in range(4)
    ]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(timeout=15)
        assert all(not child.is_alive() and child.exitcode == 0 for child in children)
        assert errors.empty()
        assert peak.value == 1
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join()


def idle_snapshot(gpu_ids):
    return {
        index: {"uuid": f"GPU-{index}", "memory_mib": 1, "driver_pids": []} for index in gpu_ids
    }


def test_gpu_group_rolls_back_partial_acquisition_and_busy_contexts(tmp_path):
    occupied = DirectoryLease(tmp_path / "GPU-2.lease", "occupied")
    assert occupied.acquire()
    with (
        pytest.raises(ResourceBusyError, match="held"),
        GpuReservation([1, 2], tmp_path, "run", snapshot=idle_snapshot),
    ):
        raise AssertionError("Should not enter")
    assert not (tmp_path / "GPU-1.lease").exists()
    occupied.release()

    def busy(ids):
        return {
            index: {"uuid": f"GPU-{index}", "memory_mib": 1024, "driver_pids": [123]}
            for index in ids
        }

    with (
        pytest.raises(ResourceBusyError, match="active contexts"),
        GpuReservation([1, 2], tmp_path, "run", snapshot=busy),
    ):
        raise AssertionError("Should not enter")
    assert list(tmp_path.iterdir()) == []
    with GpuReservation([1, 2], tmp_path, "run", snapshot=idle_snapshot) as reservation:
        reservation.heartbeat()
        assert len(reservation.leases) == 2
    assert list(tmp_path.iterdir()) == []
