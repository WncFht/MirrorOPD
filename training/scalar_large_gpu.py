"""Driver PID ownership checks restricted to physical GPUs 0--3."""

import json
import os
import subprocess
from pathlib import Path


def gpu_driver_pids():
    inventory = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    uuids = {
        int(line.split(",")[0]): line.split(",")[1].strip()
        for line in inventory.splitlines()
        if int(line.split(",")[0]) in range(4)
    }
    processes = subprocess.check_output(
        [
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        device: {
            int(line.split(",")[1])
            for line in processes.splitlines()
            if line.split(",")[0].strip() == uuid
        }
        for device, uuid in uuids.items()
    }


def register_owner(device):
    filename = os.environ.get("SCALAR_GPU_OWNER_FILE")
    if filename is None:
        raise RuntimeError("GPU workers must be started by the shared-lock controller")
    pids = gpu_driver_pids()[device]
    if len(pids) != 1:
        raise RuntimeError(f"physical GPU {device} is not exclusively owned: {pids}")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"device": device, "driver_pid": next(iter(pids)), "local_pid": os.getpid()})
        + "\n"
    )
    temporary.replace(path)
