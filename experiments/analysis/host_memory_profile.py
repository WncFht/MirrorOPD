"""Sample Linux host/process memory without importing CUDA into the monitor."""

import argparse
import json
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with args.output.open("w") as stream:
        while time.monotonic() - start < args.seconds:
            processes = []
            for proc in psutil.process_iter(["pid", "ppid", "name", "memory_info"]):
                try:
                    if proc.info["memory_info"].rss < 20 * 1024**2:
                        continue
                    mem = proc.memory_full_info()
                    processes.append(
                        {
                            "pid": proc.pid,
                            "ppid": proc.ppid(),
                            "name": proc.name(),
                            "rss": mem.rss,
                            "pss": getattr(mem, "pss", None),
                            "uss": getattr(mem, "uss", None),
                            "swap": getattr(mem, "swap", None),
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            record = {
                "time": time.time(),
                "elapsed": time.monotonic() - start,
                "virtual": psutil.virtual_memory()._asdict(),
                "swap": psutil.swap_memory()._asdict(),
                "meminfo": Path("/proc/meminfo").read_text(),
                "processes": processes,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
