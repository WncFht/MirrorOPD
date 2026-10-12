"""CPU-only partial statistics while the serial GPU experiment controller runs."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from training.gpu_exclusive import process_identity


def main():
    """Refresh statistics without restarting or interfering with the GPU job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-pid", type=int, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    suite = repo / "runs/oneshot-effective-time-20261012"
    output = repo / "results/oneshot-effective-time"
    output.mkdir(parents=True, exist_ok=True)
    identity = process_identity(args.controller_pid)
    if identity is None:
        raise RuntimeError("Report watcher requires a verified live controller")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(repo) + ":" + str(repo / "src"))
    while True:
        with (output / "analysis.log").open("a") as log:
            result = subprocess.run(  # noqa: S603 -- trusted CPU analysis argv
                [
                    str(repo.parent / "envs/opd/bin/python"),
                    str(repo / "experiments/analysis/oneshot_effective_time.py"),
                    "--suite",
                    str(suite),
                    "--output",
                    str(output),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        live = process_identity(args.controller_pid) == identity
        (output / "watcher-status.json").write_text(
            json.dumps(
                {
                    "controller_pid": args.controller_pid,
                    "controller_live": live,
                    "analysis_exit_code": result.returncode,
                    "updated_unix": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        if not live:
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
