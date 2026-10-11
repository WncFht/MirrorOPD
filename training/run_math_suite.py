"""Serialize validated math experiments in a detached supervisor process."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-pid", type=int, required=True)
    parser.add_argument("--canary-run", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = repo / "runs/math-4gpu-suite"
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "status.json"

    def record(**values):
        values["updated_unix"] = time.time()
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2) + "\n")
        temporary.replace(status_file)

    def verify(name):
        # Remote history may settle slightly after the training process exits.
        for attempt in range(5):
            with (output / f"{name}-audit.log").open("a") as log:
                result = subprocess.run(  # noqa: S603 - trusted repository commands, no shell
                    [
                        str(repo.parent / "envs/opd/bin/python"),
                        str(repo / "training/audit_math_run.py"),
                        str(repo / "runs" / name),
                        "--output",
                        str(repo / "results/training/math-4gpu" / f"{name}-verification.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode == 0:
                return
            if attempt < 4:
                time.sleep(30)
        raise RuntimeError(f"Completed-run audit failed: {name}")

    def run(method, name, *overrides):
        if (repo / "runs" / name).exists():
            raise RuntimeError(f"Refusing to replace existing run {name}")
        with (output / f"{name}-launch.log").open("w") as log:
            child = subprocess.Popen(  # noqa: S603 - fixed experiment commands, no shell
                [
                    "/bin/bash",
                    str(repo / "training/run_dapo_math_4gpu.sh"),
                    method,
                    name,
                    *overrides,
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=repo,
            )
            record(stage=name, state="running", pid=child.pid)
            exit_code = child.wait()
        if exit_code:
            raise RuntimeError(f"Training failed: {name}, exit={exit_code}")
        record(stage=name, state="verifying", process_exit_code=exit_code)
        verify(name)
        record(stage=name, state="verified")

    try:
        identity = process_identity(args.canary_pid)
        record(stage=args.canary_run, state="waiting_for_canary", pid=args.canary_pid)
        while identity is not None and process_identity(args.canary_pid) == identity:
            time.sleep(20)
        verify(args.canary_run)
        # The full validation loader queues all 570 examples to bounded vLLM
        # replicas; verify its memory and complete evaluation before long runs.
        run(
            "mirror_opd_onpolicy",
            "dapo-deepseek-onpolicy-canary-05-val-batch",
            "trainer.total_training_steps=1",
            "trainer.test_freq=1",
        )
        run("mirror_opd_onpolicy", "dapo-deepseek-opd-o50-j1")
        for method, short in (("mirror_mdpo", "mdpo"), ("mirror_bpo", "bpo")):
            run(
                method,
                f"dapo-deepseek-{short}-canary-j16",
                "trainer.total_training_steps=2",
                "data.val_max_samples=16",
                "trainer.test_freq=2",
            )
            run(method, f"dapo-deepseek-{short}-o50-j16")
        record(state="all_runs_verified")
    except Exception as error:
        record(state="stopped_on_failure", error=str(error))
        raise


if __name__ == "__main__":
    main()
