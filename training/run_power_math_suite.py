"""Serialize fresh-policy OPD, MDPO and BPO with the same power reward and avg@16."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from training.gpu_exclusive import gpu_processes
from training.oneshot_suite import snapshot_sources, write_status
from training.power_implementation_suite import acquire_lease


def plan(suite_name):
    shared = [
        "distillation.distillation_loss.mirror_kappa=0.5",
        "distillation.distillation_loss.mirror_eta=0.2",
        "distillation.distillation_loss.mirror_center_mode=topj_tail",
        "distillation.distillation_loss.mirror_action_mode=sampled",
        '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"',
        "ray_kwargs.ray_init.object_store_memory=4294967296",
    ]
    return [
        {
            "name": suite_name + "-canary",
            "method": "mirror_opd_onpolicy",
            "overrides": [
                *shared,
                "trainer.total_training_steps=2",
                "trainer.test_freq=2",
                "data.val_max_samples=4",
            ],
        },
        *[
            {"name": suite_name + "-" + name, "method": method, "overrides": shared}
            for name, method in (
                ("onpolicy", "mirror_opd_onpolicy"),
                ("mdpo", "mirror_mdpo"),
                ("bpo", "mirror_bpo"),
            )
        ],
    ]


def source_hashes(root, backend):
    records = []
    for label, directory in (("repository", root / "training"), ("backend", backend / "verl/verl")):
        records.extend(
            {
                "path": label + "/" + str(path.relative_to(directory)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".sh", ".json", ".yaml")
        )
    return records


def check_sources(root, backend, frozen):
    if source_hashes(root, backend) != frozen:
        raise RuntimeError("Training entrypoints or backend changed after source freeze")


def check_idle():
    observed = gpu_processes()
    if any(value["pids"] or value["memory_mib"] > 16 for value in observed.values()):
        raise RuntimeError(f"Physical GPUs 0--3 must be idle: {observed}")


def check_exit(child, name):
    if child.returncode:
        raise RuntimeError(f"Training failed: {name}, exit={child.returncode}")


def verify(root, suite, stage, environment):
    output = suite / (stage["name"] + "-verification.json")
    with (suite / (stage["name"] + "-audit.log")).open("a") as log:
        for attempt in range(5):
            result = subprocess.run(  # noqa: S603 -- fixed local audit, no shell interpolation
                [
                    environment["PYTHON_BIN"],
                    str(root / "training/audit_math_run.py"),
                    str(root / "runs" / stage["name"]),
                    "--output",
                    str(output),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode == 0:
                return json.loads(output.read_text())
            if attempt < 4:
                time.sleep(20)
    raise RuntimeError(f"Completed-run audit failed: {stage['name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--suite-name", default="math-avg-k05-e02-20261015")
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args()
    if Path(args.suite_name).name != args.suite_name:
        raise ValueError("Suite name must be a single path component")
    root = Path(__file__).resolve().parents[1]
    suite = root / "runs" / args.suite_name
    suite.mkdir(exist_ok=False)
    backend = args.backend.resolve()
    check_idle()
    subprocess.run(  # noqa: S603 -- prepare the exact local verifier before freezing sources
        [str(root.parent / "envs/opd/bin/python"), str(root / "training/prepare_math_backend.py")],
        env=os.environ | {"MIRROR_UPSTREAM_ROOT": str(backend)},
        check=True,
    )
    stages = plan(args.suite_name)
    if args.canary_only:
        stages = stages[:1]
    (suite / "plan.json").write_text(json.dumps(stages, indent=2) + "\n")
    snapshot_sources(root, suite / "source")
    frozen = source_hashes(root, backend)
    (suite / "runtime-sha256.json").write_text(json.dumps(frozen, indent=2) + "\n")
    environment = os.environ | {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "MIRROR_UPSTREAM_ROOT": str(backend),
        "MIRROROPD_SRC_ROOT": str(suite / "source/src"),
        "PYTHON_BIN": str(root.parent / "envs/opd/bin/python"),
        "PYTHONPATH": str(suite / "source/src") + ":" + str(root),
        "MIRROR_RAY_TMPDIR": str(Path.home() / ".cache/mopd-avg-ray"),
        "MIRROR_TMPDIR": str(Path.home() / ".cache/mopd-avg-tmp"),
        "VLLM_USE_V1": "1",
        "WANDB_RUN_GROUP": args.suite_name,
    }
    lease = acquire_lease(root, suite)
    status = suite / "status.json"
    completed = []
    child = None

    def stop(signum, _frame):
        raise InterruptedError(f"Suite interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for stage in stages:
            check_sources(root, backend, frozen)
            for _ in range(3):
                check_idle()
                lease.heartbeat()
                time.sleep(3)
            argv = [
                "/bin/bash",
                str(root / "training/run_dapo_math_4gpu.sh"),
                stage["method"],
                stage["name"],
                *stage["overrides"],
            ]
            (suite / (stage["name"] + "-command.json")).write_text(
                json.dumps(argv, indent=2) + "\n"
            )
            with (suite / (stage["name"] + "-launch.log")).open("w") as log:
                child = subprocess.Popen(  # noqa: S603 -- local launcher and explicit argument list
                    argv,
                    cwd=root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while child.poll() is None:
                    lease.heartbeat()
                    write_status(
                        status,
                        state="running",
                        stage=stage["name"],
                        controller_pid=os.getpid(),
                        child_pid=child.pid,
                        completed=completed,
                    )
                    time.sleep(10)
                check_exit(child, stage["name"])
            child = None
            write_status(status, state="verifying", stage=stage["name"], completed=completed)
            verification = verify(root, suite, stage, environment)
            completed.append({"run": stage["name"], "inner_updates": verification["inner_updates"]})
            write_status(status, state="verified", stage=stage["name"], completed=completed)
        write_status(status, state="completed", canary_only=args.canary_only, completed=completed)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        write_status(status, state="stopped_on_failure", error=str(error), completed=completed)
        raise
    finally:
        lease.release()


if __name__ == "__main__":
    main()
