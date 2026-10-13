"""Serialize the preregistered One-Shot ablations with GPU ownership checks."""

import argparse
import atexit
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from training.gpu_exclusive import GpuGroupLease, foreign_gpu_processes, gpu_processes


def experiment_plan():
    """Materialize the 45 declared arms without consulting their outcomes."""
    base = {
        "batch_size": 64,
        "response_length": 7168,
        "diagnostic_positions": 4096,
        "diagnostic_every": 5,
        "topk": 16,
        "eta": 0.3,
        "clip_grad": 1.0,
        "weight_decay": 0.01,
        "anchor_interval": 0,
        "rollout": "frozen",
        "method": "opd",
        "wandb": True,
    }
    arms = []
    for seed in (42, 142, 242):
        arms.extend(
            base
            | {
                "id": f"zero-{precision}-s{seed}",
                "seed": seed,
                "precision": precision,
                "lr": 0.0,
                "updates": 4,
                "checkpoint_steps": [],
                "diagnostic_every": 1,
            }
            for precision in ("bf16", "fp32")
        )
        for rollout, precision in (("frozen", "bf16"), ("frozen", "fp32"), ("online", "bf16")):
            for tag, lr, updates in (("5em7", 5e-7, 600), ("1em6", 1e-6, 300), ("2em6", 2e-6, 150)):
                checkpoint_steps = [round(value / (lr / 1e-6)) for value in (50, 100, 200, 300)]
                arms.append(
                    base
                    | {
                        "id": f"{rollout}-opd-{precision}-lr{tag}-s{seed}",
                        "seed": seed,
                        "precision": precision,
                        "rollout": rollout,
                        "lr": lr,
                        "updates": updates,
                        "checkpoint_steps": checkpoint_steps,
                    }
                )
        for method in ("mdpo", "bpo"):
            for interval, updates in ((0, 300), (16, 800)):
                arms.append(
                    base
                    | {
                        "id": f"frozen-{method}-bf16-reset{interval}-s{seed}",
                        "seed": seed,
                        "precision": "bf16",
                        "method": method,
                        "anchor_interval": interval,
                        "lr": 1e-6,
                        "updates": updates,
                        "checkpoint_steps": sorted({50, 100, 200, 300, updates}),
                    }
                )
    return arms


def selected_experiment_plan(suite):
    """Apply an explicit user scope revision without changing preregistered arms."""
    original = experiment_plan()
    path = Path(suite) / "execution-selection.json"
    if not path.exists():
        return original
    selection = json.loads(path.read_text())
    identifiers = selection["active_ids"]
    available = {config["id"] for config in original}
    if selection["schema_version"] != 1 or not identifiers:
        raise ValueError("Invalid experiment selection")
    if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= available:
        raise ValueError("Selection contains duplicate or unknown experiment IDs")
    return [config for config in original if config["id"] in identifiers]


def write_status(path, **values):
    """Publish a complete status file atomically for independent observers."""
    values["updated_unix"] = time.time()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)


def wait_for_idle(status, stage):
    """Require three idle observations before touching GPUs 0--3."""
    consecutive = 0
    while consecutive < 3:
        devices = gpu_processes()
        idle = all(not value["pids"] and value["memory_mib"] <= 16 for value in devices.values())
        consecutive = consecutive + 1 if idle else 0
        write_status(
            status,
            stage=stage,
            state="checking_gpu_idle" if idle else "waiting_for_gpu_idle",
            observed=devices,
        )
        time.sleep(3)


def snapshot_sources(repo, destination):
    """Freeze only the runtime files, leaving parallel repository work untouched."""
    destination.mkdir(parents=True, exist_ok=True)
    paths = list((repo / "src/mirroropd").rglob("*.py")) + [
        repo / name
        for name in (
            "training/oneshot_worker.py",
            "training/gpu_exclusive.py",
            "training/oneshot_evaluation_job.py",
            "experiments/rollout/oneshot_batch.py",
            "experiments/scoring/oneshot_math_evaluate.py",
        )
    ]
    hashes = {}
    for path in paths:
        relative = path.relative_to(repo)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "source-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def run_worker(
    repo, suite, phase, config, output, cache=None, *, worker_entry="training/oneshot_worker.py"
):
    """Run one owned process group and stop only that group on a conflict."""
    status = suite / "status.json"
    attempt = max(
        (
            int(path.name.rsplit("-", 1)[-1])
            for path in (suite / "attempts").glob(config["id"] + "-*")
        ),
        default=0,
    )
    while True:
        attempt += 1
        wait_for_idle(status, config["id"])
        attempt_dir = suite / "attempts" / f"{config['id']}-{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        owners = attempt_dir / "owners"
        owners.mkdir()
        snapshot = attempt_dir / "source"
        snapshot_sources(repo, snapshot)
        if worker_entry != "training/oneshot_worker.py":
            relative = Path(worker_entry)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "training":
                raise ValueError("Experiment worker must be a repository training entrypoint")
            shutil.copy2(repo / relative, snapshot / relative)
            hashes_path = snapshot / "source-sha256.json"
            hashes = json.loads(hashes_path.read_text())
            hashes[str(relative)] = hashlib.sha256((snapshot / relative).read_bytes()).hexdigest()
            hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")
        config_path = attempt_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        environment = os.environ.copy()
        for variable in ("WANDB_API_KEY", "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_DISABLED"):
            environment.pop(variable, None)
        environment.update(
            CUDA_VISIBLE_DEVICES="0,1,2,3",
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            OMP_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",
            NCCL_DEBUG="WARN",
            PYTHONPATH=str(snapshot) + ":" + str(snapshot / "src"),
            NETRC=str(repo.parent / ".secrets/netrc"),
            WANDB_DIR=str(output),
            WANDB_CACHE_DIR=str(repo.parent / "cache/wandb"),
            WANDB_PROJECT=config.get("wandb_project", "mirror-opd-oneshot-absorption"),
            WANDB_MODE="online",
            WANDB_BASE_URL="https://api.wandb.ai",
            WANDB_CONFIG_DIR=str(repo.parent / "config/wandb"),
            CUDA_HOME="/usr/local/cuda-12.6",
            LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
        )
        command = [
            str(repo.parent / "envs/opd/bin/python"),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(snapshot / worker_entry),
            "--phase",
            phase,
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--owners",
            str(owners),
            "--repo",
            str(repo),
        ]
        if phase == "evaluate":
            command = [
                str(repo.parent / "envs/opd/bin/python"),
                str(snapshot / "training/oneshot_evaluation_job.py"),
                "--model",
                config["model"],
                "--repo",
                str(repo),
                "--source",
                str(snapshot),
                "--output",
                str(output),
                "--owners",
                str(owners),
            ]
        if cache is not None:
            command += ["--cache", str(cache)]
        conflict = None
        with (
            (attempt_dir / "console.log").open("w") as log,
            (attempt_dir / "gpu.jsonl").open("w") as monitor,
        ):
            process = subprocess.Popen(  # noqa: S603 -- fixed local worker argv
                command,
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_status(
                status,
                stage=config["id"],
                state="running",
                phase=phase,
                pid=process.pid,
                attempt=attempt,
                output=str(output),
                owners=str(owners),
            )
            while process.poll() is None:
                observed = gpu_processes()
                monitor.write(json.dumps({"unix": time.time(), "gpus": observed}) + "\n")
                monitor.flush()
                foreign = foreign_gpu_processes(owners)
                for path in owners.glob("conflict-*.json"):
                    observed_conflict = json.loads(path.read_text())
                    foreign[observed_conflict["physical_gpu"]] = set(
                        observed_conflict["driver_pids"]
                    )
                if foreign:
                    conflict = {str(index): sorted(pids) for index, pids in foreign.items()}
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(5)
            exit_code = process.wait()
        (attempt_dir / "exit.json").write_text(
            json.dumps({"exit_code": exit_code, "foreign_driver_pids": conflict}) + "\n"
        )
        if conflict:
            write_status(
                status,
                stage=config["id"],
                state="suspended_on_gpu_conflict",
                attempt=attempt,
                conflict=conflict,
                output=str(output),
            )
            if not config.get("retry_on_gpu_conflict", True):
                raise RuntimeError(f"{config['id']} stopped on GPU ownership conflict")
            continue
        if exit_code:
            write_status(
                status,
                stage=config["id"],
                state="stopped_on_failure",
                exit_code=exit_code,
                log=str(attempt_dir / "console.log"),
            )
            raise RuntimeError(f"{config['id']} exited {exit_code}; inspect {attempt_dir}")
        return


def audit_canary(output):
    """Check native anchor sync, finite metrics, four updates, and a usable HF checkpoint."""
    records = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert [record["update_before"] for record in records] == list(range(4))
    assert abs(records[0]["metrics"]["student_center/mean"]) < 1e-6
    assert records[0]["metrics"]["student_center/rms"] < 1e-5
    assert (output / "checkpoints/update-000004/model.safetensors").exists()
    diagnostics = [
        json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert [record["update"] for record in diagnostics] == [0, 1, 2, 3, 4]
    assert all(record["metrics"]["diagnostic_positions"] == 4096 for record in diagnostics)


def evaluate_model(repo, suite, identifier, model):
    """Serialize a complete 570-question evaluation and verify its output."""
    output = suite / "evaluation" / identifier
    if not (output / "completion.json").exists():
        run_worker(
            repo, suite, "evaluate", {"id": "eval-" + identifier, "model": str(model)}, output
        )
    completion = json.loads((output / "completion.json").read_text())
    assert completion["unique_questions"] == 570
    return output


def audit_training(output, config):
    """Require the declared update ledger, final measurement, and saved models."""
    records = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    by_update = {record["update_before"]: record for record in records}
    assert set(by_update) == set(range(config["updates"]))
    for step in config["checkpoint_steps"]:
        assert (output / f"checkpoints/update-{step:06d}/model.safetensors").exists()
    final = json.loads((output / "native-final.jsonl").read_text().splitlines()[-1])
    assert final["update"] == config["updates"]


def main():
    """Run canaries first; the complete preregistered matrix follows after validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    suite = repo / "runs/oneshot-effective-time-20261012"
    suite.mkdir(parents=True, exist_ok=True)
    locks = repo.parent / "runs/gpu-locks"
    locks.mkdir(parents=True, exist_ok=True)
    lease = GpuGroupLease(locks / "physical-gpu-0-3.lease")
    while not lease.acquire():
        write_status(
            suite / "status.json",
            state="waiting_for_gpu_reservation",
            observed=gpu_processes(),
            pid=os.getpid(),
        )
        time.sleep(5)
    atexit.register(lease.release)
    plan = experiment_plan()
    assert len(plan) == 45
    assert sum(arm["updates"] for arm in plan) == 16074
    (suite / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    plan = selected_experiment_plan(suite)
    (suite / "active-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for seed in sorted({config["seed"] for config in plan}):
        cache = repo / "data/oneshot-effective-time-20261012" / f"seed-{seed}"
        prepare = plan[0] | {"id": f"prepare-s{seed}", "seed": seed}
        if not (cache / "manifest.json").exists() or not all(
            (cache / f"rank-{rank}.pt").exists() for rank in range(4)
        ):
            run_worker(repo, suite, "prepare", prepare, cache)
        if seed == 42:
            for precision in ("bf16", "fp32"):
                canary = prepare | {
                    "id": f"canary-{precision}-s42",
                    "precision": precision,
                    "lr": 1e-6,
                    "updates": 4,
                    "diagnostic_every": 1,
                    "checkpoint_steps": [4],
                    "method": "opd",
                    "rollout": "frozen",
                    "anchor_interval": 0,
                }
                output = suite / canary["id"]
                if not (output / "completion.json").exists():
                    run_worker(repo, suite, "train", canary, output, cache)
                audit_canary(output)
            write_status(suite / "status.json", stage="canaries", state="canaries_verified")
            if args.canary_only:
                evaluate_model(
                    repo,
                    suite,
                    "canary-bf16-s42-update4",
                    suite / "canary-bf16-s42/checkpoints/update-000004",
                )
                write_status(
                    suite / "status.json",
                    stage="canaries",
                    state="canaries_and_full_evaluation_verified",
                )
                return
        evaluate_model(
            repo, suite, f"initial-s{seed}", repo.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B"
        )
        if seed == 42:
            evaluate_model(repo, suite, "teacher", repo.parent / "models/JustRL-DeepSeek-1.5B")
        for config in (arm for arm in plan if arm["seed"] == seed):
            output = suite / config["id"]
            if not (output / "completion.json").exists():
                run_worker(repo, suite, "train", config, output, cache)
            audit_training(output, config)
            for step in config["checkpoint_steps"]:
                evaluate_model(
                    repo,
                    suite,
                    config["id"] + f"-update{step}",
                    output / f"checkpoints/update-{step:06d}",
                )
            write_status(
                suite / "status.json",
                stage=config["id"],
                state="training_completed",
                output=str(output),
            )
    write_status(suite / "status.json", state="all_training_and_evaluation_completed")


if __name__ == "__main__":
    main()
