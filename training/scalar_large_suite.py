"""Shared-lock, physical-GPU 0--3 queue for large scalar BPO experiments."""

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from training.scalar_large_gpu import gpu_driver_pids

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / "envs/opd/bin/python"


def inventory():
    raw = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {
        int(row.split(",")[0]): int(row.split(",")[1])
        for row in raw.splitlines()
        if int(row.split(",")[0]) in range(4)
    }


@contextlib.contextmanager
def gpu_phase(name):
    directory = ROOT.parent / "runs/gpu-locks"
    directory.mkdir(parents=True, exist_ok=True)
    handles = [(directory / f"physical-gpu-{device}.lock").open("a+") for device in range(4)]
    while True:
        acquired = []
        try:
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired.append(handle)

        except BlockingIOError:
            for handle in acquired:
                fcntl.flock(handle, fcntl.LOCK_UN)
            print(
                json.dumps({"phase": name, "status": "waiting_for_gpu_0_3", "unix": time.time()}),
                flush=True,
            )
            time.sleep(15)
        else:
            if all(value <= 512 for value in inventory().values()):
                break
            for handle in acquired:
                fcntl.flock(handle, fcntl.LOCK_UN)
            print(
                json.dumps(
                    {"phase": name, "status": "waiting_for_gpu_memory", "unix": time.time()}
                ),
                flush=True,
            )
            time.sleep(15)
    for handle in handles:
        handle.seek(0)
        handle.truncate()
        handle.write(f"scalar-large {name} pid={os.getpid()}\n")
        handle.flush()
    try:
        yield
    finally:
        for handle in handles:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def launch_phase(name, commands, output):
    output.mkdir(parents=True, exist_ok=True)
    pending = list(commands)
    active = {}
    completed = []
    with gpu_phase(name):
        while pending or active:
            for device in range(4):
                if device not in active and pending:
                    job, arguments = pending.pop(0)
                    owner_path = output / "owners" / f"device-{device}.json"
                    owner_path.unlink(missing_ok=True)
                    env = dict(
                        os.environ,
                        CUDA_DEVICE_ORDER="PCI_BUS_ID",
                        CUDA_VISIBLE_DEVICES=str(device),
                        PYTHONPATH=str(ROOT / "src") + ":" + str(ROOT),
                        TOKENIZERS_PARALLELISM="false",
                        OMP_NUM_THREADS="4",
                        SCALAR_GPU_OWNER_FILE=str(owner_path),
                    )
                    log = (output / f"{job}.log").open("a")
                    command = [str(PYTHON), *arguments, "--device", str(device)]
                    process = subprocess.Popen(  # noqa: S603 -- arguments come from the internal experiment plan
                        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    active[device] = (job, process, log)
                    print(
                        json.dumps(
                            {
                                "phase": name,
                                "job": job,
                                "device": device,
                                "pid": process.pid,
                                "command": command,
                            }
                        ),
                        flush=True,
                    )
            observed = gpu_driver_pids()
            for device in active:
                owner_path = output / "owners" / f"device-{device}.json"
                if owner_path.exists():
                    owner = json.loads(owner_path.read_text())
                    if observed[device] - {owner["driver_pid"]}:
                        for _, child, handle in active.values():
                            child.terminate()
                            child.wait()
                            handle.close()
                        raise RuntimeError(
                            "foreign GPU process detected; own jobs stopped, others preserved"
                        )
            for device, (job, process, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    completed.append({"job": job, "device": device, "exit_code": code})
                    del active[device]
                    print(json.dumps({"finished": job, "exit_code": code}), flush=True)
                    if code:
                        for _, child, handle in active.values():
                            child.terminate()
                            child.wait()
                            handle.close()
                        raise RuntimeError(f"{job} failed; partial data and checkpoints retained")
            (output / "phase-progress.json").write_text(
                json.dumps(
                    {
                        "phase": name,
                        "completed": completed,
                        "last_unix": time.time(),
                        "active": {k: v[0] for k, v in active.items()},
                        "pending": [v[0] for v in pending],
                    },
                    indent=2,
                )
                + "\n"
            )
            time.sleep(1)
    return completed


def canary_commands():
    base = {
        "cache": "runs/scalar-large/cache-canary",
        "seed": 1042,
        "train_sequences": 4,
        "batch_sequences": 2,
        "updates": 4,
        "eta": 0.3,
        "lr": 1e-6,
        "maximum": 1.6e-5,
        "precision": "fp32",
        "objective": "sampled",
        "topk": 16,
        "monitor": "full",
        "monitor_positions": 256,
        "method": "constant",
        "search_every": 1,
        "target_rate": 0.01,
        "audit_every": 2,
    }
    commands = []
    for method, objective in (
        ("constant", "sampled"),
        ("search", "sampled"),
        ("search", "sampled_full"),
        ("search", "exact"),
    ):
        name = f"canary-{method}-{objective}"
        config = base | {"method": method, "objective": objective}
        path = ROOT / "runs/scalar-large/configs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n")
        commands.append(
            (
                name,
                [
                    "training/scalar_large_worker.py",
                    "--config",
                    str(path),
                    "--output",
                    str(ROOT / "runs/scalar-large/canary" / name),
                ],
            )
        )
    return commands


def base_config(seed=1042):
    return {
        "cache": f"runs/scalar-large/cache-s{seed}",
        "seed": seed,
        "train_sequences": 64,
        "batch_sequences": 8,
        "updates": 600,
        "eta": 0.3,
        "lr": 1e-6,
        "maximum": 1.6e-5,
        "precision": "fp32",
        "objective": "sampled",
        "topk": 16,
        "monitor": "full",
        "monitor_positions": 512,
        "monitor_clusters": 8,
        "method": "constant",
        "search_every": 1,
        "target_rate": 0.01,
        "audit_every": 20,
    }


def training_commands(phase, configurations):
    result = []
    for name, config in configurations:
        config_path = ROOT / "runs/scalar-large/configs" / f"{name}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        result.append(
            (
                name,
                [
                    "training/scalar_large_worker.py",
                    "--config",
                    str(config_path),
                    "--output",
                    str(ROOT / "runs/scalar-large" / phase / name),
                ],
            )
        )
    return result


def tuning_configs():
    base = base_config() | {"updates": 240}
    result = [
        (f"tune-constant-{lr:.3g}", base | {"lr": lr})
        for lr in (2.5e-7, 5e-7, 1e-6, 1.5e-6, 2e-6, 4e-6, 8e-6, 1.6e-5)
    ]
    result += [
        (f"tune-search-every{every}", base | {"method": "search", "search_every": every})
        for every in (1, 4)
    ]
    result += [
        (f"tune-feedback-rate{rate}", base | {"method": "feedback", "target_rate": rate})
        for rate in (0.01, 0.03)
    ]
    result += [("tune-logloss-control", base | {"method": "logloss"})]
    return result


def confirmation_configs(selection):
    result = []
    for seed in (42, 142, 242, 342, 442):
        base = base_config(seed)
        for index, chosen in enumerate(selection["baseline_configs"]):
            result.append((f"confirm-constant{index + 1}-s{seed}", base | {"lr": chosen["lr"]}))
        result += [
            (
                f"confirm-search-every{every}-s{seed}",
                base | {"method": "search", "search_every": every},
            )
            for every in (1, 4)
        ]
        result += [
            (
                f"confirm-feedback-s{seed}",
                base | {"method": "feedback", "target_rate": selection["feedback_target"]},
            )
        ]
    return result


def ablation_configs(selection):
    baseline = base_config(42) | {"lr": selection["baseline_configs"][0]["lr"], "updates": 400}
    chosen = selection["primary_scalar_config"]
    scalar = base_config(42) | {
        "method": chosen["method"],
        "search_every": chosen["search_every"],
        "target_rate": chosen["target_rate"],
        "updates": 400,
    }
    arms = (("constant", baseline), ("scalar", scalar))
    conditions = [
        ("pool16", {"train_sequences": 16}),
        ("pool128", {"train_sequences": 128}),
        ("center64", {"topk": 64}),
        ("center-full", {"objective": "sampled_full"}),
        ("objective-exact", {"objective": "exact"}),
        ("precision-bf16", {"precision": "bf16"}),
        ("batch-full", {"batch_sequences": 64, "full_batch": True, "updates": 120}),
    ]
    result = [
        (f"ablate-{name}-{label}", config | changes)
        for name, changes in conditions
        for label, config in arms
    ]
    if scalar["method"] == "search":
        result += [
            ("ablate-monitor2048-scalar", scalar | {"monitor_positions": 2048}),
            ("ablate-monitor-sampled-scalar", scalar | {"monitor": "sampled"}),
            ("ablate-monitor-two-clusters-scalar", scalar | {"monitor_clusters": 2}),
        ]
    return result


def choose(root):
    from experiments.analysis.scalar_large_report import select_calibration

    return select_calibration(root / "tune", root / "selection.json")


def report(root, selection):
    from experiments.analysis.scalar_large_report import summarize

    for phase in ("tune", "confirm", "ablate"):
        summarize(
            root / phase,
            ROOT / "results/scalar-large" / phase,
            selection if phase == "confirm" else None,
        )


def audit_canaries(root):
    expected = (
        "canary-constant-sampled",
        "canary-search-sampled",
        "canary-search-sampled_full",
        "canary-search-exact",
    )
    records = []
    for name in expected:
        directory = root / "canary" / name
        summary = json.loads((directory / "summary.json").read_text())
        updates = [
            json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()
        ]
        assert summary["status"] == "complete"
        assert [row["update_after"] for row in updates] == [1, 2, 3, 4]
        assert (directory / "resume.pt").exists()
        records.append(
            {
                "name": name,
                "seconds": summary["seconds"],
                "counts": summary["counts"],
                "peak_gpu_gib": max(row["peak_gpu_gib"] for row in updates),
            }
        )
    (ROOT / "results/scalar-large/canary.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("canary-prepare", "prepare", "canary", "tune", "confirm", "ablate", "all"),
        required=True,
    )
    args = parser.parse_args()
    root = ROOT / "runs/scalar-large"
    if args.phase == "canary-prepare":
        commands = [
            (
                "canary-cache",
                [
                    "experiments/rollout/scalar_large_cache.py",
                    "--seed",
                    "1042",
                    "--train-sequences",
                    "4",
                    "--validation-sequences",
                    "4",
                    "--response-length",
                    "256",
                    "--output",
                    "runs/scalar-large/cache-canary",
                ],
            )
        ]
        launch_phase(args.phase, commands, root / args.phase)
        return
    if args.phase == "canary":
        launch_phase(args.phase, canary_commands(), root / args.phase)
        audit_canaries(root)
        return
    if args.phase in ("prepare", "all"):
        commands = [
            (
                f"cache-s{seed}",
                [
                    "experiments/rollout/scalar_large_cache.py",
                    "--seed",
                    str(seed),
                    "--output",
                    f"runs/scalar-large/cache-s{seed}",
                ],
            )
            for seed in (1042, 42, 142, 242, 342, 442)
        ]
        launch_phase("prepare", commands, root / "prepare")
    if args.phase in ("tune", "all"):
        configurations = tuning_configs()
        (root / "tune-plan.json").write_text(json.dumps(configurations, indent=2) + "\n")
        launch_phase("tune", training_commands("tune", configurations), root / "tune-controller")
        choose(root)
    if args.phase in ("confirm", "ablate", "all"):
        selection = json.loads((root / "selection.json").read_text())
        if args.phase in ("confirm", "all"):
            configurations = confirmation_configs(selection)
            (root / "confirm-plan.json").write_text(json.dumps(configurations, indent=2) + "\n")
            launch_phase(
                "confirm", training_commands("confirm", configurations), root / "confirm-controller"
            )
        if args.phase in ("ablate", "all"):
            configurations = ablation_configs(selection)
            (root / "ablate-plan.json").write_text(json.dumps(configurations, indent=2) + "\n")
            launch_phase(
                "ablate", training_commands("ablate", configurations), root / "ablate-controller"
            )
        report(root, root / "selection.json")


if __name__ == "__main__":
    main()
