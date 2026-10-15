"""Run gated production-estimator experiments on physical GPUs 0--3."""

import argparse
import json
import os
from pathlib import Path

from training.gpu_exclusive import GpuGroupLease, process_identity
from training.oneshot_suite import run_worker, write_status


def plan(phase):
    base = {
        "seed": 42,
        "kappa": 0.5,
        "eta": 0.3,
        "lr": 1e-6,
        "topj": 16,
        "method": "bpo",
        "center_mode": "topj_tail",
        "action_mode": "sampled",
        "responses_per_rank": 4,
        "train_prefixes": 16,
        "heldout_prefixes": 16,
        "updates": 20,
        "measure_every": 5,
        "retry_on_gpu_conflict": False,
        "deterministic": True,
    }
    if phase == "audit":
        return [
            base
            | {
                "id": "real-cache-audit",
                "responses_per_rank": 16,
                "train_prefixes": 4,
                "heldout_prefixes": 4,
            }
        ]
    variants = (
        ("topj16", "topj_tail", "sampled", 16),
        ("full-center", "full_vocab", "sampled", 16),
        ("full-actions", "full_vocab", "full_vocab", 16),
    )
    if phase == "lifecycle":
        return [
            base
            | {
                "id": "bpo-full-lifecycle",
                "center_mode": "full_vocab",
                "action_mode": "full_vocab",
                "responses_per_rank": 2,
                "train_prefixes": 4,
                "heldout_prefixes": 4,
                "updates": 4,
                "anchor_refresh": 2,
                "measure_every": 1,
                "checkpoint_roundtrip": True,
            }
        ]
    if phase in ("gradient", "precision"):
        return [
            base
            | {
                "id": f"{method}-parameter-gradient" + ("-fp32" if phase == "precision" else ""),
                "method": method,
                "center_mode": "full_vocab",
                "action_mode": "full_vocab",
                "responses_per_rank": 2,
                "train_prefixes": 8,
                "heldout_prefixes": 8,
                "updates": 1 if phase == "precision" else 2,
                "measure_every": 1,
                "parameter_gradient_check": True,
                "precision": "fp32" if phase == "precision" else "bf16",
            }
            for method in ("bpo", "mdpo")
        ]
    if phase not in ("canary", "train"):
        raise ValueError("Unknown experiment phase")
    return [
        base
        | {
            "id": f"{phase}-{method}-{name}",
            "method": method,
            "center_mode": center,
            "action_mode": action,
            "topj": topj,
        }
        | (
            {
                "updates": 2,
                "responses_per_rank": 1,
                "train_prefixes": 4,
                "heldout_prefixes": 4,
                "measure_every": 1,
            }
            if phase == "canary"
            else {}
        )
        for method in ("bpo", "mdpo")
        for name, center, action, topj in variants
    ]


def acquire_lease(root, suite):
    directory = root.parent / "runs/gpu-locks/physical-gpu-0-3.lease"
    owner_path = directory / "owner.json"
    if owner_path.exists():
        owner = json.loads(owner_path.read_text())
        identity = process_identity(owner["pid"])
        if identity == owner["process_identity"]:
            status = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0]
            if status not in ("T", "t"):
                raise RuntimeError(
                    "The original GPU controller is active; reservation cannot be borrowed"
                )
            (suite / "paused-reservation.json").write_text(json.dumps(owner, indent=2) + "\n")
            directory = directory / "power-implementation.lease"
    lease = GpuGroupLease(directory)
    if not lease.acquire():
        raise RuntimeError("GPU validation reservation is already held")
    return lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("audit", "canary", "train", "lifecycle", "gradient", "precision"),
        required=True,
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = root / "runs/power-implementation-20261015"
    suite.mkdir(exist_ok=True)
    configurations = plan(args.phase)
    (suite / f"{args.phase}-plan.json").write_text(json.dumps(configurations, indent=2) + "\n")
    cache = root / "data/oneshot-effective-time-20261012/seed-42"
    lease = acquire_lease(root, suite)
    try:
        for config in configurations:
            output = suite / config["id"]
            if (output / "completion.jsonl").exists():
                rows = [
                    json.loads(line)
                    for line in (output / "completion.jsonl").read_text().splitlines()
                ]
                if rows[-1].get("state") == "completed":
                    continue
            lease.heartbeat()
            run_worker(
                root,
                suite,
                "audit" if args.phase == "audit" else "train",
                config,
                output,
                cache,
                worker_entry="training/power_implementation_worker.py",
            )
        write_status(
            suite / "status.json",
            state="phase_completed",
            phase=args.phase,
            completed=[config["id"] for config in configurations],
            pid=os.getpid(),
        )
    finally:
        lease.release()


if __name__ == "__main__":
    main()
