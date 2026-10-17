"""Measure paired short real-model runs without changing the training objective.

Worker CUDA peaks come from the existing profiling spans. A separate NVML
sampler observes whole devices without creating a CUDA context in the driver.
These synthetic arithmetic prompts measure execution, not task quality.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import Job
from mirroropd.runtime.runner import run_job


def sample_devices(path, stop, gpu_ids):
    import pynvml

    pynvml.nvmlInit()
    try:
        handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in gpu_ids}
        with path.open("w") as stream:
            while not stop.is_set():
                devices = {}
                for gpu, handle in handles.items():
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    devices[str(gpu)] = {
                        "used_bytes": memory.used,
                        "total_bytes": memory.total,
                        "utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                    }
                stream.write(json.dumps({"time_unix": time.time(), "devices": devices}) + "\n")
                stream.flush()
                stop.wait(0.25)
    finally:
        pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    for split, count in (("train", 16), ("test", 4)):
        rows = [
            {
                "question": f"Compute {i + 2} + {i + 3}. Give the integer answer.",
                "answer": str(2 * i + 5),
            }
            for i in range(count)
        ]
        (inputs / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    gpu_ids = [0, 1, 2, 3]
    completed = []
    for projection in ("eager", "chunked"):
        job = Job.from_dict(
            {
                "schema_version": 2,
                "name": f"memory-stages-{projection}",
                "models": {
                    "student": str(args.model_root / "DeepSeek-R1-Distill-Qwen-1.5B"),
                    "teacher": str(args.model_root / "JustRL-DeepSeek-1.5B"),
                },
                "data": {
                    "train": [str(inputs / "train.jsonl")],
                    "validation": [str(inputs / "test.jsonl")],
                    "columns": {"prompt": "question", "answer": "answer"},
                    "source_name": "memory-probe-arithmetic",
                    "scorer": {"name": "math"},
                },
                "objective": {
                    "algorithm": "bpo",
                    "signal": {"kappa": 0.5, "eta": 0.2},
                    "estimator": {"center": "full", "integration": "full", "topj": 16},
                    "diagnostics": "none",
                },
                "memory": {
                    "projection": projection,
                    "token_chunk_size": 64,
                    "vocab_chunk_size": 4096,
                    "profile_phases": True,
                },
                "training": {
                    "outer_steps": 1,
                    "inner_steps": 2,
                    "batch_size": 4,
                    "micro_batch_size": 1,
                    "learning_rate": 0.000001,
                    "weight_decay": 0.0,
                    "prompt_tokens": 256,
                    "response_tokens": 256,
                },
                "evaluation": {
                    "every": 1,
                    "samples_per_question": 2,
                    "prompt_tokens": 256,
                    "response_tokens": 128,
                    "question_limit": 4,
                },
                "resources": {
                    "gpu_ids": gpu_ids,
                    "cpus": 16,
                    "rollout_memory_fraction": 0.15,
                    "max_sequences": 4,
                    "max_batch_tokens": 1024,
                },
                "logging": {"wandb": {"enabled": False}},
            }
        )
        atomic_json(output / "status.json", {"stage": projection, "completed": completed})
        stop = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as sampler:
            observation = sampler.submit(
                sample_devices, output / f"{projection}-nvml.jsonl", stop, gpu_ids
            )
            try:
                result = run_job(
                    job,
                    backend=args.backend,
                    directory=output / projection,
                    lease_root=args.lease_root,
                )
            finally:
                stop.set()
                observation.result()
        completed.append({"projection": projection, "verification": result})
    atomic_json(output / "status.json", {"state": "completed", "completed": completed})


if __name__ == "__main__":
    main()
