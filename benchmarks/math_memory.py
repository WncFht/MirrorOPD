"""Profile one math outer at the normal batch and context budgets on four GPUs.

Two inner updates initialize Adam states and exercise a reused frozen window.
Evaluation keeps the recipe's R16 and 31744-token cap but only four questions.
This measures capacity, not training quality or convergence.
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memory_stages import sample_devices

from mirroropd.cli import profile
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import load_job
from mirroropd.runtime.runner import run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    job = load_job(
        profile("math-four-gpu", "recipes"),
        models=profile("deepseek-justrl", "models"),
        dataset=profile("dapo-math", "datasets"),
        variables={
            "model_root": str(args.model_root.resolve()),
            "data_root": str(args.data_root.resolve()),
        },
        overrides=[
            "training.outer_steps=1",
            "training.inner_steps=2",
            "evaluation.before_training=false",
            "evaluation.every=1",
            "evaluation.question_limit=4",
            "logging.wandb.enabled=false",
            "memory.profile_phases=true",
        ],
    )
    from dataclasses import replace

    job = replace(job, data=replace(job.data, train=[str(args.train.resolve())]))
    atomic_json(output / "plan.json", job.to_dict())
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as sampler:
        observation = sampler.submit(
            sample_devices, output / "nvml.jsonl", stop, job.resources.gpu_ids
        )
        try:
            result = run_job(
                job,
                backend=args.backend,
                directory=output / "native",
                lease_root=args.lease_root,
            )
        finally:
            stop.set()
            observation.result()
    atomic_json(output / "status.json", {"state": "completed", "verification": result})


if __name__ == "__main__":
    main()
