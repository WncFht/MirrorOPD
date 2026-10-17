"""Exercise chunked projection and compact cache in the pinned four-GPU trainer.

This is a bounded synthetic canary, not a task-quality evaluation. The caller
must wait for existing jobs; run_job acquires the standard GPU reservations.
"""

import argparse
from pathlib import Path

from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import Job
from mirroropd.runtime.runner import run_job

CASES = (
    ("bpo", "sampled", "full"),
    ("bpo", "full", "full"),
    ("bpo", "full", "none"),
    ("mdpo", "sampled", "none"),
    ("mdpo", "full", "full"),
    ("on_policy", "full", "full"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--borrowed-slot", type=Path)
    parser.add_argument("--case", choices=["-".join(case) for case in CASES])
    parser.add_argument("--anchor-storage", choices=["log_probs", "hidden"], default="log_probs")
    parser.add_argument("--center", choices=["full", "topj_tail"])
    args = parser.parse_args()
    if args.anchor_storage == "hidden" and args.case != "bpo-sampled-full":
        parser.error("Hidden canary requires --case bpo-sampled-full")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from native_validation import tiny_inputs

    models = tiny_inputs(args.output / "inputs")
    completed = []
    for algorithm, integration, diagnostics in CASES:
        name = f"{algorithm}-{integration}-{diagnostics}"
        if args.case is not None and args.case != name:
            continue
        job = Job.from_dict(
            {
                "schema_version": 2,
                "name": name,
                "models": models,
                "data": {
                    "train": [str(args.output / "inputs/train.jsonl")],
                    "validation": [str(args.output / "inputs/test.jsonl")],
                    "source_name": "synthetic-qa",
                    "columns": {"prompt": "question", "answer": "answer"},
                    "scorer": {"name": "exact"},
                },
                "objective": {
                    "algorithm": algorithm,
                    "estimator": {
                        "center": args.center
                        or (
                            "full"
                            if integration == "full" or args.anchor_storage == "hidden"
                            else "topj_tail"
                        ),
                        "integration": integration,
                        "topj": 16,
                    },
                    "diagnostics": diagnostics,
                },
                "memory": {
                    "projection": "chunked",
                    "anchor_storage": args.anchor_storage,
                    "token_chunk_size": 3,
                    "vocab_chunk_size": 7,
                    "profile_phases": True,
                },
                "training": {
                    "outer_steps": 2,
                    "inner_steps": 1 if algorithm == "on_policy" else 2,
                    "batch_size": 4,
                    "prompt_tokens": 64,
                    "response_tokens": 16,
                    "temperature": 0.7,
                },
                "evaluation": {
                    "every": 2,
                    "samples_per_question": 2,
                    "prompt_tokens": 64,
                    "response_tokens": 32,
                    "question_limit": 4,
                },
                "resources": {
                    "gpu_ids": [0, 1, 2, 3],
                    "cpus": 16,
                    "rollout_memory_fraction": 0.05,
                    "max_sequences": 4,
                    "max_batch_tokens": 256,
                },
            }
        )
        atomic_json(args.output / "status.json", {"stage": name, "completed": completed})
        result = run_job(
            job,
            backend=args.backend,
            directory=args.output / name,
            lease_root=args.lease_root,
            borrowed_slot=args.borrowed_slot,
        )
        completed.append({"name": name, "verification": result})
    atomic_json(args.output / "status.json", {"state": "completed", "completed": completed})


if __name__ == "__main__":
    main()
