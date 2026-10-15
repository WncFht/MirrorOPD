"""Validate model/data portability, full integration and the long production evaluation budget."""

import argparse
import json
import time
from pathlib import Path

from mirroropd.cli import profile
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.config import Job, load_job, with_algorithm
from mirroropd.runtime.reservation import process_identity
from mirroropd.runtime.runner import run_job


def tiny_inputs(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<unk>": 3,
        **{f"t{i}": i + 4 for i in range(28)},
    }
    special = list(vocabulary)[:4]
    raw = Tokenizer(WordLevel(vocabulary, unk_token=special[3]))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token=special[0],
        eos_token=special[1],
        bos_token=special[2],
        unk_token=special[3],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ bos_token }}{{ message['content'] }}{% endfor %}"
    )
    paths = {}
    # Head dimensions 64/128 exercise the supported FlashAttention path. Smaller
    # heads select vLLM's experimental flex backend on the pinned stack.
    for role, width, padded, seed in (("student", 128, 64, 10), ("teacher", 256, 96, 20)):
        path = directory / role
        path.mkdir()
        torch.manual_seed(seed)
        config = Qwen2Config(
            vocab_size=padded,
            hidden_size=width,
            intermediate_size=width * 2,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8192,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            attention_dropout=0.1,
        )
        Qwen2ForCausalLM(config).save_pretrained(path)
        tokenizer.save_pretrained(path)
        paths[role] = str(path.resolve())
    for split, count in (("train", 16), ("test", 4)):
        rows = [
            {"question": f"t{i % 28} t{(i + 1) % 28}", "answer": f"t{(i + 2) % 28}"}
            for i in range(count)
        ]
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--borrowed-slot", type=Path)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    models = tiny_inputs(args.output / "tiny-inputs")
    plans = [
        Job.from_dict(
            {
                "schema_version": 2,
                "name": "tiny-" + algorithm + "-full",
                "models": models,
                "data": {
                    "train": [str(args.output / "tiny-inputs/train.jsonl")],
                    "validation": [str(args.output / "tiny-inputs/test.jsonl")],
                    "source_name": "synthetic-qa",
                    "columns": {"prompt": "question", "answer": "answer"},
                    "scorer": {"name": "exact"},
                },
                "objective": {
                    "algorithm": algorithm,
                    "estimator": {"center": "full", "integration": "full", "topj": 16},
                },
                "training": {
                    "outer_steps": 2,
                    "inner_steps": 1 if algorithm == "on_policy" else 2,
                    "batch_size": 4,
                    "prompt_tokens": 64,
                    "response_tokens": 16,
                },
                "evaluation": {
                    "every": 2,
                    "samples_per_question": 2,
                    "prompt_tokens": 64,
                    "response_tokens": 32,
                    "question_limit": 4,
                },
                "resources": {
                    "cpus": 16,
                    "rollout_memory_fraction": 0.05,
                    "max_sequences": 4,
                    "max_batch_tokens": 256,
                },
            }
        )
        for algorithm in ("bpo", "mdpo", "on_policy")
    ]
    large = load_job(
        profile("math-four-gpu", "recipes"),
        models=profile("deepseek-justrl", "models"),
        dataset=profile("dapo-math", "datasets"),
        variables={
            "model_root": str(args.model_root.resolve()),
            "data_root": str(args.data_root.resolve()),
        },
        overrides=[
            "training.outer_steps=2",
            "evaluation.every=2",
            "evaluation.question_limit=4",
            "logging.wandb.enabled=true",
        ],
    )
    plans.append(with_algorithm(large, "on_policy"))
    atomic_json(args.output / "plan.json", [job.to_dict() for job in plans])
    if args.wait_pid is not None:
        identity = process_identity(args.wait_pid)
        if identity is not None:
            atomic_json(args.output / "wait.json", {"pid": args.wait_pid, "identity": identity})
            while process_identity(args.wait_pid) == identity:
                time.sleep(5)
    completed = []
    for index, job in enumerate(plans):
        atomic_json(
            args.output / "status.json",
            {"state": "running", "stage": job.name, "completed": completed},
        )
        try:
            result = run_job(
                job,
                backend=args.backend,
                directory=args.output / f"{index:02d}-{job.name}",
                lease_root=args.lease_root,
                borrowed_slot=args.borrowed_slot,
            )
        except Exception as error:
            atomic_json(
                args.output / "status.json",
                {
                    "state": "failed",
                    "stage": job.name,
                    "completed": completed,
                    "error": str(error),
                },
            )
            raise
        completed.append(
            {
                "run": job.name,
                "optimizer_updates": result["optimizer_updates"],
                "passed": result["passed"],
            }
        )
    atomic_json(args.output / "status.json", {"state": "completed", "completed": completed})


if __name__ == "__main__":
    main()
