"""Complete native math evaluation of one checkpoint on one owned GPU replica."""

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from training.gpu_exclusive import register_gpu_owner


def initialize_cpu_scorer():
    """Prevent scoring workers from opening CUDA contexts."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def score_output(row):
    """Reuse the exact boxed-answer and Math-Verify scorer of the math backend."""
    from math_verify.errors import TimeoutException
    from verl.utils.reward_score.math_verify import _verify_in_subprocess as verify_math_output
    from verl.utils.reward_score.ttrl_math import compute_score as boxed_score

    result = boxed_score(row["response"], row["ground_truth"])
    status = "valid"
    if result["format_score"] != 1.0:
        # This worker already has its own main thread and signal handlers.
        # Reuse the backend's verifier directly rather than nesting its
        # process pool inside this process pool.
        try:
            correct = bool(
                verify_math_output(r"\boxed{" + row["ground_truth"] + "}", row["response"][-2000:])
            )
        except TimeoutException:
            correct = False
            status = "timeout"
        result = {"acc": correct, "format_score": 0.0, "score": float(correct)}
    return row | {
        "correct": bool(result["acc"]),
        "format_score": float(result["format_score"]),
        "score": float(result["score"]),
        "score_status": status,
    }


def main():
    """Generate and score the assigned quarter of all 570 benchmark questions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owners", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    if os.environ["CUDA_VISIBLE_DEVICES"] != str(args.rank) or args.rank not in range(4):
        raise RuntimeError("Each evaluation replica must own one physical GPU in 0--3")
    import pandas as pd
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = []
    source_hashes = {}
    for task, expected in (("MATH-500", 500), ("AMC23", 40), ("AIME25", 30)):
        path = args.repo.parent / "MD-OPD/data/math" / task / "test.parquet"
        frame = pd.read_parquet(path)
        if len(frame) != expected:
            raise ValueError(f"{task} must contain {expected} questions")
        source_hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, item in frame.iterrows():
            uid = f"{task}/{index}"
            prompt = tokenizer.apply_chat_template(
                list(item.prompt), tokenize=False, add_generation_prompt=True
            )
            rows.append(
                {
                    "uid": uid,
                    "task": task,
                    "index": int(index),
                    "data_source": str(item.data_source),
                    "ground_truth": str(item.reward_model["ground_truth"]),
                    "prompt": prompt,
                }
            )
    assigned = rows[args.rank :: 4]
    raw_path = args.output / f"rank-{args.rank}-generations.jsonl"
    if raw_path.exists():
        generated = [json.loads(line) for line in raw_path.read_text().splitlines()]
    else:
        generated = generate_outputs(args, assigned)
        raw_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated)
        )
    with (
        ProcessPoolExecutor(
            max_workers=4,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_cpu_scorer,
        ) as executor,
        (args.output / f"rank-{args.rank}.jsonl").open("w") as stream,
    ):
        scored = []
        for row in executor.map(score_output, generated):
            scored.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    (args.output / f"rank-{args.rank}-metadata.json").write_text(
        json.dumps(
            {
                "examples": len(scored),
                "data_sha256": source_hashes,
                "greedy": True,
                "max_response_tokens": 7168,
                "max_context_tokens": 9216,
                "physical_gpu": args.rank,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rank": args.rank, "examples_scored": len(scored)}), flush=True)


def generate_outputs(args, assigned):
    """Persist generated tokens before entering any CPU scoring pool."""
    import torch
    from vllm import LLM, SamplingParams

    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    register_gpu_owner(args.owners, args.rank)

    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=9216,
        max_num_batched_tokens=9216,
        max_num_seqs=32,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        seed=0,
        disable_log_stats=True,
    )
    outputs = engine.generate(
        [row["prompt"] for row in assigned],
        SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=7168),
        use_tqdm=False,
    )
    generated = []
    for row, output in zip(assigned, outputs, strict=True):
        response = output.outputs[0]
        generated.append(
            row
            | {
                "response": response.text,
                "response_token_ids": response.token_ids,
                "response_tokens": len(response.token_ids),
                "prompt_tokens": len(output.prompt_token_ids),
                "finish_reason": response.finish_reason,
                "budget_exhausted": response.finish_reason == "length",
            }
        )
    return generated


if __name__ == "__main__":
    main()
