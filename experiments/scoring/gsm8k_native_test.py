"""Evaluate a pretrained model with the exact training test prompts and scorer."""

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from mirroropd.training.gsm8k import score_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_parquet(args.test)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                list(row.prompt), tokenize=True, add_generation_prompt=True
            )
        }
        for row in data.itertuples()
    ]
    engine = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=2560,
        gpu_memory_utilization=0.8,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        enforce_eager=True,
        seed=0,
    )
    outputs = engine.generate(
        prompts, SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=2048)
    )
    records = []
    for row, output in zip(data.itertuples(), outputs, strict=True):
        answer = output.outputs[0]
        scores = score_response(answer.text, row.reward_model["ground_truth"])
        records.append(
            {
                "example_id": int(row.extra_info["index"]),
                **scores,
                "response_tokens": len(answer.token_ids),
                "budget_exhausted": float(answer.finish_reason == "length"),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per-example.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    fields = [key for key in records[0] if key != "example_id"]
    summary = {
        "model": args.model,
        "examples": len(records),
        "test": str(args.test),
        "max_new_tokens": 2048,
        "temperature": 0,
        "native_template": True,
        "means": {key: sum(r[key] for r in records) / len(records) for key in fields},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
