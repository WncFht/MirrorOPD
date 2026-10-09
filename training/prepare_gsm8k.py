"""Materialize GSM8K train/test parquet files for the verl runner.

The test split is used directly by verl's validation loader.  There is no
additional validation split: every evaluation point is a fixed, untouched
GSM8K test set, which makes curves comparable across outer steps.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def answer_from_solution(solution: str) -> str:
    match = re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", solution)
    if not match:
        raise ValueError("GSM8K answer has no #### marker")
    return match[-1].replace(",", "")


def convert(split: str, output: Path) -> None:
    rows = []
    for index, example in enumerate(load_dataset("openai/gsm8k", "main", split=split)):
        question = example["question"].strip()
        prompt = (
            "Solve this math word problem. Show your reasoning, then put the final "
            "numeric answer on the last line in exactly this format: #### <number>.\n\n"
            f"Problem: {question}"
        )
        rows.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_from_solution(example["answer"]),
                },
                "extra_info": {"index": index, "split": split},
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"{split}: {len(rows)} rows -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    convert("train", args.output_dir / "train.parquet")
    convert("test", args.output_dir / "test.parquet")


if __name__ == "__main__":
    main()
