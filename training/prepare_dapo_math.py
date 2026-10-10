"""Prepare DAPO-Math-17k and MATH-500 in the runner's aligned parquet schema.

The DAPO release contains 100 copies of each problem.  We preserve the first
occurrence in source order, remove only the release's answer-format wrapper,
and keep the original answer in ``reward_model.ground_truth``.  The raw files
are downloaded to a cache outside the repository; only the compact aligned
outputs and their manifest are experiment inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

_DAPO_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
_DAPO_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
_MATH_REPO = "HuggingFaceH4/MATH-500"
_MATH_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of "
    "the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def _download(repo: str, filename: str, revision: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )


def _object(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _aligned(content: str, answer: Any, source: str, index: int, split: str, **extra: Any) -> dict:
    info = {"index": f"{source}-{index}", "source": source, "split": split, **extra}
    return {
        "prompt": [{"role": "user", "content": content.strip()}],
        "data_source": source,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": info,
    }


def build_dapo(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pq.ParquetFile(path)
    seen: set[str] = set()
    rows: list[dict] = []
    for batch in raw.iter_batches(
        batch_size=2048, columns=["prompt", "reward_model", "extra_info"]
    ):
        for record in batch.to_pylist():
            info = _object(record["extra_info"])
            if not isinstance(info, dict) or "index" not in info:
                raise ValueError("DAPO row has no stable extra_info.index")
            source_index = str(info["index"])
            if source_index in seen:
                continue
            seen.add(source_index)
            prompt = _object(record["prompt"])
            content = prompt[0]["content"]
            content = content.removeprefix(_PREFIX)
            content = content.removesuffix(_SUFFIX)
            reward = _object(record["reward_model"])
            answer = reward["ground_truth"] if isinstance(reward, dict) else reward
            rows.append(
                _aligned(content, answer, "DAPO", len(rows), "train", dapo_index=source_index)
            )
    frame = pd.DataFrame(rows)
    metadata = {
        "raw_rows": raw.metadata.num_rows,
        "unique_rows": len(frame),
        "dedup_key": "extra_info.index, first occurrence in source order",
        "wrapper_removed": True,
    }
    return frame, metadata


def build_math500(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                _aligned(
                    item["problem"],
                    item["answer"],
                    "MATH-500",
                    index,
                    "test",
                    level=item.get("level", ""),
                    subject=item.get("subject", ""),
                )
            )
    return pd.DataFrame(rows), {"raw_rows": len(rows), "unique_rows": len(rows)}


def _write(frame: pd.DataFrame, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    dapo_path = _download(_DAPO_REPO, "data/dapo-math-17k.parquet", _DAPO_REVISION, args.cache_dir)
    math_path = _download(_MATH_REPO, "test.jsonl", _MATH_REVISION, args.cache_dir)
    dapo, dapo_meta = build_dapo(dapo_path)
    math500, math_meta = build_math500(math_path)
    if len(dapo) != 17917 or len(math500) != 500:
        raise ValueError("Pinned source does not match the expected dataset sizes")
    easy = math500[
        math500.extra_info.map(lambda info: str(info["level"]) in ("1", "2", "Level 1", "Level 2"))
    ]
    train_questions = {row[0]["content"] for row in dapo.prompt}
    overlaps = [
        info["index"]
        for prompt, info in zip(math500.prompt, math500.extra_info, strict=True)
        if prompt[0]["content"] in train_questions
    ]
    test_questions = {row[0]["content"] for row in math500.prompt}
    clean = dapo[~dapo.prompt.map(lambda prompt: prompt[0]["content"] in test_questions)]
    train_out = args.output_dir / "dapo-math-17k" / "train.parquet"
    test_out = args.output_dir / "math-500" / "test.parquet"
    manifest = {
        "dataset": "DAPO-Math-17k + MATH-500",
        "dapo_revision": _DAPO_REVISION,
        "math500_revision": _MATH_REVISION,
        "selection": "DAPO unique problems for train; MATH-500 reference test; no AMC/AIME",
        "train": {**dapo_meta, "sha256": _write(dapo, train_out), "path": str(train_out)},
        "test": {**math_meta, "sha256": _write(math500, test_out), "path": str(test_out)},
        "easy_test": {
            "count": len(easy),
            "sha256": _write(easy, args.output_dir / "math-500-easy" / "test.parquet"),
        },
        "exact_train_test_prompt_overlap": overlaps,
        "clean_train": {
            "count": len(clean),
            "sha256": _write(clean, args.output_dir / "dapo-math-17k-clean" / "train.parquet"),
            "policy": "exclude exact MATH-500 question matches; retain official source separately",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (train_out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "dapo-math-17k-clean" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    gsm_path = args.output_dir.parent / "gsm8k-full/test.parquet"
    if gsm_path.exists():
        combined = pd.concat([easy, pd.read_parquet(gsm_path)], ignore_index=True)
        combined["extra_info"] = combined.extra_info.map(
            lambda info: {
                "index": str(info["index"]),
                "split": "test",
                "level": str(info.get("level", "")),
            }
        )
        _write(combined, args.output_dir / "simple-tests/test.parquet")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
