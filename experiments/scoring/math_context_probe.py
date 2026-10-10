"""Resumable native-template math evaluation and response-budget diagnostics."""

import argparse
import hashlib
import json
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from mirroropd.training.gsm8k import score_response
from training.math_reward import compute_score


def score(source, text, truth):
    if source == "openai/gsm8k":
        scores = score_response(text, truth)
        return {**scores, "format_score": scores["format_boxed"]}
    return compute_score(source, text, truth)


def summarize(records):
    result = {}
    groups = {
        "all": records,
        "level-1-2": [r for r in records if str(r["level"]) in ("1", "2", "Level 1", "Level 2")],
    }
    for name, rows in groups.items():
        if not rows:
            continue
        result[name] = {
            "count": len(rows),
            "accuracy": float(np.mean([r["scores"]["score"] for r in rows])),
            "boxed_rate": float(np.mean([r["scores"]["format_score"] for r in rows])),
            "context_exhausted": float(np.mean([r["finish_reason"] == "length" for r in rows])),
            "prompt_max": max(r["prompt_tokens"] for r in rows),
            "response_quantiles": dict(
                zip(
                    ("p50", "p90", "p95", "p99", "max"),
                    np.quantile(
                        [r["response_tokens"] for r in rows], (0.5, 0.9, 0.95, 0.99, 1)
                    ).tolist(),
                    strict=True,
                )
            ),
            "budget_accuracy": {
                str(b): float(np.mean([r["budget_scores"][str(b)] for r in rows]))
                for b in (512, 1024, 2048, 4096)
            },
            "budget_overflow_rate": {
                str(b): float(np.mean([r["response_tokens"] > b for r in rows]))
                for b in (512, 1024, 2048, 4096)
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--response-budget", type=int)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    if args.context > config.max_position_embeddings:
        raise ValueError("Context exceeds native model position limit")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "model_commit": config._commit_hash,  # noqa: SLF001
        "context": args.context,
        "temperature": args.temperature,
        "seed": 42,
        "limit": args.limit,
        "response_budget": args.response_budget,
        "native_template": True,
        "data": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data},
        "packages": {p: version(p) for p in ("torch", "transformers", "vllm", "math-verify")},
        "verifier": "One-Shot-OPD ttrl_mathverify",
        "verifier_source_sha256": {
            str(
                p.relative_to(
                    Path(find_spec("verl.utils.reward_score.ttrl_mathverify").origin).parent
                )
            ): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(find_spec("verl.utils.reward_score.ttrl_mathverify").origin).parent.rglob(
                "*.py"
            )
        },
    }
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Resume inputs or settings differ")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    engine = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.context,
        gpu_memory_utilization=0.7,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        enforce_eager=True,
        seed=42,
    )
    for path in args.data:
        frame = pd.read_parquet(path)
        if args.limit:
            frame = frame.sample(n=min(args.limit, len(frame)), random_state=42)
        output = args.output / f"{path.parent.name}-{path.stem}.jsonl"
        records = (
            [json.loads(line) for line in output.read_text().splitlines()]
            if output.exists()
            else []
        )
        if len(records) > len(frame):
            raise ValueError("Resume output exceeds input length")
        for record, row in zip(records, frame.itertuples(), strict=False):
            if record["index"] != row.extra_info["index"]:
                raise ValueError("Resume output order differs from input")
        with output.open("a", encoding="utf-8") as handle:
            for start in range(len(records), len(frame), args.chunk_size):
                batch = list(frame.iloc[start : start + args.chunk_size].itertuples())
                prompts = [
                    tokenizer.apply_chat_template(
                        list(r.prompt), tokenize=True, add_generation_prompt=True
                    )
                    for r in batch
                ]
                if any(len(p) >= args.context for p in prompts):
                    raise ValueError("Prompt fills context; refusing silent truncation")
                params = [
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=1,
                        top_k=-1,
                        max_tokens=min(args.response_budget or args.context, args.context - len(p)),
                    )
                    for p in prompts
                ]
                generated = engine.generate([{"prompt_token_ids": p} for p in prompts], params)
                for row, prompt, result in zip(batch, prompts, generated, strict=True):
                    answer = result.outputs[0]
                    record = {
                        "index": row.extra_info["index"],
                        "level": row.extra_info.get("level"),
                        "prompt_tokens": len(prompt),
                        "response_tokens": len(answer.token_ids),
                        "finish_reason": answer.finish_reason,
                        "response": answer.text,
                        "scores": score(
                            row.data_source, answer.text, row.reward_model["ground_truth"]
                        ),
                        "budget_scores": {},
                    }
                    for budget in (512, 1024, 2048, 4096):
                        prefix = tokenizer.decode(
                            answer.token_ids[:budget], skip_special_tokens=True
                        )
                        record["budget_scores"][str(budget)] = score(
                            row.data_source, prefix, row.reward_model["ground_truth"]
                        )["score"]
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                handle.flush()
                print(f"{path}: {len(records)}/{len(frame)}", flush=True)
                (args.output / f"{output.stem}-summary.json").write_text(
                    json.dumps(summarize(records), indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
