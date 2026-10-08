#!/usr/bin/env python3
"""Collect prefix-level M_K and entropy statistics from Qwen3 GSM8K rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def numeric_answer(text: str) -> str | None:
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return matches[-1].replace(",", "") if matches else None


def extract_prediction(text: str) -> str | None:
    matches = re.findall(
        r"(?:####|answer is|answer:|=)\s*([-+]?\d[\d,]*(?:\.\d+)?)", text, flags=re.IGNORECASE
    )
    return matches[-1].replace(",", "") if matches else None


@torch.inference_mode()
def rollout(args, output: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device
    )
    model.eval()
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    dataset = dataset.select(range(min(args.n, len(dataset))))
    rows = []
    summaries = []
    for example_id, example in enumerate(dataset):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = generated.sequences[0, inputs.input_ids.shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        target = numeric_answer(example["answer"])
        prediction = extract_prediction(generated_text)
        prompt_rows = []
        for step, (token_id, score) in enumerate(
            zip(generated_ids.tolist(), generated.scores)
        ):
            # Keep the probability-mass calculation in float64: Qwen's logits
            # are often so peaked that float32 rounds M32 to exactly one.
            logits = score[0].double()
            probs = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probs, k=min(32, probs.numel()))
            top_mask = torch.zeros_like(probs, dtype=torch.bool)
            top_mask[top_indices] = True
            tail32_mass = (
                probs[~top_mask].sum().clamp_min(torch.finfo(probs.dtype).tiny)
            )
            token_prob = probs[token_id]
            rank = int((probs > token_prob).sum().item()) + 1
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum().item())
            item = {
                "example_id": example_id,
                "step": step,
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "m1": float(top_values[:1].sum()),
                "m4": float(top_values[:4].sum()),
                "m8": float(top_values[:8].sum()),
                "m16": float(top_values[:16].sum()),
                "m32": float(1.0 - tail32_mass.item()),
                "tail32_mass": float(tail32_mass.item()),
                "entropy_nats": entropy,
                "sampled_prob": float(token_prob.item()),
                "sampled_rank": rank,
                "sampled_in_top32": rank <= 32,
            }
            rows.append(item)
            prompt_rows.append(item)
        summaries.append(
            {
                "example_id": example_id,
                "question": example["question"],
                "target": target,
                "prediction": prediction,
                "exact_numeric_match": target is not None and target == prediction,
                "generated_tokens": len(prompt_rows),
                "mean_m32": float(np.mean([r["m32"] for r in prompt_rows]))
                if prompt_rows
                else math.nan,
                "median_m32": float(np.median([r["m32"] for r in prompt_rows]))
                if prompt_rows
                else math.nan,
                "mean_entropy_nats": float(
                    np.mean([r["entropy_nats"] for r in prompt_rows])
                )
                if prompt_rows
                else math.nan,
            }
        )
        if (example_id + 1) % args.log_every == 0:
            print(f"processed {example_id + 1}/{len(dataset)} examples", flush=True)
    (output / "prefix_metrics.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    with (output / "example_summaries.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "split": args.split,
                "n_examples": len(dataset),
                "temperature": args.temperature,
                "top_p": 1.0,
                "max_new_tokens": args.max_new_tokens,
                "device": device,
                "dtype": str(dtype),
                "n_prefixes": len(rows),
                "definition": "M_K is the probability mass of the exact top-K tokens of the temperature-scaled sampling distribution at each generated prefix.",
            },
            indent=2,
        )
    )
    return rows, summaries


def plot(rows, output: Path):
    m32 = np.asarray([row["m32"] for row in rows])
    tail32 = np.asarray([row["tail32_mass"] for row in rows])
    entropy = np.asarray([row["entropy_nats"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    tail_bins = (
        np.geomspace(tail32.min(), tail32.max(), 40)
        if tail32.max() > tail32.min()
        else 10
    )
    axes[0].hist(tail32, bins=tail_bins, color="tab:blue", alpha=0.85)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("tail mass $1-M_{32}(\\pi)$")
    axes[0].set_ylabel("prefix count")
    axes[0].set_title("M32 distribution (shown as tail mass)")
    axes[1].hist(entropy, bins=40, color="tab:green", alpha=0.85)
    axes[1].set_xlabel("entropy (nats)")
    axes[1].set_ylabel("prefix count")
    axes[1].set_title("Entropy distribution")
    axes[2].scatter(entropy, tail32, s=3, alpha=0.12, rasterized=True)
    bins = np.linspace(entropy.min(), entropy.max(), 25)
    centers, means, lows, highs = [], [], [], []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (entropy >= left) & (entropy < right)
        if mask.sum() < 5:
            continue
        centers.append((left + right) / 2)
        means.append(np.mean(tail32[mask]))
        lows.append(np.quantile(tail32[mask], 0.1))
        highs.append(np.quantile(tail32[mask], 0.9))
    axes[2].plot(centers, means, "o-", color="tab:red", label="bin mean")
    axes[2].fill_between(
        centers, lows, highs, color="tab:red", alpha=0.2, label="10--90%"
    )
    axes[2].set_xlabel("entropy (nats)")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("tail mass $1-M_{32}(\\pi)$")
    axes[2].set_title("M32 tail versus entropy")
    axes[2].legend()
    fig.suptitle("Qwen3-1.7B GSM8K prefix-level token distributions")
    fig.tight_layout()
    fig.savefig(output / "m32_entropy_distribution.png", dpi=220)
    fig.savefig(output / "m32_entropy_distribution.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--log-every", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, _ = rollout(args, output)
    plot(rows, output)


if __name__ == "__main__":
    main()
