#!/usr/bin/env python3
"""Compute exact full-vocabulary Top-K mass along saved GSM8K rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(x) for x in Path(args.rollouts).read_text().splitlines() if x.strip()
    ]
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    kw = {"torch_dtype": dtype, "device_map": args.device}
    if args.quantization != "none":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.quantization == "4bit",
            load_in_8bit=args.quantization == "8bit",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    prefix = []
    prompt = []
    with torch.inference_mode():
        for i, ex in enumerate(rows):
            gen = ex["generated_token_ids"]
            n = len(gen)
            if not n:
                continue
            ids = torch.tensor(
                [ex["prompt_token_ids"] + gen[:-1]],
                dtype=torch.long,
                device=args.device,
            )
            logits = (
                model(input_ids=ids, use_cache=False, logits_to_keep=n)
                .logits[0]
                .float()
            )
            probs = torch.softmax(logits, dim=-1)
            topvals, topidx = torch.topk(probs, k=32, dim=-1)
            # Compute the tail directly.  ``1 - M32`` loses all significant
            # digits when the head already carries nearly unit mass.
            tail_mask = torch.ones_like(probs, dtype=torch.bool)
            tail_mask.scatter_(1, topidx, False)
            tail32 = probs.masked_select(tail_mask).view(probs.shape[0], -1).sum(dim=1)
            own = []
            for t in range(n):
                vals = topvals[t]
                item = {"example_id": i, "step": t}
                for k in (1, 2, 4, 8, 16, 32):
                    item[f"m{k}"] = float(vals[:k].sum())
                item["tail32"] = max(float(tail32[t]), 1e-300)
                prefix.append(item)
                own.append(item)
            prompt.append(
                {
                    "example_id": i,
                    "generated_tokens": n,
                    **{
                        f"mean_m{k}": sum(x[f"m{k}"] for x in own) / n
                        for k in (1, 2, 4, 8, 16, 32)
                    },
                    **{
                        f"median_m{k}": sorted(x[f"m{k}"] for x in own)[n // 2]
                        for k in (1, 2, 4, 8, 16, 32)
                    },
                }
            )
            if (i + 1) % 25 == 0:
                print(
                    f"scored {i + 1}/{len(rows)} examples, {len(prefix)} prefixes",
                    flush=True,
                )
    for name, data in [("prefix_mk.csv", prefix), ("prompt_mk.csv", prompt)]:
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "n_examples": len(rows),
                "n_prefixes": len(prefix),
                "quantization": args.quantization,
                "definition": "exact full-vocabulary forward pass; M_K is Student/model top-K next-token mass",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
