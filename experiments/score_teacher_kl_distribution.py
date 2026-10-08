#!/usr/bin/env python3
"""Compute full-vocabulary K_k^T = KL(student || teacher) distributions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_model(name: str, device: str, quantization: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    kw = {"torch_dtype": dtype, "device_map": device}
    if device.startswith("cuda") and quantization != "none":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return model


@torch.inference_mode()
def sequence_logits(model, prompt_ids, generated_ids, device):
    if not generated_ids:
        return torch.empty((0, model.config.vocab_size), dtype=torch.float32)
    ids = torch.tensor(
        [prompt_ids + generated_ids[:-1]], dtype=torch.long, device=device
    )
    return (
        model(input_ids=ids, use_cache=False, logits_to_keep=len(generated_ids))
        .logits[0]
        .float()
    )


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--student", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", default="Qwen/Qwen3-4B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    examples = [
        json.loads(x) for x in Path(args.rollouts).read_text().splitlines() if x.strip()
    ]
    quant = None if args.quantization == "none" else args.quantization
    student = load_model(args.student, args.device, quant)
    teacher = load_model(args.teacher, args.device, quant)
    prefix_rows = []
    prompt_rows = []
    with torch.inference_mode():
        for ex_idx, ex in enumerate(examples):
            gen = ex["generated_token_ids"]
            p_logits = sequence_logits(
                student, ex["prompt_token_ids"], gen, args.device
            )
            q_logits = sequence_logits(
                teacher, ex["prompt_token_ids"], gen, args.device
            )
            p_log = torch.log_softmax(p_logits, dim=-1)
            q_log = torch.log_softmax(q_logits, dim=-1)
            p = p_log.exp()
            kl = (p * (p_log - q_log)).sum(dim=-1)
            entropy = -(p * p_log).sum(dim=-1)
            vals = kl.detach().cpu().numpy().astype(np.float64)
            ents = entropy.detach().cpu().numpy().astype(np.float64)
            for t, (value, ent) in enumerate(zip(vals, ents)):
                prefix_rows.append(
                    {
                        "example_id": ex_idx,
                        "step": t,
                        "generated_tokens": len(gen),
                        "kt_kl": float(value),
                        "student_entropy_nats": float(ent),
                    }
                )
            prompt_rows.append(
                {
                    "example_id": ex_idx,
                    "generated_tokens": len(gen),
                    "mean_kt_kl": float(vals.mean()),
                    "median_kt_kl": float(np.median(vals)),
                    "p90_kt_kl": float(np.quantile(vals, 0.90)),
                    "max_kt_kl": float(vals.max()),
                    "mean_entropy_nats": float(ents.mean()),
                }
            )
            del p_logits, q_logits, p_log, q_log, p, kl, entropy
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if (ex_idx + 1) % 25 == 0:
                print(
                    f"scored {ex_idx + 1}/{len(examples)} examples, {len(prefix_rows)} prefixes",
                    flush=True,
                )
    write_csv(out / "kt_prefix.csv", prefix_rows)
    write_csv(out / "kt_prompt.csv", prompt_rows)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "student": args.student,
                "teacher": args.teacher,
                "rollouts": str(args.rollouts),
                "n_examples": len(examples),
                "n_prefixes": len(prefix_rows),
                "quantization": args.quantization,
                "definition": "K_k^T(s)=KL(pi_k(.|s)||pi_T(.|s)), exact full-vocabulary value on NF4-scored models",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
