#!/usr/bin/env python3
"""Exact prefix-level reverse-KL and Top-K scalar estimators for a rollout.

The rollout is sampled from the student. For every generated token we evaluate
the full student and teacher vocabularies, then compute the deterministic
Top-K head, its tail bias, and the one-sampled-token tail correction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_model(name: str, device: str, quantization: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    kwargs = {"torch_dtype": dtype, "device_map": device}
    if device.startswith("cuda") and quantization:
        if quantization not in {"4bit", "8bit"}:
            raise ValueError("quantization must be 4bit or 8bit")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model


@torch.inference_mode()
def sequence_logits(model, prompt_ids: list[int], generated_ids: list[int], device: str):
    if not generated_ids:
        return torch.empty((0, model.config.vocab_size), dtype=torch.float32)
    # Omitting the final generated token makes the last n logits exactly the
    # distributions that predict generated_ids[0:n]. logits_to_keep avoids
    # materialising logits for prompt positions.
    ids = torch.tensor([prompt_ids + generated_ids[:-1]], dtype=torch.long, device=device)
    out = model(input_ids=ids, use_cache=False, logits_to_keep=len(generated_ids))
    return out.logits[0].float().cpu()


def write_rows(path: Path, rows: list[dict]):
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
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-generated-tokens", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    examples = [json.loads(line) for line in Path(args.rollouts).read_text().splitlines() if line.strip()]
    if args.max_examples:
        examples = examples[:args.max_examples]
    quantization = None if args.quantization == "none" else args.quantization
    student = load_model(args.student, args.device, quantization)
    teacher = load_model(args.teacher, args.device, quantization)
    # Keep only aggregate data locally; raw per-prefix data stays on archbox.
    sums = {k: [] for k in range(33)}
    prefix_count = 0
    example_rows = []
    for ex_idx, ex in enumerate(examples):
        gen = ex["generated_token_ids"]
        if args.max_generated_tokens:
            gen = gen[: args.max_generated_tokens]
        p_logits = sequence_logits(student, ex["prompt_token_ids"], gen, args.device)
        q_logits = sequence_logits(teacher, ex["prompt_token_ids"], gen, args.device)
        if p_logits.shape != q_logits.shape:
            raise RuntimeError(f"shape mismatch at example {ex_idx}: {p_logits.shape} vs {q_logits.shape}")
        p_log = torch.log_softmax(p_logits, dim=-1)
        q_log = torch.log_softmax(q_logits, dim=-1)
        p = p_log.exp()
        log_ratio = p_log - q_log
        token_ids = torch.tensor(gen, dtype=torch.long)
        for t in range(len(gen)):
            pt, qt, lt, tok = p[t], q_log[t], log_ratio[t], int(token_ids[t])
            true_kl = float((pt * lt).sum())
            entropy = float(-(pt * p_log[t]).sum())
            _, top_indices = torch.topk(pt, k=32)
            rank = int((pt > pt[tok]).sum()) + 1
            row_base = {"example_id": ex_idx, "step": t, "true_kl": true_kl,
                        "entropy_nats": entropy, "sampled_prob": float(pt[tok]),
                        "sampled_rank": rank, "sampled_in_top32": rank <= 32}
            total_second = float((pt * lt.square()).sum())
            head = 0.0
            head_second = 0.0
            mass = 0.0
            teacher_head_mass = 0.0
            for k in range(33):
                if k:
                    idx = int(top_indices[k - 1])
                    head += float(pt[idx] * lt[idx])
                    head_second += float(pt[idx] * lt[idx] * lt[idx])
                    mass += float(pt[idx])
                    teacher_head_mass += float(q_log[t, idx].exp())
                tail_mean = true_kl - head
                tail_second = total_second - head_second
                variance = max(0.0, tail_second - tail_mean * tail_mean)
                correction = float(lt[tok]) if rank > k else 0.0
                corrected_observed = head + correction
                # The normalized Top-K objective renormalizes both policies on
                # the same student-defined support S_K.  It is undefined at K=0.
                normalized = (head / mass + np.log(teacher_head_mass / mass)) if k else float("nan")
                sums[k].append((head, true_kl, variance, corrected_observed, mass, normalized))
                row_base[f"m{k}"] = mass
                row_base[f"trunc_bias{k}"] = head - true_kl
                row_base[f"corr_observed_error{k}"] = corrected_observed - true_kl
                row_base[f"corr_variance{k}"] = variance
                row_base[f"normalized_kl{k}"] = normalized
            # Store only a compact audit sample of token rows.
            if prefix_count < 2000:
                example_rows.append(row_base)
            prefix_count += 1
        del p_logits, q_logits, p_log, q_log, p, log_ratio
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
        if (ex_idx + 1) % 25 == 0:
            print(f"scored {ex_idx + 1}/{len(examples)} examples, {prefix_count} prefixes", flush=True)
    summary = []
    for k in range(33):
        arr = np.asarray(sums[k], dtype=np.float64)
        head, true, variance, corrected, mass, normalized = arr.T
        trunc_error = head - true
        # Conditional MSE averages over prefixes; correction is unbiased
        # conditionally, while the observed one-token error is retained too.
        normalized_error = normalized - true
        if k == 0:
            mean_normalized = bias_normalized = rmse_normalized = float("nan")
        else:
            mean_normalized = float(np.mean(normalized))
            bias_normalized = float(np.mean(normalized_error))
            rmse_normalized = float(np.sqrt(np.mean(normalized_error ** 2)))
        summary.append({"top_k": k, "prefixes": len(arr),
            "mean_topk_mass": float(mass.mean()),
            "mean_true_kl": float(true.mean()),
            "bias_trunc": float(trunc_error.mean()),
            "variance_trunc": 0.0,
            "rmse_trunc": float(np.sqrt(np.mean(trunc_error ** 2))),
            "bias_corrected_theory": 0.0,
            "variance_corrected": float(variance.mean()),
            "rmse_corrected_theory": float(np.sqrt(np.mean(variance))),
            "bias_corrected_observed": float((corrected - true).mean()),
            "rmse_corrected_observed": float(np.sqrt(np.mean((corrected - true) ** 2))),
            "mean_normalized_kl": mean_normalized,
            "bias_normalized": bias_normalized,
            "variance_normalized": 0.0,
            "rmse_normalized": rmse_normalized,
        })
    write_rows(out / "kl_summary_by_k.csv", summary)
    write_rows(out / "prefix_audit_sample.csv", example_rows)
    (out / "metadata.json").write_text(json.dumps({
        "student": args.student, "teacher": args.teacher,
        "rollouts": str(args.rollouts), "n_examples": len(examples),
        "n_prefixes": prefix_count, "temperature": 1.0,
        "definition": "reverse KL KL(student||teacher), exact full-vocabulary reference; sampled token is the student rollout token",
        "scoring_quantization": args.quantization,
        "topk": "student probability top-K; corrected estimator adds log student/teacher ratio only when sampled token is outside the head; normalized estimator renormalizes both policies on the same student-defined support",
    }, indent=2))
    print(json.dumps({"n_examples": len(examples), "n_prefixes": prefix_count}, indent=2))


if __name__ == "__main__":
    main()
