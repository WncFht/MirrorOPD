"""Independent CPU checkpoint readout on early, versioned diagnostic prefixes."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    """Compare literal HF BF16/FP32 forwards without FSDP or a CUDA context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--informative", action="store_true")
    args = parser.parse_args()
    if os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("This independent probe must run with CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(8)
    rows = [
        json.loads(line) for line in (args.cache / "trajectories.jsonl").read_text().splitlines()
    ]
    selected = sorted(
        (position, row["rank"], row["sample_index"])
        for row in rows
        for position in row["selected_positions"]
    )[:8]
    examples = []
    ranks = (
        list(range(4)) if args.informative else sorted({rank for position, rank, index in selected})
    )
    for rank in ranks:
        artifact = torch.load(
            args.cache / f"rank-{rank}.pt", weights_only=False, map_location="cpu"
        )
        if args.informative:
            candidates = []
            for index, sequence in enumerate(artifact["sequences"]):
                for row, position in enumerate(sequence["selected"].tolist()):
                    if position <= 512:
                        qlog, teacher = sequence["q_logp_bf16"][row], sequence["teacher_logp"][row]
                        initial_kl = float((qlog.exp() * (qlog - teacher)).sum())
                        if initial_kl >= 0.02:
                            candidates.append((position, rank, rank * 16 + index))
            selected = candidates
        for position, owner, index in selected:
            if owner != rank:
                continue
            sequence = artifact["sequences"][index - rank * 16]
            row = sequence["selected"].tolist().index(position)
            examples.append(
                {
                    "sample_index": index,
                    "position": position,
                    "ids": sequence["ids"][: sequence["start"] + position + 1].unsqueeze(0).clone(),
                    "teacher_logp": sequence["teacher_logp"][row].clone(),
                    "vocab": artifact["vocab"],
                }
            )
        del artifact
        gc.collect()
    if args.informative:
        examples = sorted(
            examples, key=lambda example: (example["position"], example["sample_index"])
        )[:8]
    report = {
        "device": "CPU only",
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint),
        "examples": len(examples),
        "selection": "Eight earliest positions with initial anchor-teacher KL >=0.02 and position <=512"
        if args.informative
        else "Eight earliest positions from the original fixed diagnostic reservoir",
        "conditions": {},
    }
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=dtype, attn_implementation="sdpa", local_files_only=True
        ).eval()
        scores = []
        for example in examples:
            with torch.no_grad():
                hidden = model.model(
                    example["ids"], attention_mask=torch.ones_like(example["ids"]), use_cache=False
                ).last_hidden_state[:, -1]
                logp = model.lm_head(hidden)[0, : example["vocab"]].float().log_softmax(-1)
                teacher = example["teacher_logp"]
                score = {
                    "sample_index": example["sample_index"],
                    "position": example["position"],
                    "prefix_tokens": example["ids"].shape[-1],
                    "student_teacher_kl": float((logp.exp() * (logp - teacher)).double().sum()),
                    "teacher_student_kl": float((teacher.exp() * (teacher - logp)).double().sum()),
                }
                if name == "bf16":
                    example["bf16_logp"] = logp.clone()
                else:
                    qlog = example["bf16_logp"]
                    score["bf16_fp32_kl"] = float((qlog.exp() * (qlog - logp)).double().sum())
            scores.append(score)
        report["conditions"][name] = scores
        del model
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
