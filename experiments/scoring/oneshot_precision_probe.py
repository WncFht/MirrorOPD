"""Short single-GPU FSDP probe of the first-update KL precision effect."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from transformers import AutoModelForCausalLM, AutoTokenizer

from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.losses import compute_mirror_opd_token_loss


def measure_rounding(root):
    """Count original weights that can cross a BF16 bin after a 1e-6 change."""
    report = {}
    path = root.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.12.mlp.down_proj.weight",
        ):
            weights = handle.get_tensor(key)
            master = weights.float()
            changed = ((master + 1e-6).bfloat16() != weights.bfloat16()) | (
                (master - 1e-6).bfloat16() != weights.bfloat16()
            )
            report[key] = {
                "elements": weights.numel(),
                "source_dtype": str(weights.dtype),
                "fraction_bf16_can_change_for_abs_delta_at_most_1e-6": float(
                    changed.float().mean()
                ),
                "max_abs_weight": float(master.abs().max()),
            }
    (root / "results/oneshot-inner-audit/bf16-rounding-bound.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main():
    """Compare zero-lr and small-lr updates on identical fixed short states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounding-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    (root / "results/oneshot-inner-audit").mkdir(parents=True, exist_ok=True)
    measure_rounding(root)
    if args.rounding_only:
        return
    output = root / "results/oneshot-inner-audit/precision-probe.json"
    scratch = root / "runs/oneshot-inner-audit"
    scratch.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method="file://" + str(scratch / f"nccl-{os.getpid()}"), rank=0, world_size=1
    )
    student_path = root.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(student_path)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Compute the sum of all positive integers from 1 to 100. Explain your reasoning.",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).cuda()
    initial = (
        AutoModelForCausalLM.from_pretrained(
            student_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        ids = initial.generate(
            prompt, max_new_tokens=128, do_sample=True, temperature=1.0, top_p=1.0, top_k=0
        )
    del initial
    gc.collect()
    torch.cuda.empty_cache()
    actions = ids.roll(-1, dims=-1)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    valid[:, prompt.shape[-1] - 1 : -1] = True
    report = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "seed": 42,
        "response_tokens": int(valid.sum()),
        "world_size": 1,
        "attention": "sdpa",
        "eta": 0.3,
        "topk": 16,
        "conditions": {},
    }
    for label, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        teacher = (
            AutoModelForCausalLM.from_pretrained(
                root.parent / "models/JustRL-DeepSeek-1.5B",
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
            .cuda()
            .eval()
        )
        with torch.no_grad():
            teacher_logits = teacher(ids, use_cache=False).logits[..., : len(tokenizer)]
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        student = (
            AutoModelForCausalLM.from_pretrained(
                student_path, torch_dtype=torch.float32, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        student.gradient_checkpointing_enable()
        student = FullyShardedDataParallel(
            student,
            device_id=0,
            mixed_precision=MixedPrecision(
                param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            ),
            use_orig_params=True,
        )
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=0.0,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=False,
        )
        with torch.no_grad():
            anchor_logits = student(ids, use_cache=False).logits[..., : len(tokenizer)]
            cache = build_mirror_opd_cache_from_full_logits(
                anchor_logits, teacher_logits, actions, valid, 16
            )
        del anchor_logits, teacher_logits
        records = []
        for index in range(5):
            student.train()
            optimizer.zero_grad()
            logits = student(ids, use_cache=False).logits[..., : len(tokenizer)]
            fields = compute_mirror_opd_token_loss(
                logits[valid],
                actions[valid],
                cache["mirror_anchor_ids"][valid],
                cache["mirror_anchor_head_log_probs"][valid],
                cache["mirror_anchor_tail_log_prob"][valid],
                cache["mirror_anchor_action_log_prob"][valid],
                cache["mirror_teacher_action_log_prob"][valid],
                cache["mirror_teacher_center"][valid],
                "mdpo",
                0.3,
            )
            fields["distillation_losses"].mean().backward()
            grad = student.clip_grad_norm_(2.0)
            records.append(
                {
                    "updates_before_measurement": index,
                    "student_center_mean": float(fields["mirror_student_center"].mean()),
                    "teacher_center_mean": float(fields["mirror_teacher_center"].mean()),
                    "current_gap_mean": float(
                        (
                            fields["mirror_sampled_log_ratio"] - fields["mirror_sampled_reward"]
                        ).mean()
                    ),
                    "grad_norm": float(grad),
                }
            )
            optimizer.param_groups[0]["lr"] = 0.0 if index == 0 else 1e-6
            optimizer.step()
            del logits, fields
        report["conditions"][label] = records
        del optimizer, student, cache
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
