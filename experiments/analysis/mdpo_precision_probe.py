"""Measure actual AdamW parameter movement under the production MDPO score.

Controlled real-model probe, not a benchmark: identical prompt/action and
eta/lr for BF16 versus FP32 parameter storage. Forward uses BF16 autocast in
both cases. Report zero-update drift separately from optimization movement.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mirroropd.training.cache import build_mirror_anchor_cache, complete_mirror_teacher_cache
from mirroropd.training.losses import compute_mirror_opd_token_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    ids = tok(
        "Janet has 16 eggs. She eats 3 and uses 4. She sells", return_tensors="pt"
    ).input_ids.cuda()
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-1.7B-Instruct", torch_dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        teacher_logits = teacher(ids).logits[:, :-1].float()
    del teacher
    torch.cuda.empty_cache()
    out = {}
    for dtype in (torch.bfloat16, torch.float32):
        model = (
            AutoModelForCausalLM.from_pretrained(
                "HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype=dtype
            )
            .cuda()
            .eval()
        )
        actions = ids[:, 1:]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids).logits[:, :-1]
            cache = build_mirror_anchor_cache(
                logits, actions, torch.ones_like(actions, dtype=torch.bool), 16
            )
            cache = complete_mirror_teacher_cache(teacher_logits, cache)
        before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        optim = torch.optim.AdamW(model.parameters(), lr=3e-7, weight_decay=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zero_logits = model(ids).logits[:, :-1]
        zero_logp = torch.log_softmax(zero_logits.detach().float(), -1)
        zero_ratio = (
            zero_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            - cache["mirror_anchor_action_log_prob"]
        )
        zero_drift = zero_ratio.square().mean().sqrt().item()
        del zero_logits, zero_logp, zero_ratio
        records = []
        for step in range(4):
            optim.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids).logits[:, :-1]
                r = compute_mirror_opd_token_loss(
                    logits,
                    actions,
                    cache["mirror_anchor_ids"],
                    cache["mirror_anchor_head_log_probs"],
                    cache["mirror_anchor_tail_log_prob"],
                    cache["mirror_anchor_action_log_prob"],
                    cache["mirror_teacher_action_log_prob"],
                    cache["mirror_teacher_center"],
                    "mdpo",
                    0.3,
                )
                loss = r["distillation_losses"].mean()
            records.append(
                {
                    "pre_update_step": step,
                    "student_center": r["mirror_student_center"].mean().item(),
                    "log_ratio_rms": r["mirror_sampled_log_ratio"].square().mean().sqrt().item(),
                }
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optim.step()
        changed = total = 0
        delta2 = max_delta = 0.0
        for n, p in model.named_parameters():
            now = p.detach().cpu()
            diff = now.float() - before[n].float()
            changed += int((now != before[n]).sum())
            total += p.numel()
            delta2 += float(diff.square().sum())
            max_delta = max(max_delta, float(diff.abs().max()))
        out[str(dtype)] = {
            "zero_update_log_ratio_rms": zero_drift,
            "changed_parameters": changed,
            "total_parameters": total,
            "changed_fraction": changed / total,
            "parameter_delta_l2": delta2**0.5,
            "parameter_delta_max": max_delta,
            "pre_update_records": records,
        }
        del model, optim, before, r, logits, loss
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
