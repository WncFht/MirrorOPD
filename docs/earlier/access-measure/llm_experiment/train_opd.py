#!/usr/bin/env python3
"""Small real-model MirrorOPD experiment.

The implementation intentionally makes the outer/inner boundary explicit:
all rollout tensors, anchor log-probs, teacher log-probs and eta are frozen
within one outer step. Only after the inner budget is exhausted is the student
weight state advanced to the next anchor.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

METHODS = ("pmd_part", "pmd_mean", "bpo", "mdpo")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def token_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits[:, :-1], dim=-1).gather(-1, labels[:, 1:, None]).squeeze(-1)


def make_prompts(tokenizer, n: int, seed: int) -> list[str]:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "wiki.valid.raw"
    if not raw_path.exists():
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
        urllib.request.urlretrieve(url, raw_path)
    texts = [
        line.strip()
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if len(line.strip()) >= 80
    ]
    rng = random.Random(seed)
    rng.shuffle(texts)
    prompts = []
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=48, add_special_tokens=False)["input_ids"]
        if len(ids) >= 16:
            prompts.append(tokenizer.decode(ids[:16], skip_special_tokens=True))
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        raise RuntimeError(f"only found {len(prompts)} usable prompts")
    return prompts


@torch.no_grad()
def rollout(
    student,
    teacher,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    eta: float,
    device: torch.device,
):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(
        device
    )
    anchor_out = student.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    labels = anchor_out.clone()
    prompt_len = enc["input_ids"].shape[1]
    # Left padding is disabled by the tokenizer setup; mask prompt and pads.
    token_mask = torch.ones_like(labels, dtype=torch.float32)
    token_mask[:, :prompt_len] = 0.0
    token_mask[labels == tokenizer.pad_token_id] = 0.0
    anchor_logits = student(anchor_out).logits
    teacher_logits = teacher(anchor_out).logits
    all_anchor = F.log_softmax(anchor_logits[:, :-1], dim=-1)
    all_teacher = F.log_softmax(teacher_logits[:, :-1], dim=-1)
    log_anchor = all_anchor.gather(-1, labels[:, 1:, None]).squeeze(-1)
    log_teacher = all_teacher.gather(-1, labels[:, 1:, None]).squeeze(-1)
    # Exact categorical log-partition for q ∝ anchor^(1-eta) teacher^eta.
    log_partition_all = torch.logsumexp((1.0 - eta) * all_anchor + eta * all_teacher, dim=-1)
    log_partition = log_partition_all
    mask = token_mask[:, 1:]
    return (
        anchor_out,
        log_anchor.detach(),
        log_teacher.detach(),
        log_partition.detach(),
        mask.detach(),
        prompt_len,
    )


def inner_loss(
    method: str,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    log_anchor: torch.Tensor,
    log_teacher: torch.Tensor,
    log_partition: torch.Tensor,
    mask: torch.Tensor,
    eta: float,
    vocab_chunk: int = 0,  # noqa: ARG001 — legacy caller compatibility; full logits are used
) -> tuple[torch.Tensor, dict[str, float]]:
    log_student = token_logprob(student_logits, labels)
    reward = log_teacher - log_anchor
    ratio_log = log_student - log_anchor
    m = mask
    if method == "pmd_part":
        coeff = eta * reward - log_partition - ratio_log
        loss = masked_mean(0.5 * coeff.square(), m)
    elif method == "pmd_mean":
        center = masked_mean(reward, m).detach()
        coeff = eta * (reward - center) - ratio_log
        loss = masked_mean(0.5 * coeff.square(), m)
    elif method == "bpo":
        d = eta * reward - ratio_log
        center = masked_mean(d, m).detach()
        coeff = d - center
        loss = masked_mean(0.5 * coeff.square(), m)
    elif method == "mdpo":
        # Anchor-rollout policy-gradient estimator for the KL surrogate.  The
        # advantage is stopped before multiplying by the sampled student
        # log-prob; otherwise minimizing ``-E[r-ratio/eta]`` has the wrong sign
        # because the sampled action distribution is fixed at the anchor.
        coeff = (reward - ratio_log / eta).detach()
        loss = -masked_mean(coeff * log_student, m)
    else:
        raise ValueError(method)
    with torch.no_grad():
        diag = {
            "inner_loss": float(loss.detach().cpu()),
            "reward_mean": float(masked_mean(reward, m).cpu()),
            "student_anchor_logratio": float(masked_mean(ratio_log, m).cpu()),
            "residual_rms": float(
                torch.sqrt(masked_mean((ratio_log - eta * reward).square(), m)).cpu()
            ),
            "coefficient_var": float(torch.var(coeff[m.bool()]).cpu()) if bool(m.any()) else 0.0,
        }
    return loss, diag


def evaluate(student, teacher, labels, log_anchor, log_teacher, mask):  # noqa: ARG001 — teacher log-probabilities are already cached
    with torch.no_grad():
        logits = student(labels).logits
        log_student = token_logprob(logits, labels)
        nll_student = -masked_mean(log_student, mask)
        nll_teacher = -masked_mean(log_teacher, mask)
        kl_st_t = masked_mean(log_student - log_teacher, mask)
        reward = masked_mean(log_teacher - log_anchor, mask)
    return {
        "student_nll": float(nll_student),
        "teacher_nll": float(nll_teacher),
        "kl_student_teacher": float(kl_st_t),
        "opd_reward": float(reward),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, default="pmd_part")
    # The default pair is deliberately asymmetric: a 14M student is cheap
    # enough for many outer/inner ablations, while the 70M teacher remains in
    # the same Pythia/GPT-NeoX/tokenizer family and supplies a nontrivial
    # distillation signal.  The former 70M/160M pair is still selectable from
    # the command line for a scale-up comparison.
    ap.add_argument("--teacher", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--student", default="EleutherAI/pythia-14m")
    ap.add_argument("--outer-steps", type=int, default=8)
    ap.add_argument("--inner-steps", type=int, default=50)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--eta-final", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    teacher = (
        AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    # Keep the trainable tiny student in fp32 for stable tiny-batch updates;
    # the frozen teacher remains fp16 to reduce inference memory.
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float32).to(
        device
    )
    for p in teacher.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    prompts = make_prompts(tokenizer, args.prompts, args.seed)
    run_name = f"{args.method}-eta{args.eta:g}-seed{args.seed}"
    metrics_path = args.output_dir / f"{run_name}.jsonl"
    with metrics_path.open("w", encoding="utf-8") as mf:
        for outer in range(args.outer_steps):
            eta = (
                args.eta
                if args.eta_final is None
                else args.eta + (args.eta_final - args.eta) * outer / max(1, args.outer_steps - 1)
            )
            # Freeze anchor by snapshotting student weights and using the snapshot
            # for rollout/log-prob computation. The student itself is only updated
            # after this point, and no anchor tensors are recomputed in the inner loop.
            # At this point ``student`` is exactly the frozen anchor. Rollout and
            # anchor log-probs are computed before the first optimizer step; the
            # resulting tensors are detached and reused throughout this outer step.
            student.eval()
            sequences, log_anchor, log_teacher, log_partition, mask, _prompt_len = rollout(
                student,
                teacher,
                tokenizer,
                prompts,
                args.max_new_tokens,
                eta,
                device,
            )
            student.train()
            last = {}
            for _inner in range(args.inner_steps):
                optimizer.zero_grad(set_to_none=True)
                loss_total = 0.0
                for start in range(0, len(prompts), args.batch_size):
                    sl = slice(start, start + args.batch_size)
                    loss, diag = inner_loss(
                        args.method,
                        student(sequences[sl]).logits,
                        sequences[sl],
                        log_anchor[sl],
                        log_teacher[sl],
                        log_partition[sl],
                        mask[sl],
                        eta,
                    )
                    loss_total = loss_total + loss / math.ceil(len(prompts) / args.batch_size)
                    last = diag
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
            ev = evaluate(student, teacher, sequences, log_anchor, log_teacher, mask)
            row = {
                "outer": outer,
                "eta": eta,
                "inner_steps": args.inner_steps,
                **last,
                **ev,
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            print(json.dumps(row, ensure_ascii=False))
            del sequences, log_anchor, log_teacher, log_partition, mask
            torch.cuda.empty_cache()
    student.save_pretrained(args.output_dir / f"{run_name}-final")
    tokenizer.save_pretrained(args.output_dir / f"{run_name}-final")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
