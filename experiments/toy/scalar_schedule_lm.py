"""Fixed q-rollout DeepSeek benchmark using the production sampled kernel."""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from scalar_schedule_benchmark import thresholds
from transformers import AutoModelForCausalLM, AutoTokenizer

from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.losses import compute_mirror_opd_token_loss
from mirroropd.training.scalar_schedule import absorption_feedback, residual_line_search

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def load_model(path):
    return (
        AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        )
        .cuda()
        .eval()
    )


def prediction_logits(model, ids, start, *, mixed_precision=True, attention_mask=None):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
        hidden = model.model(
            input_ids=ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state[:, start:-1]
        return model.lm_head(hidden).float().reshape(-1, model.config.vocab_size)


def prepare(args):
    import pandas as pd

    torch.manual_seed(args.seed)
    student_path = WORKSPACE / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    teacher_path = WORKSPACE / "models/JustRL-DeepSeek-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(
        student_path, local_files_only=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    frame = pd.read_parquet(WORKSPACE / "MD-OPD/data/math/DAPO/train.parquet")
    prompts = [
        tokenizer.apply_chat_template(list(row), tokenize=False, add_generation_prompt=True)
        for row in frame.prompt.iloc[: args.prompts]
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.prompt_length,
    ).to("cuda")
    student = load_model(student_path)
    generation_config = copy.deepcopy(student.generation_config)
    generation_config.eos_token_id = None
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"),
    ):
        ids = student.generate(
            **inputs,
            do_sample=True,
            temperature=1,
            top_p=1,
            top_k=0,
            max_new_tokens=args.response_length,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
        )
        full_mask = torch.cat(
            (inputs["attention_mask"], torch.ones_like(ids[:, args.prompt_length :])), dim=1
        )
        q_logits = prediction_logits(
            student,
            ids,
            args.prompt_length - 1,
            mixed_precision=args.precision == "bf16",
            attention_mask=full_mask,
        ).cpu()
    del student
    torch.cuda.empty_cache()
    teacher = load_model(teacher_path)
    with torch.no_grad():
        t_logits = prediction_logits(
            teacher,
            ids,
            args.prompt_length - 1,
            mixed_precision=args.precision == "bf16",
            attention_mask=full_mask,
        ).cpu()
    actions = ids[:, args.prompt_length :].reshape(-1).cpu()
    artifact = {
        "ids": ids.cpu(),
        "attention_mask": full_mask.cpu(),
        "q_logits": q_logits,
        "t_logits": t_logits,
        "actions": actions,
        "start": args.prompt_length - 1,
        "seed": args.seed,
        "prompts": prompts,
        "prompt_length": args.prompt_length,
        "response_length": args.response_length,
        "student": str(student_path),
        "teacher": str(teacher_path),
        "temperature": 1,
        "top_p": 1,
        "top_k": 0,
        "truncated_prompts": True,
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.cache)
    print(
        json.dumps(
            {"cache": str(args.cache), "prefixes": len(actions), "vocab": q_logits.shape[-1]}
        ),
        flush=True,
    )


def train(args):
    torch.manual_seed(args.seed)
    artifact = torch.load(args.cache, weights_only=False, map_location="cpu")
    ids = artifact["ids"].cuda()
    attention_mask = artifact.get("attention_mask", torch.ones_like(artifact["ids"]))
    attention_mask = attention_mask.cuda()
    args.prompts = ids.shape[0]
    args.prompt_length = artifact["prompt_length"]
    args.response_length = artifact["response_length"]
    q_logits, t_logits = artifact["q_logits"].cuda(), artifact["t_logits"].cuda()
    actions = artifact["actions"].cuda()
    cache = build_mirror_opd_cache_from_full_logits(
        q_logits, t_logits, actions, torch.ones_like(actions, dtype=torch.bool), args.topk
    )
    qlog, tlog = q_logits.log_softmax(-1), t_logits.log_softmax(-1)
    q = qlog.exp()
    target = (1 - args.eta) * qlog + args.eta * tlog
    del q_logits, t_logits
    student = load_model(artifact["student"])
    parameters = list(student.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False
    )
    forwards = 0

    def problem():
        nonlocal forwards
        forwards += 1
        z = prediction_logits(
            student,
            ids,
            artifact["start"],
            mixed_precision=args.precision == "bf16",
            attention_mask=attention_mask,
        )
        logp = z.log_softmax(-1)
        delta = logp - target
        residual = delta - (q * delta).sum(-1, keepdim=True)
        exact = (q * residual.square()).sum(-1).mean() / (2 * args.eta)
        teacher_kl = (logp.exp() * (logp - tlog)).sum(-1).mean()
        if args.objective == "exact":
            return exact, exact.detach(), exact.detach(), teacher_kl.detach()
        out = compute_mirror_opd_token_loss(
            z,
            actions,
            cache["mirror_anchor_ids"],
            cache["mirror_anchor_head_log_probs"],
            cache["mirror_anchor_tail_log_prob"],
            cache["mirror_anchor_action_log_prob"],
            cache["mirror_teacher_action_log_prob"],
            cache["mirror_teacher_center"],
            "bpo",
            args.eta,
        )
        metric = out["mirror_score_coefficient"].square().mean() * (args.eta / 2)
        return out["distillation_losses"].mean(), metric, exact.detach(), teacher_kl.detach()

    with torch.no_grad():
        _, initial_metric, initial_exact, initial_kl = problem()
    initial_metric = float(initial_metric)
    last = initial_metric
    current_lr = args.lr
    rows = []
    rejected = 0
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".jsonl")
    with log_path.open("w") as stream:
        for iteration in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss, metric, exact, teacher_kl = problem()
            value = float(metric)
            row = {
                "step": iteration,
                "metric": value,
                "exact": float(exact),
                "teacher_kl": float(teacher_kl),
                "lr": current_lr,
                "forwards": forwards,
                "seconds": time.perf_counter() - start_time,
            }
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if not math.isfinite(value) or value > initial_metric * 1e6:
                break
            if args.method == "logloss":
                loss = loss * (initial_metric / max(value, initial_metric * 1e-6))
            if args.method == "feedback" and iteration:
                current_lr = absorption_feedback(current_lr, last, value, 0.03, args.maximum)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 2)
            row["grad_norm"] = float(grad_norm)
            optimizer.param_groups[0]["lr"] = current_lr
            if args.method == "search":
                old = [p.detach().clone() for p in parameters]
                optimizer.step()
                # Moments advance once. Read the nominal Adam direction
                # directly, avoiding fp32 cancellation of tiny displacements.
                direction = []
                for p in parameters:
                    state = optimizer.state[p]
                    count = int(state["step"])
                    first = state["exp_avg"] / (1 - 0.9**count)
                    denominator = (state["exp_avg_sq"] / (1 - 0.999**count)).sqrt().add_(1e-8)
                    direction.append(first / denominator)

                def install(alpha, old=old, direction=direction):
                    with torch.no_grad():
                        for p, before, d in zip(parameters, old, direction, strict=True):
                            p.copy_(before)
                            p.add_(d, alpha=-alpha)

                def evaluate(alpha):
                    install(alpha)
                    with torch.no_grad():
                        _, trial_metric, trial_exact, _ = problem()
                    return float(trial_exact if args.monitor == "exact" else trial_metric)

                monitored = float(exact) if args.monitor == "exact" else value
                result = residual_line_search(
                    evaluate, monitored, current_lr, args.maximum, minimum=1e-10
                )
                install(result.step)
                rejected += result.step == 0
                row.update(
                    selected_lr=result.step,
                    search_forwards=result.evaluations,
                    slope=result.slope,
                    curvature=result.curvature,
                )
                current_lr = (
                    max(args.lr * 0.01, current_lr * 0.5) if result.step == 0 else result.step
                )
                del old, direction
            else:
                optimizer.step()
            last = value
            if iteration % 10 == 0:
                print(json.dumps(row), flush=True)
        with torch.no_grad():
            _, metric, exact, kl = problem()
        rows.append(
            {
                "step": len(rows),
                "metric": float(metric),
                "exact": float(exact),
                "teacher_kl": float(kl),
                "lr": current_lr,
                "forwards": forwards,
                "seconds": time.perf_counter() - start_time,
            }
        )
    torch.cuda.synchronize()
    args.output.write_text(
        json.dumps(
            {
                "configuration": {
                    k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                },
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                "initial_exact": float(initial_exact),
                "initial_kl": float(initial_kl),
                "backwards": len(rows) - 1,
                "forwards": forwards,
                "rejected": rejected,
                "seconds": time.perf_counter() - start_time,
                "thresholds": thresholds([r["exact"] for r in rows]),
                "trajectory": rows,
            }
        )
    )
    print(
        json.dumps(
            {"output": str(args.output), "final_exact": float(exact), "final_kl": float(kl)}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "runs/scalar-schedule/deepseek-cache.pt"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/scalar-schedule/lm.json")
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--response-length", type=int, default=24)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--maximum", type=float, default=1.6e-5)
    parser.add_argument("--objective", choices=("exact", "sampled"), default="sampled")
    parser.add_argument("--monitor", choices=("exact", "sampled"), default="sampled")
    parser.add_argument(
        "--method", choices=("constant", "feedback", "logloss", "search"), default="constant"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.prepare:
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
