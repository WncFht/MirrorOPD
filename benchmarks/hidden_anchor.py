"""Paired exact-center sampled BPO benchmark, with independent process peaks.

CPU is always available. CUDA requires an idle physical GPU and acquires the
same reservation as native jobs. Peak RSS includes the Python/Torch runtime;
CUDA peaks are process-local allocated/reserved bytes, not full-card NVML.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from mirroropd.core.cache import build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective
from mirroropd.core.hidden import build_hidden_anchor, sampled_loss, score_hidden_teacher
from mirroropd.core.kernels import projected_loss
from mirroropd.runtime.memory import tensor_bytes


def measure(
    storage,
    *,
    device="cpu",
    dtype="fp32",
    tokens=128,
    vocab=8192,
    width=128,
    token_chunk=32,
    vocab_chunk=1024,
):
    torch.set_num_threads(2)
    torch.manual_seed(1810)
    target = torch.device(device)
    base = torch.float64 if dtype == "fp64" else torch.float32
    hq, ht = [torch.randn(tokens, width, dtype=base, device=target) * 0.2 for _ in range(2)]
    wq, wt = [torch.randn(vocab, width, dtype=base, device=target) * 0.2 for _ in range(2)]
    h, w = hq.clone().requires_grad_(), (wq + 0.01 * torch.randn_like(wq)).requires_grad_()
    actions = torch.arange(tokens, device=target) % vocab
    valid = torch.ones(tokens, dtype=torch.bool, device=target)
    objective = Objective(estimator=Estimator("full", "sampled", 16), diagnostics="basic")
    context = CacheContext(
        0, "synthetic-teacher", "synthetic-vocab", vocab, 0.7, objective.signal, objective.estimator
    )
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def precision():
        return (
            torch.autocast(target.type, dtype=torch.bfloat16) if dtype == "bf16" else nullcontext()
        )

    start = time.perf_counter()
    with precision(), torch.no_grad():
        if storage == "hidden":
            context = replace(
                context, anchor_storage="hidden", anchor_head_path="synthetic-head.pt"
            )
            frozen_weight = wq.to(torch.bfloat16 if dtype == "bf16" else base).clone()
            from mirroropd.core.cache import AnchorCache, FrozenCache, select_valid

            anchors = [
                build_hidden_anchor(
                    hq[begin : begin + token_chunk],
                    wq,
                    None,
                    actions[begin : begin + token_chunk],
                    context,
                    vocab_chunk=vocab_chunk,
                )
                for begin in range(0, tokens, token_chunk)
            ]
            anchor = AnchorCache(
                context=context,
                **{
                    name: torch.cat([p.tensors()[name] for p in anchors])
                    for name in anchors[0].tensors()
                },
            )
            del anchors
            pieces = [
                score_hidden_teacher(
                    ht[begin : begin + token_chunk],
                    wt,
                    None,
                    select_valid(anchor, begin, begin + token_chunk),
                    frozen_weight,
                    None,
                    diagnostics="basic",
                    vocab_chunk=vocab_chunk,
                )
                for begin in range(0, tokens, token_chunk)
            ]
            cache = FrozenCache(
                context=context,
                **anchor.tensors(),
                teacher_action=torch.cat([p.teacher_action for p in pieces]),
                reward_center=torch.cat([p.reward_center for p in pieces]),
            )
            del anchor, pieces
        else:
            from mirroropd.core.cache import FrozenCache

            pieces = [
                build_cache(
                    F.linear(hq[begin : begin + token_chunk], wq) / 0.7,
                    F.linear(ht[begin : begin + token_chunk], wt) / 0.7,
                    actions[begin : begin + token_chunk],
                    valid[begin : begin + token_chunk],
                    context,
                    diagnostics="basic",
                )
                for begin in range(0, tokens, token_chunk)
            ]
            cache = FrozenCache(
                context=context,
                **{
                    name: torch.cat([p.tensors()[name] for p in pieces])
                    for name in pieces[0].tensors()
                },
            )
            del pieces
    sync()
    build_seconds = time.perf_counter() - start
    cache_bytes = tensor_bytes(cache.tensors().values())
    start = time.perf_counter()
    with precision():
        result = (
            sampled_loss(
                h,
                w,
                None,
                cache,
                objective,
                frozen_weight,
                None,
                token_chunk=token_chunk,
                vocab_chunk=vocab_chunk,
            )
            if storage == "hidden"
            else projected_loss(
                h,
                w,
                None,
                cache,
                objective,
                anchor_version=0,
                token_chunk=token_chunk,
                vocab_chunk=vocab_chunk,
                temperature=0.7,
            )
        )
        result.loss.mean().backward()
    sync()
    update_seconds = time.perf_counter() - start
    return {
        "storage": storage,
        "device": device,
        "precision": dtype,
        "shape": {
            "tokens": tokens,
            "vocab": vocab,
            "width": width,
            "token_chunk": token_chunk,
            "vocab_chunk": vocab_chunk,
        },
        "cache_storage_bytes": cache_bytes,
        "shared_head_bytes": tensor_bytes([frozen_weight]) if storage == "hidden" else 0,
        "build_seconds": build_seconds,
        "update_seconds": update_seconds,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "loss_mean": result.loss.mean().item(),
        "student_kl_mean": result.diagnostics["student_anchor_kl"].mean().item(),
        "teacher_center_mean": cache.reward_center.mean().item(),
        "hidden_gradient_norm": h.grad.norm().item(),
        "head_gradient_norm": w.grad.norm().item(),
    }


def precision_errors(low_precision):
    """Independent eager LP reference with matched BF16 projection semantics."""
    from mirroropd.core.objectives import compute_loss

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(18)
        n, v, d = 9, 113, 32
        hq, ht, h = [torch.randn(n, d) * 0.2 for _ in range(3)]
        wq, wt, w = [torch.randn(v, d) * 0.2 for _ in range(3)]
        h.requires_grad_()
        w.requires_grad_()
        actions = torch.arange(n) * 11
        spec = Objective(estimator=Estimator("full", "sampled", 4), diagnostics="basic")
        context = CacheContext(0, "t", "v", v, 0.7, spec.signal, spec.estimator)
        with torch.autocast("cpu", dtype=torch.bfloat16) if low_precision else nullcontext():
            eager = build_cache(
                F.linear(hq, wq) / 0.7,
                F.linear(ht, wt) / 0.7,
                actions,
                torch.ones(n, dtype=torch.bool),
                context,
                diagnostics="basic",
            )
            compact = replace(context, anchor_storage="hidden", anchor_head_path="test.pt")
            anchor = build_hidden_anchor(hq, wq, None, actions, compact, vocab_chunk=29)
            frozen_weight = wq.to(torch.bfloat16 if low_precision else torch.float32)
            cache = score_hidden_teacher(
                ht, wt, None, anchor, frozen_weight, None, diagnostics="basic", vocab_chunk=29
            )
            actual = sampled_loss(
                h, w, None, cache, spec, frozen_weight, None, token_chunk=4, vocab_chunk=29
            )
            expected = compute_loss(F.linear(h, w) / 0.7, eager, spec, anchor_version=0)
            ga = torch.autograd.grad(actual.loss.sum(), (h, w))
            ge = torch.autograd.grad(expected.loss.sum(), (h, w))
        return {
            "precision": "bf16" if low_precision else "fp32",
            "loss_max_abs": (actual.loss - expected.loss).abs().max().item(),
            "teacher_center_max_abs": (cache.reward_center - eager.reward_center)
            .abs()
            .max()
            .item(),
            "student_kl_max_abs": (
                actual.diagnostics["student_anchor_kl"] - expected.diagnostics["student_anchor_kl"]
            )
            .abs()
            .max()
            .item(),
            "gradient_max_abs": [(a - e).abs().max().item() for a, e in zip(ga, ge, strict=True)],
            "gradient_relative_l2": [
                ((a - e).norm() / e.norm()).item() for a, e in zip(ga, ge, strict=True)
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp64"], default="fp32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=["hidden", "log_probs"])
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()
    if args.worker:
        for warm in ("log_probs", "hidden"):
            measure(
                warm,
                device="cuda" if args.gpu is not None else "cpu",
                dtype=args.precision,
                tokens=8,
                vocab=64,
                width=16,
                token_chunk=4,
                vocab_chunk=17,
            )
        print(
            json.dumps(
                measure(
                    args.worker,
                    device="cuda" if args.gpu is not None else "cpu",
                    dtype=args.precision,
                    tokens=args.tokens,
                    vocab=args.vocab,
                    width=args.width,
                )
            )
        )
        return
    if args.output is None:
        parser.error("--output is required")
    from mirroropd.runtime.reservation import GpuReservation

    lease = (
        GpuReservation(
            [args.gpu], Path.home() / ".cache/mirroropd/gpu-leases", "hidden-anchor-benchmark"
        )
        if args.gpu is not None
        else nullcontext()
    )
    records = []
    with lease:
        for storage in ("log_probs", "hidden") * args.repeats:
            command = [
                sys.executable,
                __file__,
                "--worker",
                storage,
                "--precision",
                args.precision,
                "--tokens",
                str(args.tokens),
                "--vocab",
                str(args.vocab),
                "--width",
                str(args.width),
            ]
            env = os.environ.copy()
            if args.gpu is not None:
                command += ["--gpu", str(args.gpu)]
                env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            raw = subprocess.check_output(command, text=True, env=env)  # noqa: S603 — fixed entrypoint and parsed numeric options
            records.append(json.loads(raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": records,
                "torch": torch.__version__,
                "cpu_precision_check": precision_errors(args.precision == "bf16"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
