"""Compare projection+objective forward/backward on reproducible fixed inputs.

Run baseline and current in separate interpreters with their own PYTHONPATH.
CPU reports saved storage, not GPU peak. CUDA runs acquire the existing lease
protocol before creating a CUDA context. Dense evidence, weight gradients and
model-backbone memory are reported as separate scope boundaries.
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mirroropd.core.cache import build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective
from mirroropd.core.objectives import compute_loss
from mirroropd.runtime.artifacts import atomic_json, file_digest
from mirroropd.runtime.reservation import GpuReservation


def run_case(args, algorithm, integration, diagnostics):
    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device
    h = torch.randn(args.prefixes, args.width, dtype=dtype, device=device, requires_grad=True)
    w = (
        torch.randn(args.vocab, args.width, dtype=dtype, device=device) / args.width**0.5
    ).requires_grad_()
    q = torch.randn(args.prefixes, args.vocab, dtype=dtype, device=device)
    teacher = torch.randn_like(q)
    actions = torch.randint(args.vocab, (args.prefixes,), device=device)
    valid = torch.ones(args.prefixes, device=device, dtype=torch.bool)
    spec = Objective(
        algorithm,
        estimator=Estimator(
            "full" if integration == "full" else "topj_tail", integration, min(16, args.vocab)
        ),
        diagnostics=diagnostics,
    )
    context = CacheContext(
        0,
        "synthetic-fixed-teacher",
        "synthetic-vocabulary",
        args.vocab,
        1.0,
        spec.signal,
        spec.estimator,
    )
    start = time.perf_counter()
    frozen = build_cache(q, teacher, actions, valid, context, diagnostics=diagnostics)
    if device == "cuda":
        torch.cuda.synchronize()
    cache_seconds = time.perf_counter() - start
    cache_bytes = sum(x.numel() * x.element_size() for x in frozen.tensors().values())

    def step(h, w, frozen):
        h.grad = w.grad = None
        if args.execution == "eager":
            result = compute_loss(F.linear(h, w), frozen, spec, anchor_version=0)
        else:
            from mirroropd.core.kernels import projected_loss

            result = projected_loss(
                h,
                w,
                None,
                frozen,
                spec,
                anchor_version=0,
                token_chunk=args.chunk,
                vocab_chunk=args.vocab_chunk,
            )
        result.loss.mean().backward()
        return result

    for _ in range(2):
        step(h, w, frozen)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
    else:
        resident = None
    times = []
    for _ in range(args.iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = step(h, w, frozen)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    saved = {}

    def pack(x):
        key = (str(x.device), x.untyped_storage().data_ptr())
        saved[key] = {
            "bytes": x.untyped_storage().nbytes(),
            "shape": list(x.shape),
            "requires_grad": x.requires_grad,
        }
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        result = step(h, w, frozen)
    row = {
        "algorithm": algorithm,
        "integration": integration,
        "diagnostics": diagnostics,
        "execution": args.execution,
        "prefixes": args.prefixes,
        "vocab": args.vocab,
        "width": args.width,
        "dtype": args.dtype,
        "cache_bytes": cache_bytes,
        "cache_build_seconds": cache_seconds,
        "median_ms": statistics.median(times),
        "resident_allocated_bytes": resident,
        "peak_allocated_bytes": peak,
        "incremental_peak_bytes": peak - resident if peak is not None else None,
        "saved_unique_storage_bytes": sum(x["bytes"] for x in saved.values()),
        "saved_trainable_vocab_shapes": [
            x["shape"]
            for x in saved.values()
            if x["requires_grad"] and len(x["shape"]) == 2 and x["shape"][-1] == args.vocab
        ],
        "loss_mean": result.loss.mean().item(),
        "gradient_norm": w.grad.norm().item(),
    }
    gradient = torch.cat([h.grad.flatten(), w.grad.flatten()]).detach().cpu()
    del result, frozen, h, w, q, teacher, actions, valid
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return row, gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=["eager", "chunked"], default="chunked")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lease-root", type=Path)
    parser.add_argument("--borrowed-slot", type=Path)
    parser.add_argument("--prefixes", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--vocab-chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=522)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--algorithm", choices=["all", "on_policy", "mdpo", "bpo"], default="all")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if (
        min(args.prefixes, args.vocab, args.width, args.chunk, args.vocab_chunk, args.iterations)
        < 1
    ):
        parser.error("Dimensions, chunks and iterations must be positive")
    if args.device == "cuda":
        if args.lease_root is None:
            parser.error("CUDA benchmark requires --lease-root")
        if torch.cuda.is_initialized():
            raise RuntimeError("Reservation must precede CUDA initialization")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    import mirroropd

    source = Path(mirroropd.__file__).parent
    files = {
        str(p.relative_to(source)): file_digest(p)
        for p in source.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    metadata = {
        "scope": "LM-head projection plus objective forward/backward; synthetic fixed conditional distributions; excludes backbone, RPC, KV and optimizer",
        "device": args.device,
        "torch": torch.__version__,
        "source": str(source),
        "source_files": files,
        "seed": args.seed,
        "iterations": args.iterations,
        "token_chunk": args.chunk,
        "vocab_chunk": args.vocab_chunk,
        "input_protocol": "torch.randn h then scaled w then q then teacher then randint actions; fixed seed per case",
    }
    from contextlib import nullcontext

    lease = (
        GpuReservation(
            [args.gpu], args.lease_root, "memory-benchmark", borrowed_slot=args.borrowed_slot
        )
        if args.device == "cuda"
        else nullcontext()
    )
    baseline = json.loads(args.baseline.read_text())["cases"] if args.baseline else []
    cases = []
    with lease:
        for algorithm in (
            ["on_policy", "mdpo", "bpo"] if args.algorithm == "all" else [args.algorithm]
        ):
            for integration in ["sampled", "full"]:
                for diagnostics in ["none", "basic", "full"]:
                    row, gradient = run_case(args, algorithm, integration, diagnostics)
                    key = f"{algorithm}-{integration}-{diagnostics}"
                    if baseline:
                        old = next(
                            r
                            for r in baseline
                            if (r["algorithm"], r["integration"], r["diagnostics"])
                            == (algorithm, integration, diagnostics)
                        )
                        expected = torch.load(
                            args.baseline.parent / (key + "-gradient.pt"), weights_only=True
                        )
                        delta = (gradient - expected).double()
                        row.update(
                            gradient_max_abs_difference=delta.abs().max().item(),
                            gradient_relative_l2_difference=(
                                delta.norm() / expected.double().norm().clamp_min(1e-30)
                            ).item(),
                            baseline_median_ms=old["median_ms"],
                            baseline_saved_unique_storage_bytes=old["saved_unique_storage_bytes"],
                            baseline_peak_allocated_bytes=old["peak_allocated_bytes"],
                        )
                        torch.testing.assert_close(gradient, expected, atol=2e-6, rtol=2e-4)
                        if row["gradient_relative_l2_difference"] > 2e-4:
                            raise AssertionError(f"Projection parameter gradient differs: {row}")
                    torch.save(gradient, args.output.parent / (key + "-gradient.pt"))
                    cases.append(row)
                    atomic_json(args.output, {"metadata": metadata, "cases": cases})
                    if args.device == "cuda":
                        lease.heartbeat()
    print(json.dumps({"output": str(args.output), "cases": len(cases)}))


if __name__ == "__main__":
    main()
