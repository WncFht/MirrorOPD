"""Reproducible GPU objective benchmarks on saved real conditional distributions."""

import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import torch

from mirroropd.core.cache import build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective, Signal
from mirroropd.core.objectives import compute_loss
from mirroropd.runtime.artifacts import atomic_json, file_digest
from mirroropd.runtime.reservation import GpuReservation


def benchmark(inputs, objective, prefixes, *, iterations):
    q = inputs["q"][:prefixes].cuda()
    teacher = inputs["t"][:prefixes].cuda()
    actions = inputs["actions"][:prefixes].cuda()
    context = CacheContext(
        0,
        inputs["source_sha256"],
        "saved-vocabulary",
        q.shape[-1],
        1.0,
        objective.signal,
        objective.estimator,
    )
    cache = build_cache(
        q,
        teacher,
        actions,
        torch.ones_like(actions, dtype=torch.bool),
        context,
        diagnostics=objective.diagnostics,
    )
    z = q.clone().requires_grad_()
    if objective.algorithm != "on_policy":
        with torch.no_grad():
            z[:, 0] += 0.2

    def step(student, frozen, spec):
        student.grad = None
        result = compute_loss(student, frozen, spec, anchor_version=0)
        result.loss.mean().backward()
        return result

    for _ in range(3):
        step(z, cache, objective)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(iterations):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = step(z, cache, objective)
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop))
    peak = torch.cuda.max_memory_allocated()
    gradient = z.grad.cpu()
    record = {
        "prefixes": prefixes,
        "vocab": q.shape[-1],
        "algorithm": objective.algorithm,
        "center": objective.estimator.center,
        "integration": objective.estimator.integration,
        "diagnostics": objective.diagnostics,
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "peak_allocated_bytes": peak,
        "loss_kind": result.kind,
        "loss_mean": result.loss.mean().item(),
        "gradient_norm": gradient.norm().item(),
    }
    del result, z, cache, q, teacher, actions
    gc.collect()
    torch.cuda.empty_cache()
    return record, gradient


def compare_baseline(record, gradient, baseline, baseline_dir):
    # Historical evidence is read only here; production accepts no legacy modes.
    method = "opd_onpolicy" if record["algorithm"] == "on_policy" else record["algorithm"]
    center = "full_vocab" if record["center"] == "full" else record["center"]
    integration = "full_vocab" if record["integration"] == "full" else record["integration"]
    matches = [
        row
        for row in baseline
        if (
            row["prefixes"],
            row["algorithm"],
            row["center"],
            row["integration"],
            row["diagnostics"],
        )
        == (record["prefixes"], method, center, integration, record["diagnostics"])
    ]
    if not matches:
        return
    old = matches[0]
    key = f"{record['prefixes']}-{method}-{center}-{integration}"
    expected = torch.load(
        baseline_dir / (key + "-gradient.pt"), map_location="cpu", weights_only=True
    )
    difference = (gradient - expected).double()
    maximum = difference.abs().max().item()
    relative = difference.norm().item() / max(expected.double().norm().item(), 1e-30)
    record.update(
        baseline_median_ms=old["median_ms"],
        baseline_peak_allocated_bytes=old["peak_allocated_bytes"],
        speed_ratio=old["median_ms"] / record["median_ms"],
        gradient_max_abs_difference=maximum,
        gradient_relative_l2_difference=relative,
    )
    if maximum > 2e-6 or relative > 2e-4:
        raise AssertionError(f"Kernel gradient differs from recorded baseline: {record}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--borrowed-slot", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("Benchmark must start before any CUDA context is initialized")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else []
    records = []
    metadata = {
        "schema_version": 2,
        "scope": "Objective forward/backward including selected diagnostic level; not whole-model throughput",
        "input_sha256": file_digest(args.inputs),
        "warmup_iterations": 3,
        "measured_iterations": args.iterations,
        "torch": torch.__version__,
        "physical_gpu": args.gpu,
        "kappa": 0.5,
        "eta": 0.2,
    }
    with GpuReservation(
        [args.gpu], args.lease_root, "kernel-benchmark", borrowed_slot=args.borrowed_slot
    ) as reservation:
        for prefixes in sorted({16, min(128, len(inputs["q"]))}):
            for algorithm in ("on_policy", "mdpo", "bpo"):
                for center, integration in (
                    ("topj_tail", "sampled"),
                    ("full", "sampled"),
                    ("full", "full"),
                ):
                    for diagnostics in ("none", "basic", "full"):
                        reservation.heartbeat()
                        objective = Objective(
                            algorithm,
                            Signal(0.5, 0.2),
                            Estimator(center, integration, 16),
                            diagnostics,
                        )
                        record, gradient = benchmark(
                            inputs, objective, prefixes, iterations=args.iterations
                        )
                        if args.baseline:
                            compare_baseline(record, gradient, baseline, args.baseline.parent)
                        records.append(record)
                        atomic_json(args.output, {"metadata": metadata, "cases": records})
    print(json.dumps({"cases": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
