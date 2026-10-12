"""Frozen quadratic and categorical benchmarks; no Hessian in tested search."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.losses import compute_mirror_opd_token_loss
from mirroropd.training.scalar_schedule import absorption_feedback, residual_line_search


def thresholds(values):
    result = {}
    for ratio in (0.1, 0.01, 0.001):
        cut = values[0] * ratio
        first = next((i for i, v in enumerate(values) if v <= cut), None)
        sustained = next((i for i in range(len(values) - 4) if max(values[i : i + 5]) <= cut), None)
        result[str(ratio)] = {"first": first, "sustained5": sustained}
    return result


def run(problem, initial, method, lr, steps, optimizer_name="adam", maximum=None):
    maximum = maximum or lr * 100
    theta = torch.nn.Parameter(initial.clone())
    optimizer = (
        torch.optim.AdamW([theta], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        if optimizer_name == "adam"
        else torch.optim.SGD([theta], lr=lr)
    )
    trajectory = []
    forwards = 0
    rejected = 0
    start = time.perf_counter()
    initial_metric = float(problem(theta)[1].detach())
    last = initial_metric
    current_lr = lr
    for iteration in range(steps):
        optimizer.zero_grad()
        loss, metric, exact = problem(theta)
        forwards += 1
        value = float(metric.detach())
        trajectory.append(
            {
                "step": iteration,
                "metric": value,
                "exact": float(exact.detach()),
                "lr": current_lr,
                "forwards": forwards,
            }
        )
        if not math.isfinite(value) or value > initial_metric * 1e8:
            break
        if method == "logloss":
            scale = initial_metric / max(value, initial_metric * 1e-12)
            loss = loss * scale
        elif method == "sqrtloss":
            loss = loss * math.sqrt(initial_metric / max(value, initial_metric * 1e-12))
        elif method == "inverse_lr":
            current_lr = min(maximum, lr * initial_metric / max(value, initial_metric * 1e-12))
        elif method == "feedback" and iteration:
            current_lr = absorption_feedback(current_lr, last, value, 0.03, maximum)
        loss.backward()
        old = theta.detach().clone()
        # Advance the standard optimizer once; all subsequent trials reuse this
        # direction and do not update moments or gradients.
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.step()
        displacement = theta.detach() - old
        direction = -displacement / current_lr
        if method == "polyak":
            alignment = float((theta.grad * direction).sum())
            selected = min(maximum, 0.05 * value / alignment) if alignment > 0 else 0
            with torch.no_grad():
                theta.copy_(old - selected * direction)
            current_lr = max(selected, lr * 1e-6)
        elif method == "search":

            def evaluate(alpha, old=old, direction=direction):
                with torch.no_grad():
                    return float(problem(old - alpha * direction)[1])

            result = residual_line_search(evaluate, value, current_lr, maximum)
            forwards += result.evaluations
            rejected += result.step == 0
            with torch.no_grad():
                theta.copy_(old - result.step * direction)
            current_lr = max(result.step, current_lr * 0.5) if result.step == 0 else result.step
        last = value
    with torch.no_grad():
        _, metric, exact = problem(theta)
    forwards += 1
    trajectory.append(
        {
            "step": len(trajectory),
            "metric": float(metric),
            "exact": float(exact),
            "lr": current_lr,
            "forwards": forwards,
        }
    )
    return {
        "method": method,
        "lr": lr,
        "optimizer": optimizer_name,
        "maximum": maximum,
        "backwards": len(trajectory) - 1,
        "forwards": forwards,
        "rejected": rejected,
        "seconds": time.perf_counter() - start,
        "thresholds": thresholds([r["exact"] for r in trajectory]),
        "trajectory": trajectory,
    }


def quadratic(seed, condition, rotated):
    generator = torch.Generator().manual_seed(seed)
    dimension = 48
    basis = (
        torch.linalg.qr(
            torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
        ).Q
        if rotated
        else torch.eye(dimension, dtype=torch.float64)
    )
    spectrum = torch.logspace(-math.log10(condition), 0, dimension, dtype=torch.float64)
    matrix = (basis * spectrum) @ basis.T
    initial = basis @ (
        torch.randn(dimension, generator=generator, dtype=torch.float64) / spectrum.sqrt()
    )

    def problem(theta):
        loss = 0.5 * theta @ matrix @ theta
        return loss, loss, loss

    return problem, initial


def categorical(seed, kind, samples, topk):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(128, 24, generator=generator, dtype=torch.float64) * torch.logspace(
        -2, 0, 24, dtype=torch.float64
    )
    initial = torch.randn(24, 32, generator=generator, dtype=torch.float64) * 0.1
    teacher_theta = initial + torch.randn(24, 32, generator=generator, dtype=torch.float64) * 0.4
    nonlinear = kind == "nonlinear"

    def logits(theta):
        linear = features @ theta
        return linear + 0.2 * linear.square() if nonlinear else linear

    q_logits = logits(initial).detach()
    t_logits = logits(teacher_theta).detach()
    q = q_logits.softmax(-1)
    target = 0.7 * q_logits + 0.3 * t_logits
    actions = torch.multinomial(q, samples, replacement=True, generator=generator)
    repeated_q = q_logits.repeat_interleave(samples, 0)
    repeated_t = t_logits.repeat_interleave(samples, 0)
    cache = build_mirror_opd_cache_from_full_logits(
        repeated_q, repeated_t, actions.flatten(), torch.ones(128 * samples, dtype=torch.bool), topk
    )

    def problem(theta):
        z = logits(theta)
        residual = z - target
        residual = residual - (q * residual).sum(-1, keepdim=True)
        exact = (q * residual.square()).sum(-1).mean() / 0.6
        if kind == "exact":
            return exact, exact, exact
        out = compute_mirror_opd_token_loss(
            z.repeat_interleave(samples, 0),
            cache["mirror_actions"],
            cache["mirror_anchor_ids"],
            cache["mirror_anchor_head_log_probs"],
            cache["mirror_anchor_tail_log_prob"],
            cache["mirror_anchor_action_log_prob"],
            cache["mirror_teacher_action_log_prob"],
            cache["mirror_teacher_center"],
            "bpo",
            0.3,
        )
        surrogate = out["distillation_losses"].mean()
        metric = out["mirror_score_coefficient"].square().mean() * 0.15
        return surrogate, metric, exact

    return problem, initial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("quadratic", "categorical"), required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for seed in range(args.seeds):
        if args.suite == "quadratic":
            cases = [
                (condition, rotated)
                for condition in (10, 1000, 100000)
                for rotated in (False, True)
            ]
        else:
            cases = [
                (kind, samples, topk)
                for kind in ("exact", "sampled", "nonlinear")
                for samples, topk in ((1, 32), (8, 32), (8, 8), (8, 1))
                if kind != "exact" or (samples, topk) == (1, 32)
            ]
        for case in cases:
            problem, initial = (
                quadratic(seed, *case) if args.suite == "quadratic" else categorical(seed, *case)
            )
            optimizers = ("gd", "adam") if args.suite == "quadratic" else ("adam",)
            for optimizer in optimizers:
                grid = (
                    (0.1, 0.5, 1.0, 1.8, 1.99)
                    if optimizer == "gd"
                    else (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
                )
                arms = [("constant", x) for x in grid] + [
                    (name, grid[1])
                    for name in (
                        "search",
                        "feedback",
                        "inverse_lr",
                        "polyak",
                        "logloss",
                        "sqrtloss",
                    )
                ]
                for method, lr in arms:
                    result = run(
                        problem,
                        initial,
                        method,
                        lr,
                        args.steps,
                        optimizer,
                        maximum=8 if optimizer == "gd" else 1,
                    )
                    result.update(seed=seed, case=case)
                    results.append(result)
            print(json.dumps({"seed": seed, "case": case, "completed": len(results)}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "suite": args.suite,
                        "steps": args.steps,
                        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "results": results,
                    }
                )
            )


if __name__ == "__main__":
    main()
