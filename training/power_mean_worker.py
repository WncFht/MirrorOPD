"""Four-owned-GPU full-parameter fitting of frozen complete-vocabulary targets."""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.rollout.oneshot_batch import model_path, one_prompt
from mirroropd.proximal import (
    power_log_target,
    renyi_inner_bound,
    sampled_target_loss,
    target_divergence,
)
from training.gpu_exclusive import register_gpu_owner
from training.oneshot_worker import ResponseModel, reduce_sums, save_checkpoint


def panels(artifact, config, device="cuda"):
    """Batch 64 uses all responses and disjoint positions; legacy panels split responses."""
    result = {"train": [], "heldout": []}
    batch64 = config.get("batch_size") == 64
    train_count = config.get("train_positions_per_response", 16)
    groups = (
        ("train", range(16) if batch64 else range(4), train_count),
        (
            "heldout",
            range(16) if batch64 else range(4, 8),
            config.get("heldout_positions_per_response", 16 if batch64 else 32),
        ),
    )
    if batch64 and len(artifact["sequences"]) != 16:
        raise ValueError("Batch 64 requires exactly 16 cached responses per rank")
    for group, indices, count in groups:
        for index in indices:
            sequence = artifact["sequences"][index]
            size = len(sequence["selected"])
            order = torch.randperm(size, generator=torch.Generator().manual_seed(4242 + index))
            offset = train_count if batch64 and group == "heldout" else 0
            chosen = order[offset : min(size, offset + count)].sort().values
            if not len(chosen):
                raise ValueError("Insufficient disjoint heldout positions in response cache")
            qlog = sequence["q_logp_bf16"][chosen].clone().double()
            tlog = sequence["teacher_logp"][chosen].clone().double()
            qlog -= qlog.logsumexp(-1, keepdim=True)
            tlog -= tlog.logsumexp(-1, keepdim=True)
            mlog = power_log_target(qlog, tlog, config["kappa"], config["strength"])
            result[group].append(
                {
                    "response_index": index,
                    "ids": sequence["ids"].clone(),
                    "start": sequence["start"],
                    "selected": sequence["selected"][chosen].clone(),
                    "response_tokens": sequence["response_tokens"],
                    "qlog": qlog.to(device),
                    "tlog": tlog.to(device),
                    "mlog": mlog.to(device),
                }
            )
    return result


def measure(model, sequences, config, *, native, group="train"):
    """Read exact conditional losses at fixed states without changing parameters."""
    model.train(native)
    sums = {}
    for sequence in sequences:
        with torch.no_grad():
            logits = model(
                sequence["ids"].cuda().unsqueeze(0), sequence["start"], sequence["selected"].cuda()
            )
            logp = logits.double().log_softmax(-1)
            qlog, tlog, mlog = (sequence[key] for key in ("qlog", "tlog", "mlog"))
            fields = {
                "fit_divergence": target_divergence(
                    logp,
                    mlog,
                    config.get("fit_kappa", config["kappa"]),
                    config["fit"],
                    anchor_logp=qlog,
                ),
                "teacher_reverse_kl": target_divergence(logp, tlog, 0),
                "teacher_forward_kl": target_divergence(tlog, logp, 0),
                "anchor_reverse_kl": target_divergence(logp, qlog, 0),
                "anchor_to_student_kl": target_divergence(qlog, logp, 0),
                "target_reverse_kl": target_divergence(logp, mlog, 0),
                "matched_target_divergence": target_divergence(
                    logp, mlog, config["kappa"], "power"
                ),
                "matched_teacher_divergence": target_divergence(
                    logp, tlog, config["kappa"], "power"
                ),
                "teacher_reverse_kl_upper_certificate": renyi_inner_bound(logp, mlog, tlog),
            }
            fields["normalized_fit_divergence"] = fields["fit_divergence"] / config["strength"]
            probability, target, teacher = logp.exp(), mlog.exp(), tlog.exp()
            midpoint = torch.logaddexp(logp, tlog) - math.log(2)
            fields.update(
                {
                    "target_teacher_reverse_kl": (target * (mlog - tlog)).sum(-1),
                    "target_teacher_forward_kl": (teacher * (tlog - mlog)).sum(-1),
                    "target_anchor_fisher_rao_squared": 4
                    * torch.exp((mlog + qlog) / 2).sum(-1).clamp(0, 1).acos().square(),
                    "teacher_entropy": -(teacher * tlog).sum(-1),
                    "student_entropy": -(probability * logp).sum(-1),
                    "teacher_total_variation": (probability - teacher).abs().sum(-1) / 2,
                    "teacher_jensen_shannon": (
                        (probability * (logp - midpoint)).sum(-1)
                        + (teacher * (tlog - midpoint)).sum(-1)
                    )
                    / 2,
                    "teacher_hellinger_squared": (probability.sqrt() - teacher.sqrt())
                    .square()
                    .sum(-1),
                    "target_forward_kl": (target * (mlog - logp)).sum(-1),
                    "anchor_fisher_rao_squared": 4
                    * torch.exp((logp + qlog) / 2).sum(-1).clamp(0, 1).acos().square(),
                }
            )
            if config.get("diagnostic_output"):
                from pathlib import Path

                row = {
                    "update": config.get("diagnostic_update", 0),
                    "response_index": dist.get_rank() * 16 + sequence["response_index"],
                    "prefix_count": len(logp),
                    **{key: float(value.mean()) for key, value in fields.items()},
                }
                suffix = "native" if native else "fp32-converted"
                path = (
                    Path(config["diagnostic_output"])
                    / f"{group}-{suffix}-responses-rank{dist.get_rank()}.jsonl"
                )
                with path.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            for key, values in fields.items():
                sums[key] = sums.get(key, 0.0) + float(values.sum())
            sums["count"] = sums.get("count", 0.0) + len(logp)
    return reduce_sums(sums)


def record(output, filename, value, run):
    if dist.get_rank() == 0:
        with (output / filename).open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        if run is not None:
            update = value["update"]
            scope = filename.removesuffix(".jsonl")
            run.log(
                {
                    scope + "/updates": update,
                    **{
                        scope + "/" + key: item
                        for key, item in value.items()
                        if isinstance(item, (int, float))
                    },
                }
            )


def backward_once(model, sequences, config, count, update):
    rank = dist.get_rank()
    model.train()
    model.zero_grad(set_to_none=True)
    objective = 0.0
    ratio_second = 0.0
    noise = {
        "coefficient_second": 0.0,
        "coefficient_fourth": 0.0,
        "importance_fourth": 0.0,
        "importance_gt10_fraction": 0.0,
    }
    for index, sequence in enumerate(sequences):
        with contextlib.nullcontext() if index == len(sequences) - 1 else model.no_sync():
            logits = model(
                sequence["ids"].cuda().unsqueeze(0),
                sequence["start"],
                sequence["selected"].cuda(),
            )
            if config["sample_count"]:
                generator = torch.Generator(device=torch.device("cuda", rank)).manual_seed(
                    42 + rank * 100003 + update * 1009 + index * 17
                )
                proposal = (
                    logits.float().log_softmax(-1).detach()
                    if config.get("sampler", "anchor") == "student"
                    else sequence["qlog"]
                )
                actions = torch.multinomial(
                    proposal.exp(),
                    config["sample_count"],
                    replacement=True,
                    generator=generator,
                )
                fields = sampled_target_loss(
                    logits,
                    proposal,
                    sequence["mlog"],
                    actions,
                    config.get("fit_kappa", config["kappa"]),
                    config["strength"],
                    config["fit"],
                )
                loss = fields["loss"]
                coefficient, ratio = (
                    fields["coefficient"].detach().double(),
                    fields["importance_ratio"].detach().double(),
                )
                noise["coefficient_second"] += float(coefficient.square().mean(-1).sum())
                noise["coefficient_fourth"] += float(coefficient.pow(4).mean(-1).sum())
                noise["importance_fourth"] += float(ratio.pow(4).mean(-1).sum())
                noise["importance_gt10_fraction"] += float((ratio > 10).double().mean(-1).sum())
                ratio_second += float(fields["importance_ratio"].square().mean(-1).sum())
            else:
                loss = (
                    target_divergence(
                        logits.float().log_softmax(-1),
                        sequence["mlog"],
                        config.get("fit_kappa", config["kappa"]),
                        config["fit"],
                        anchor_logp=sequence["qlog"],
                    )
                    / config["strength"]
                )
            if not torch.isfinite(loss).all():
                raise FloatingPointError("A power-target fitting loss is nonfinite")
            objective += float(loss.detach().sum())
            (loss.sum() * dist.get_world_size() / float(count)).backward()
    return objective, ratio_second, noise


def gradient_probe(model, sequences, config, count, output, stage="initial"):
    """Measure full-parameter gradient covariance before any optimizer update."""
    draws = config["gradient_probe_draws"]
    backward_once(model, sequences, config | {"sample_count": 0}, count, 0)
    exact = {
        name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None
    }
    mean = {name: torch.zeros_like(value) for name, value in exact.items()}
    scalar = torch.zeros(2, device="cuda", dtype=torch.float64)
    samples = []
    for repeat in range(draws):
        backward_once(model, sequences, config | {"sample_count": 4}, count, repeat)
        norm2 = torch.zeros((), device="cuda", dtype=torch.float64)
        error2 = torch.zeros_like(norm2)
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            current = parameter.grad.detach()
            mean[name] += current / draws
            norm2 += current.double().square().sum()
            error2 += (current - exact[name]).double().square().sum()
        scalar += torch.stack((norm2, error2))
        pair = torch.stack((norm2, error2))
        dist.all_reduce(pair)
        samples.append(
            {
                "draw": repeat,
                "gradient_norm_squared": float(pair[0]),
                "squared_error_to_full_vocab_backward": float(pair[1]),
            }
        )
    totals = torch.zeros(3, device="cuda", dtype=torch.float64)
    for name, reference in exact.items():
        totals[0] += reference.double().square().sum()
        totals[1] += mean[name].double().square().sum()
        totals[2] += (mean[name] - reference).double().square().sum()
    dist.all_reduce(totals)
    dist.all_reduce(scalar)
    variance = (float(scalar[0]) - draws * float(totals[1])) / (draws - 1)
    if not torch.isfinite(scalar).all() or not torch.isfinite(totals).all():
        raise FloatingPointError("Parameter gradient probe is nonfinite")
    if variance < -1e-9:
        raise RuntimeError("Parameter covariance trace is numerically invalid")
    if dist.get_rank() == 0:
        result = {
            "draws": draws,
            "sample_count": 4,
            "exact_gradient_norm_squared": float(totals[0]),
            "empirical_parameter_covariance_trace": max(variance, 0),
            "mean_squared_error_to_full_vocab_backward": float(scalar[1]) / draws,
            "squared_sample_mean_error": float(totals[2]),
            "samples": samples,
            "stage": stage,
            "relative_parameter_noise": max(variance, 0) / max(float(totals[0]), 1e-30),
            "scope": "Instrument: fresh auxiliary actions at fixed parameters compared to full-vocabulary backward, including BF16 rounding. Does not imply that main training uses sampled actions; exact-expectation BPO has no conditional action sampling noise.",
        }
        (
            output / ("gradient-probe.json" if stage == "initial" else "gradient-probe-final.json")
        ).write_text(json.dumps(result, indent=2) + "\n")
    del exact, mean
    model.zero_grad(set_to_none=True)
    gc.collect()


def train(args, config, rank):
    cache = torch.load(args.cache / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
    vocab = cache["vocab"]
    data = panels(cache, config)
    del cache
    gc.collect()
    tokenizer, _, _ = one_prompt(args.repo)
    hf = AutoModelForCausalLM.from_pretrained(
        model_path(args.repo, "student"),
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if rank == 0:
        (args.output / "model-info.json").write_text(
            json.dumps(
                {
                    "parameter_count": sum(p.numel() for p in hf.parameters()),
                    "student": str(model_path(args.repo, "student")),
                    "vocabulary": vocab,
                    "master_dtype": "float32",
                    "native_forward_dtype": "bfloat16",
                    "physical_gpus": [0, 1, 2, 3],
                },
                indent=2,
            )
            + "\n"
        )
    os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = "1"
    model = FullyShardedDataParallel(
        ResponseModel(hf, vocab),
        device_id=rank,
        auto_wrap_policy=ModuleWrapPolicy({Qwen2DecoderLayer}),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        ),
        use_orig_params=True,
        sync_module_states=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    count = torch.tensor(
        sum(len(s["selected"]) for s in data["train"]), dtype=torch.float64, device="cuda"
    )
    dist.all_reduce(count)
    run = None
    if rank == 0 and config["wandb"]:
        import wandb

        run = wandb.init(
            entity="wncfht",
            project=config["wandb_project"],
            id=hashlib.sha256(config["id"].encode()).hexdigest()[:8],
            name=config["id"],
            group="power-mean-20261013",
            config=config,
            dir=str(args.output),
            resume="allow",
        )
        (args.output / "wandb-run.json").write_text(
            json.dumps({"id": run.id, "url": run.url}) + "\n"
        )
    config = config | {"diagnostic_output": str(args.output), "diagnostic_update": 0}
    initial = measure(model, data["train"], config, native=True)
    if initial["anchor_to_student_kl"] > 1e-5:
        raise RuntimeError("Native full-model forward does not reproduce the frozen anchor")
    record(args.output, "train-native.jsonl", {"update": 0, **initial}, run)
    record(
        args.output,
        "heldout-native.jsonl",
        {"update": 0, **measure(model, data["heldout"], config, native=True, group="heldout")},
        run,
    )
    for group in ("train", "heldout"):
        record(
            args.output,
            group + "-fp32-converted.jsonl",
            {"update": 0, **measure(model, data[group], config, native=False, group=group)},
            run,
        )
    if config.get("gradient_probe_draws", 0):
        gradient_probe(model, data["train"], config, count, args.output)
    begin = time.monotonic()
    for update in range(config["updates"]):
        objective, ratio_second, noise = backward_once(model, data["train"], config, count, update)
        norm = float(model.clip_grad_norm_(float("inf")))
        if not math.isfinite(norm):
            raise FloatingPointError("Power-target parameter gradient is nonfinite")
        optimizer.step()
        metrics = reduce_sums(
            {
                "count": float(count) / 4,
                "objective": objective,
                **({"importance_second": ratio_second, **noise} if config["sample_count"] else {}),
            }
        )
        record(
            args.output,
            "updates.jsonl",
            {
                "update": update + 1,
                "grad_norm": norm,
                "peak_allocated_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_memory_gib": torch.cuda.max_memory_reserved() / 2**30,
                "response_batch_size": config.get("batch_size", 16),
                "prefix_batch_size": int(count),
                "action_expectation": "fresh_auxiliary_mc"
                if config["sample_count"]
                else "exact_full_vocabulary",
                "objective_kind": "score_surrogate"
                if config["sample_count"]
                else "exact_normalized_loss",
                "seconds_elapsed": time.monotonic() - begin,
                **metrics,
            },
            run,
        )
        if (update + 1) % 10 == 0 or update + 1 == config["updates"]:
            config["diagnostic_update"] = update + 1
            for group in ("train", "heldout"):
                record(
                    args.output,
                    group + "-native.jsonl",
                    {
                        "update": update + 1,
                        **measure(model, data[group], config, native=True, group=group),
                    },
                    run,
                )
                record(
                    args.output,
                    group + "-fp32-converted.jsonl",
                    {
                        "update": update + 1,
                        **measure(model, data[group], config, native=False, group=group),
                    },
                    run,
                )
        if rank == 0:
            (args.output / "progress.json").write_text(
                json.dumps(
                    {
                        "updates_completed": update + 1,
                        "updates_requested": config["updates"],
                        "state": "running",
                    }
                )
                + "\n"
            )
    if config.get("gradient_probe_draws", 0):
        gradient_probe(model, data["train"], config, count, args.output, stage="final")
    save_checkpoint(
        args.repo,
        args.output,
        model,
        optimizer,
        tokenizer,
        data["train"],
        config["updates"],
        config,
    )
    dist.barrier()
    if rank == 0:
        (args.output / "completion.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "updates_completed": config["updates"],
                    "seconds_elapsed": time.monotonic() - begin,
                }
            )
            + "\n"
        )
        if run is not None:
            run.finish()


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train",), required=True)
    for key in ("config", "output", "owners", "repo", "cache"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3" or rank not in range(4):
        raise RuntimeError("Power workers require precisely physical GPUs 0--3")
    torch.cuda.set_device(rank)
    torch.empty(1, device="cuda")
    register_gpu_owner(args.owners, rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    if dist.get_world_size() != 4:
        raise RuntimeError("The power study uses precisely physical GPUs 0--3")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    if config["fit"] == "bpo" and config.get("sampler", "anchor") != "anchor":
        raise ValueError("This BPO estimator requires frozen-anchor action sampling")
    torch.manual_seed(config["seed"])
    train(args, config, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
