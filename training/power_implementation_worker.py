"""Four-H800 correctness and fitting experiments for the production mirror API."""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.rollout.oneshot_batch import model_path
from mirroropd.integrations.verl.fsdp import compute_mirror_opd_from_cache
from mirroropd.proximal import power_log_target
from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.losses import full_vocab_mirror_loss
from training.gpu_exclusive import register_gpu_owner
from training.oneshot_worker import ResponseModel, reduce_sums


def build_cache(qlog, tlog, actions, config):
    return build_mirror_opd_cache_from_full_logits(
        qlog,
        tlog,
        actions,
        torch.ones_like(actions, dtype=torch.bool),
        config["topj"],
        center_mode=config["center_mode"],
        kappa=config["kappa"],
        eta=config["eta"],
    )


def production_fields(logits, cache, config):
    settings = SimpleNamespace(
        loss_mode="mirror_" + config["method"],
        mirror_eta=config["eta"],
        mirror_kappa=config["kappa"],
        mirror_center_mode=config["center_mode"],
        mirror_action_mode=config["action_mode"],
    )
    values = {key: value.unsqueeze(0) for key, value in cache.items()}
    output = compute_mirror_opd_from_cache(
        logits.unsqueeze(0), values, SimpleNamespace(distillation_loss=settings), "bshd"
    )
    return {key: value.squeeze(0) for key, value in output.items()}


def load_data(args, config, rank):
    source = args.cache / f"rank-{rank}.pt"
    manifest = json.loads((args.cache / "manifest.json").read_text())
    with source.open("rb") as stream:
        cache_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if cache_sha256 != manifest["rank_sha256"][str(rank)]:
        raise RuntimeError("Historical model cache does not match its manifest")
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    data = []
    train_count, heldout_count = config["train_prefixes"], config["heldout_prefixes"]
    for index, original in enumerate(artifact["sequences"][: config["responses_per_rank"]]):
        count = len(original["selected"])
        order = torch.randperm(
            count, generator=torch.Generator().manual_seed(4242 + 16 * rank + index)
        )
        if count < train_count + heldout_count:
            raise ValueError("Insufficient disjoint cached prefixes for this experiment")
        groups = {}
        for group, chosen in (
            ("train", order[:train_count].sort().values),
            ("heldout", order[train_count : train_count + heldout_count].sort().values),
        ):
            selected = original["selected"][chosen]
            qlog = original["q_logp_bf16"][chosen].float().cuda()
            tlog = original["teacher_logp"][chosen].float().cuda()
            qlog = qlog - qlog.logsumexp(-1, keepdim=True)
            tlog = tlog - tlog.logsumexp(-1, keepdim=True)
            actions = original["ids"][original["start"] + 1 + selected].cuda()
            groups[group] = {
                "selected": selected.cuda(),
                "qlog": qlog,
                "tlog": tlog,
                "initial_qlog": qlog.clone(),
                "actions": actions,
                "cache": build_cache(qlog, tlog, actions, config),
            }
        data.append(
            {"ids": original["ids"].cuda().unsqueeze(0), "start": original["start"], **groups}
        )
    vocab = artifact["vocab"]
    del artifact
    gc.collect()
    metadata = {
        "rank": rank,
        "cache": str(source),
        "cache_sha256": cache_sha256,
        "responses": len(data),
        "vocabulary": vocab,
        "train_prefixes": len(data) * train_count,
        "heldout_prefixes": len(data) * heldout_count,
        "response_lengths": [int(row["ids"].shape[-1] - row["start"] - 1) for row in data],
        "config": config,
    }
    (args.output / f"input-rank-{rank}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return data, vocab


def emit(output, name, value):
    if dist.get_rank() == 0:
        with (output / name).open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")


def audit(args, config, rank):
    data, _ = load_data(args, config, rank)
    rows = []
    for index, sequence in enumerate(data):
        item = sequence["train"]
        qlog, tlog = item["qlog"].double(), item["tlog"].double()
        generator = torch.Generator(device=qlog.device).manual_seed(71 + 100 * rank + index)
        z = (
            qlog
            + 0.2
            * torch.randn(qlog.shape, generator=generator, device=qlog.device, dtype=torch.float64)
        ).requires_grad_()
        for method in ("bpo", "mdpo"):
            target = power_log_target(qlog, tlog, config["kappa"], config["eta"])
            lp = z.log_softmax(-1)
            if method == "bpo":
                residual = lp - target
                residual -= (qlog.exp() * residual).sum(-1, keepdim=True)
                independent = (qlog.exp() * residual.square()).sum() / (2 * config["eta"])
            else:
                independent = (lp.exp() * (lp - target)).sum() / config["eta"]
            oracle = torch.autograd.grad(independent, z, retain_graph=True)[0]
            exact = full_vocab_mirror_loss(z, qlog, tlog, method, config["eta"], config["kappa"])
            gradient = torch.autograd.grad(exact.sum(), z, retain_graph=True)[0]
            error = float((gradient - oracle).abs().max())
            if error > 1e-10:
                raise RuntimeError(f"Independent complete-vocabulary gradient failed: {error}")
            for topj in (0, 16, 64, 256, qlog.shape[-1]):
                options = config | {
                    "method": method,
                    "topj": topj,
                    "center_mode": "topj_tail",
                    "action_mode": "sampled",
                }
                grouped_cache = build_cache(qlog, tlog, item["actions"], options)
                complete_cache = build_cache(
                    qlog, tlog, item["actions"], options | {"center_mode": "full_vocab"}
                )
                grouped = production_fields(z, grouped_cache, options)
                complete = production_fields(
                    z, complete_cache, options | {"center_mode": "full_vocab"}
                )
                kt_error = (
                    grouped_cache["mirror_teacher_center"] - complete_cache["mirror_teacher_center"]
                )
                ks_error = grouped["mirror_student_center"] - complete["mirror_student_center"]
                coefficient_error = (
                    grouped["mirror_score_coefficient"] - complete["mirror_score_coefficient"]
                )
                expected = (
                    ks_error / config["eta"] - kt_error
                    if method == "bpo"
                    else torch.zeros_like(kt_error)
                )
                if not torch.allclose(coefficient_error, expected, atol=1e-10, rtol=1e-9):
                    raise RuntimeError(
                        "Grouped coefficient bias does not match its analytic formula"
                    )
                if topj == qlog.shape[-1] and (
                    kt_error.abs().max() > 1e-10 or ks_error.abs().max() > 1e-10
                ):
                    raise RuntimeError("Full head does not recover complete-vocabulary centers")
                cache_bytes = sum(
                    value.numel() * value.element_size() for value in grouped_cache.values()
                )
                rows.append(
                    {
                        "rank": rank,
                        "response": index + 16 * rank,
                        "prefixes": len(qlog),
                        "method": method,
                        "topj": topj,
                        "teacher_center_mae": float(kt_error.abs().mean()),
                        "student_center_mae": float(ks_error.abs().mean()),
                        "score_coefficient_mae": float(coefficient_error.abs().mean()),
                        "teacher_lower_bound_max_violation": float(kt_error.max()),
                        "full_gradient_max_error": error,
                        "compact_cache_bytes": cache_bytes,
                        "full_cache_bytes": sum(
                            v.numel() * v.element_size() for v in complete_cache.values()
                        ),
                    }
                )
            del z.grad
        del z
    (args.output / f"audit-rank-{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
    dist.barrier()
    emit(args.output, "completion.jsonl", {"phase": "audit", "state": "completed", "ranks": 4})


def measure(model, data, config):
    model.train()
    sums = {}
    with torch.no_grad():
        for sequence in data:
            for group in ("train", "heldout"):
                item = sequence[group]
                logits = model(sequence["ids"], sequence["start"], item["selected"])
                lp = logits.double().log_softmax(-1)
                qlog, tlog = item["qlog"].double(), item["tlog"].double()
                target = power_log_target(qlog, tlog, config["kappa"], config["eta"])
                values = {
                    "loss": full_vocab_mirror_loss(
                        lp, qlog, tlog, config["method"], config["eta"], config["kappa"]
                    ),
                    "teacher_reverse_kl": (lp.exp() * (lp - tlog)).sum(-1),
                    "anchor_student_kl": (qlog.exp() * (qlog - lp)).sum(-1),
                    "target_reverse_kl": (lp.exp() * (lp - target)).sum(-1),
                }
                for key, value in values.items():
                    sums[group + "/" + key] = sums.get(group + "/" + key, 0.0) + float(value.sum())
            sums["count"] = sums.get("count", 0.0) + len(sequence["train"]["selected"])
    # Train and heldout use equal prefix counts in this controlled protocol.
    if config["train_prefixes"] != config["heldout_prefixes"]:
        raise ValueError("Separate train/heldout means require equal diagnostic prefix counts")
    return reduce_sums(sums)


def save_local_checkpoint(model, optimizer, data, args, update):
    """Save exact same-topology FSDP parameter and Adam shards, plus frozen caches."""
    parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    anchors = [
        {
            group: {key: row[group][key].cpu() for key in ("qlog", "actions")}
            for group in ("train", "heldout")
        }
        for row in data
    ]
    path = args.output / f"checkpoint-rank-{dist.get_rank()}.pt"
    torch.save(
        {
            "parameters": parameters,
            "optimizer": optimizer.state_dict(),
            "anchors": anchors,
            "update": update,
            "world_size": dist.get_world_size(),
            "config": json.loads(args.config.read_text()),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        path,
    )
    return path


def restore_local_checkpoint(model, optimizer, data, args, config):
    saved = torch.load(
        args.output / f"checkpoint-rank-{dist.get_rank()}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if saved["world_size"] != dist.get_world_size() or saved["config"] != config:
        raise ValueError("Checkpoint topology or experiment configuration does not match")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(saved["parameters"][name].to(parameter.device))
            if not torch.equal(parameter.detach().cpu(), saved["parameters"][name]):
                raise RuntimeError("Restored parameter shard differs from its checkpoint")
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    restored_optimizer = optimizer.state_dict()
    for key, state in saved["optimizer"]["state"].items():
        for field, value in state.items():
            actual = restored_optimizer["state"][key][field]
            if isinstance(value, torch.Tensor):
                if not torch.equal(actual.cpu(), value.cpu()):
                    raise RuntimeError("Restored Adam state differs from its checkpoint")
            elif actual != value:
                raise RuntimeError("Restored optimizer metadata differs from its checkpoint")
    for row, anchors in zip(data, saved["anchors"], strict=True):
        for group in ("train", "heldout"):
            item = row[group]
            item.update({key: value.cuda() for key, value in anchors[group].items()})
            item["cache"] = build_cache(item["qlog"], item["tlog"], item["actions"], config)
    return saved["update"]


def backward_pass(model, data, config, count, *, independent=False):
    model.train()
    model.zero_grad(set_to_none=True)
    total = 0.0
    for index, row in enumerate(data):
        item = row["train"]
        with contextlib.nullcontext() if index == len(data) - 1 else model.no_sync():
            logits = model(row["ids"], row["start"], item["selected"])
            if independent:
                logp = logits.float().log_softmax(-1)
                target = power_log_target(
                    item["qlog"], item["tlog"], config["kappa"], config["eta"]
                )
                if config["method"] == "mdpo":
                    loss = (logp.exp() * (logp - target)).sum(-1) / config["eta"]
                else:
                    residual = logp - target
                    residual = residual - (item["qlog"].exp() * residual).sum(-1, keepdim=True)
                    loss = (item["qlog"].exp() * residual.square()).sum(-1) / (2 * config["eta"])
            else:
                loss = production_fields(logits, item["cache"], config)["distillation_losses"]
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite production loss")
            total += float(loss.detach().sum())
            (loss.sum() * dist.get_world_size() / count).backward()
    norm = float(model.clip_grad_norm_(float("inf")))
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite global gradient norm")
    return total, norm


def check_parameter_gradients(model, data, config, count, output, stage):
    backward_pass(model, data, config, count)
    production = {
        name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None
    }
    backward_pass(model, data, config, count, independent=True)
    moments = torch.zeros(4, device="cuda", dtype=torch.float64)
    maximum = torch.zeros((), device="cuda", dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        actual, expected = production[name].double(), parameter.grad.detach().double()
        moments += torch.stack(
            (
                actual.square().sum(),
                expected.square().sum(),
                (actual - expected).square().sum(),
                (actual * expected).sum(),
            )
        )
        if actual.numel():
            maximum = torch.maximum(maximum, (actual - expected).abs().max())
    dist.all_reduce(moments)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    relative = math.sqrt(float(moments[2] / moments[1].clamp_min(1e-30)))
    cosine = float(moments[3] / (moments[0] * moments[1]).sqrt().clamp_min(1e-30))
    threshold = 1e-4 if config.get("precision") == "fp32" else 0.03
    minimum_cosine = 1 - 1e-7 if config.get("precision") == "fp32" else 0.999
    if relative > threshold or cosine < minimum_cosine:
        raise RuntimeError(
            f"Full-parameter BF16 gradient mismatch: relative={relative}, cosine={cosine}"
        )
    emit(
        output,
        "parameter-gradient-check.jsonl",
        {
            "stage": stage,
            "relative_l2_error": relative,
            "cosine": cosine,
            "max_absolute_error": float(maximum),
            "production_norm": math.sqrt(float(moments[0])),
            "reference_norm": math.sqrt(float(moments[1])),
            "scope": "All FSDP parameter shards, configured forward precision, FP32 master/reduction",
            "precision": config.get("precision", "bf16"),
        },
    )
    del production
    model.zero_grad(set_to_none=True)


def train(args, config, rank):
    data, vocab = load_data(args, config, rank)
    hf = AutoModelForCausalLM.from_pretrained(
        model_path(args.repo, "student"),
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    forward_dtype = torch.float32 if config.get("precision") == "fp32" else torch.bfloat16
    if rank == 0:
        (args.output / "model-info.json").write_text(
            json.dumps(
                {
                    "parameters": sum(parameter.numel() for parameter in hf.parameters()),
                    "student": str(model_path(args.repo, "student")),
                    "vocabulary": vocab,
                    "master_dtype": "float32",
                    "forward_dtype": str(forward_dtype),
                    "gradient_reduction_dtype": "float32",
                    "deterministic_algorithms": True,
                },
                indent=2,
            )
            + "\n"
        )
    model = FullyShardedDataParallel(
        ResponseModel(hf, vocab),
        device_id=rank,
        auto_wrap_policy=ModuleWrapPolicy({Qwen2DecoderLayer}),
        mixed_precision=MixedPrecision(
            param_dtype=forward_dtype, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        ),
        use_orig_params=True,
        sync_module_states=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.0, foreach=False
    )
    count = torch.tensor(
        sum(len(row["train"]["selected"]) for row in data), device="cuda", dtype=torch.float64
    )
    dist.all_reduce(count)
    if forward_dtype == torch.float32:
        model.train()
        with torch.no_grad():
            for row in data:
                for group in ("train", "heldout"):
                    item = row[group]
                    item["qlog"] = (
                        model(row["ids"], row["start"], item["selected"])
                        .float()
                        .log_softmax(-1)
                        .detach()
                    )
                    item["cache"] = build_cache(item["qlog"], item["tlog"], item["actions"], config)
    before = measure(model, data, config)
    if abs(before["train/anchor_student_kl"]) > 1e-5:
        raise RuntimeError("Native forward does not reproduce the historical frozen anchor")
    emit(args.output, "measurements.jsonl", {"update": 0, **before})
    if config.get("parameter_gradient_check"):
        check_parameter_gradients(model, data, config, count, args.output, "initial")
    begin = time.monotonic()
    for update in range(1, config["updates"] + 1):
        total, norm = backward_pass(model, data, config, count)
        optimizer.step()
        emit(
            args.output,
            "updates.jsonl",
            {
                "update": update,
                "grad_norm": norm,
                "seconds_elapsed": time.monotonic() - begin,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                **reduce_sums({"count": float(count) / 4, "proxy": total}),
            },
        )
        if update % config.get("measure_every", 5) == 0 or update == config["updates"]:
            emit(
                args.output,
                "measurements.jsonl",
                {"update": update, **measure(model, data, config)},
            )
        if update == config.get("anchor_refresh", -1):
            with torch.no_grad():
                for row in data:
                    for group in ("train", "heldout"):
                        item = row[group]
                        logits = model(row["ids"], row["start"], item["selected"])
                        item["qlog"] = logits.float().log_softmax(-1).detach()
                        item["actions"] = torch.multinomial(item["qlog"].exp(), 1).squeeze(-1)
                        item["cache"] = build_cache(
                            item["qlog"], item["tlog"], item["actions"], config
                        )
            refreshed = measure(model, data, config)
            if abs(refreshed["train/anchor_student_kl"]) > 1e-6:
                raise RuntimeError("Refreshed anchor does not match current native model")
            emit(
                args.output,
                "anchor-refresh.jsonl",
                {
                    "update": update,
                    "scope": "Fixed-prefix anchor refresh; no new response generation",
                    **refreshed,
                },
            )
    final = measure(model, data, config)
    if config.get("parameter_gradient_check"):
        check_parameter_gradients(model, data, config, count, args.output, "after_updates")
    if config.get("checkpoint_roundtrip"):
        save_local_checkpoint(model, optimizer, data, args, config["updates"])
        backward_pass(model, data, config, count)
        optimizer.step()
        continuation = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.001)
        restored_step = restore_local_checkpoint(model, optimizer, data, args, config)
        restored = measure(model, data, config)
        error = max(abs(restored[key] - final[key]) for key in final)
        if error > 1e-10 or restored_step != config["updates"]:
            raise RuntimeError("Checkpoint roundtrip changed model diagnostics")
        backward_pass(model, data, config, count)
        optimizer.step()
        maximum = 0.0
        for name, parameter in model.named_parameters():
            if parameter.numel():
                maximum = max(
                    maximum, float((parameter.detach().cpu() - continuation[name]).abs().max())
                )
        value = torch.tensor(maximum, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        if value.item() != 0:
            raise RuntimeError("Restored checkpoint does not reproduce the next optimizer update")
        emit(
            args.output,
            "checkpoint-roundtrip.jsonl",
            {
                "update": restored_step,
                "diagnostic_max_error": error,
                "same_topology": True,
                "continuation_parameter_max_error": value.item(),
                "extra_verification_updates": 2,
            },
        )
    dist.barrier()
    emit(
        args.output,
        "completion.jsonl",
        {
            "state": "completed",
            "phase": "train",
            "updates": config["updates"],
            "seconds": time.monotonic() - begin,
            "final": final,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "train"), required=True)
    for name in ("config", "output", "owners", "repo", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rank = int(os.environ["LOCAL_RANK"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3" or rank not in range(4):
        raise RuntimeError("Workers require precisely physical GPUs 0--3")
    torch.cuda.set_device(rank)
    torch.empty(1, device="cuda")
    register_gpu_owner(args.owners, rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    if dist.get_world_size() != 4:
        raise RuntimeError("The experiment requires four workers")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    torch.manual_seed(config["seed"])
    (args.output / f"worker-source-rank-{rank}.json").write_text(
        json.dumps({"sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}) + "\n"
    )
    (audit if args.phase == "audit" else train)(args, config, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
