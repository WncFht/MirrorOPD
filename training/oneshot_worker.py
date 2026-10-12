"""Four-GPU controlled One-Shot experiments using the production mirror kernel."""

import argparse
import contextlib
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from experiments.rollout.oneshot_batch import (
    generate_batch,
    load_model,
    model_path,
    one_prompt,
    prepare_batch,
    score_teacher,
    select_positions,
)
from mirroropd.training.cache import build_mirror_anchor_cache
from mirroropd.training.losses import compute_mirror_opd_token_loss
from mirroropd.training.metrics import token_moments
from training.gpu_exclusive import register_gpu_owner


class ResponseModel(torch.nn.Module):
    """Keep the backbone and response head inside one FSDP root forward."""

    def __init__(self, model, vocab):
        """Retain the original HF model so checkpoint keys stay recoverable."""
        super().__init__()
        self.hf = model
        self.vocab = vocab

    def forward(self, ids, start, positions=None):
        """Select predecessor positions before projecting onto the vocabulary."""
        hidden = self.hf.model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).last_hidden_state[0, start:-1]
        if positions is not None:
            hidden = hidden[positions]
        return self.hf.lm_head(hidden)[..., : self.vocab]


def distribution_sums(logits, qlog, tlog, actions, eta):
    """Full-vocabulary diagnostics with an explicit fixed-prefix measure."""
    with torch.no_grad():
        logp = logits.detach().float().log_softmax(-1)
        qlog, tlog = qlog.to(logits.device), tlog.to(logits.device)
        q, p = qlog.exp(), logp.exp()
        target = ((1 - eta) * qlog + eta * tlog).log_softmax(-1)
        delta = logp - target
        residual = delta - (q * delta).sum(-1, keepdim=True)
        selected = actions.to(logits.device).unsqueeze(-1)
        gap = (tlog - logp).gather(-1, selected).squeeze(-1)
        teacher_ids = tlog.topk(16, dim=-1).indices
        student_ids = logp.topk(16, dim=-1).indices
        values = {
            "anchor_teacher_full_kl": (q * (qlog - tlog)).sum(-1),
            "student_teacher_full_kl": (p * (logp - tlog)).sum(-1),
            "teacher_student_full_kl": (tlog.exp() * (tlog - logp)).sum(-1),
            "anchor_student_full_kl": (q * (qlog - logp)).sum(-1),
            "mirror_target_full_kl": (p * delta).sum(-1),
            "bpo_target_variance": (q * residual.square()).sum(-1) / (2 * eta),
            "fixed_action_abs_gap": gap.abs(),
            "fixed_action_signed_gap": gap,
            "teacher_logprob_regression": (q * (logp - tlog).square()).sum(-1) / 2,
            "top16_teacher_overlap": student_ids.unsqueeze(-1)
            .eq(teacher_ids.unsqueeze(-2))
            .any(-1)
            .float()
            .mean(-1),
        }
        return {key: float(value.double().sum()) for key, value in values.items()} | {
            "count": len(logp)
        }


def reduce_sums(sums):
    """Reduce sufficient statistics before computing any means."""
    keys = sorted(sums)
    values = torch.tensor([sums[key] for key in keys], device="cuda", dtype=torch.float64)
    dist.all_reduce(values)
    result = dict(zip(keys, values.cpu().tolist(), strict=True))
    count = result.pop("count")
    return {key: value / count for key, value in result.items()} | {"diagnostic_positions": count}


def measure_native(model, sequences, config):
    """Measure every action after the final update without another optimizer step."""
    model.train()
    sums = {
        "count": 0.0,
        "action_abs_gap": 0.0,
        "action_signed_gap": 0.0,
        "action_gap_square": 0.0,
        "anchor_student_tail_kl": 0.0,
    }
    for sequence in sequences:
        with torch.no_grad():
            logits = model(sequence["ids"].cuda().unsqueeze(0), sequence["start"])
            cache = {
                key: value.cuda() for key, value in sequence["cache_" + config["precision"]].items()
            }
            actions = cache["mirror_actions"]
            fields = compute_mirror_opd_token_loss(
                logits,
                actions,
                cache["mirror_anchor_ids"],
                cache["mirror_anchor_head_log_probs"],
                cache["mirror_anchor_tail_log_prob"],
                cache["mirror_anchor_action_log_prob"],
                cache["mirror_teacher_action_log_prob"],
                cache["mirror_teacher_center"],
                config["method"] if config["method"] in ("mdpo", "bpo") else "mdpo",
                config["eta"],
            )
            gap = cache["mirror_teacher_action_log_prob"] + fields["mirror_action_nll"]
            sums["count"] += len(gap)
            sums["action_abs_gap"] += float(gap.double().abs().sum())
            sums["action_signed_gap"] += float(gap.double().sum())
            sums["action_gap_square"] += float(gap.double().square().sum())
            sums["anchor_student_tail_kl"] += float(fields["mirror_student_center"].double().sum())
    result = reduce_sums(sums)
    result["action_gap_rms"] = result.pop("action_gap_square") ** 0.5
    return result


def evaluate_fixed(model, sequences, precision, eta, *, native=False):
    """Use FP32 evaluation independently of the training forward precision."""
    model.train(native)
    accumulated = {}
    for sequence in sequences:
        positions = sequence["selected"].cuda()
        with torch.no_grad():
            logits = model(sequence["ids"].cuda().unsqueeze(0), sequence["start"], positions)
            expected_dtype = torch.bfloat16 if native and precision == "bf16" else torch.float32
            if logits.dtype != expected_dtype:
                raise RuntimeError("Diagnostic readout precision does not match its declared mode")
            actions = sequence["ids"][sequence["start"] + 1 :][sequence["selected"]]
            values = distribution_sums(
                logits, sequence["q_logp_" + precision], sequence["teacher_logp"], actions, eta
            )
        for key, value in values.items():
            accumulated[key] = accumulated.get(key, 0.0) + value
    return reduce_sums(accumulated)


def current_anchor(model, sequences, precision, topk):
    """Freeze the complete current-batch anchor before an inner update window."""
    model.train()
    for sequence in sequences:
        with torch.no_grad():
            logits = model(sequence["ids"].cuda().unsqueeze(0), sequence["start"])
            actions = sequence["ids"][sequence["start"] + 1 :].cuda()
            cache = build_mirror_anchor_cache(
                logits, actions, torch.ones_like(actions, dtype=torch.bool), topk
            )
            sequence["anchor_" + precision] = {key: value.cpu() for key, value in cache.items()}
            sequence["q_logp_" + precision] = (
                logits.float().log_softmax(-1)[sequence["selected"]].cpu()
            )


def write_record(path, record, wandb_run=None):
    """Append one rank-zero record after global reductions."""
    if dist.get_rank() == 0:
        with path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if wandb_run is not None:
            scope = {
                "updates.jsonl": "train",
                "native-final.jsonl": "native_final",
                "fixed-native.jsonl": "fixed_native",
                "diagnostics.jsonl": "fixed_fp32",
                "inner-window-final.jsonl": "window_final",
            }[path.name]
            update = record["update_before"] if scope == "train" else record["update"]
            payload = {scope + "/optimizer_updates": update}
            for key, value in record["metrics"].items():
                if isinstance(value, dict):
                    payload.update(
                        {scope + "/" + key + "/" + field: number for field, number in value.items()}
                    )
                else:
                    payload[scope + "/" + key] = value
            wandb_run.log(payload)


def record_fixed_readouts(model, sequences, config, output, update, wandb_run):
    """Separate the deployed precision from an FP32 readout of the same weights."""
    eta = 1.0 if config["method"] == "opd" else config["eta"]
    native = evaluate_fixed(model, sequences, config["precision"], eta, native=True)
    fp32 = (
        native
        if config["precision"] == "fp32"
        else evaluate_fixed(model, sequences, config["precision"], eta)
    )
    for filename, measurement, metrics in (
        ("fixed-native.jsonl", "native-fixed-initial-prefixes", native),
        ("diagnostics.jsonl", "fp32-fixed-initial-prefixes", fp32),
    ):
        write_record(
            output / filename,
            {
                "update": update,
                "effective_time": config["lr"] / 1e-6 * update,
                "measurement": measurement,
                "metrics": metrics,
            },
            wandb_run,
        )


def save_checkpoint(root, output, model, optimizer, tokenizer, sequences, step, config):
    """Commit a full HF checkpoint and four matching optimizer shards."""
    directory = output / "checkpoints" / f"update-{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, state_config
    ):
        state = model.state_dict()
    if dist.get_rank() == 0:
        state = {key.removeprefix("hf."): value.contiguous() for key, value in state.items()}
        save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
        AutoConfig.from_pretrained(
            model_path(root, "student"), local_files_only=True
        ).save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        (directory / "experiment.json").write_text(
            json.dumps({"update": step, "config": config}, indent=2) + "\n"
        )
    optimizer_state = optimizer.state_dict()
    optimizer_state["state"] = {
        key: {
            field: value.cpu() if isinstance(value, torch.Tensor) else value
            for field, value in fields.items()
        }
        for key, fields in optimizer_state["state"].items()
    }
    torch.save(optimizer_state, directory / f"optimizer-rank-{dist.get_rank()}.pt")
    if config["anchor_interval"]:
        precision = config["precision"]
        torch.save(
            [
                {key: sequence[key] for key in ("cache_" + precision, "q_logp_" + precision)}
                for sequence in sequences
            ],
            directory / f"anchor-rank-{dist.get_rank()}.pt",
        )
    dist.barrier()
    if dist.get_rank() == 0:
        temporary = output / "resume.tmp"
        temporary.write_text(json.dumps({"update": step, "checkpoint": str(directory)}) + "\n")
        temporary.replace(output / "resume.json")


def step_update(model, optimizer, sequences, config, *, measure_distribution):
    """Accumulate a token-weighted global batch and perform one AdamW update."""
    precision = config["precision"]
    count = torch.tensor(
        sum(sequence["response_tokens"] for sequence in sequences),
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(count)
    total_tokens = float(count)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = {}
    measured = {}
    for index, sequence in enumerate(sequences):
        synchronize = index == len(sequences) - 1
        with contextlib.nullcontext() if synchronize else model.no_sync():
            logits = model(sequence["ids"].cuda().unsqueeze(0), sequence["start"])
            cache = {key: value.cuda() for key, value in sequence["cache_" + precision].items()}
            actions = cache["mirror_actions"]
            with torch.no_grad() if config["method"] == "opd" else contextlib.nullcontext():
                fields = compute_mirror_opd_token_loss(
                    logits,
                    actions,
                    cache["mirror_anchor_ids"],
                    cache["mirror_anchor_head_log_probs"],
                    cache["mirror_anchor_tail_log_prob"],
                    cache["mirror_anchor_action_log_prob"],
                    cache["mirror_teacher_action_log_prob"],
                    cache["mirror_teacher_center"],
                    config["method"] if config["method"] in ("mdpo", "bpo") else "mdpo",
                    config["eta"],
                )
            action_logp = -fields["mirror_action_nll"]
            advantage = cache["mirror_teacher_action_log_prob"] - action_logp
            if config["method"] == "opd":
                # This is MD-OPD's frozen opd_noratio score surrogate. The
                # advantage is current, even though the sampled actions stay fixed.
                action_lp_grad = (
                    logits.float().log_softmax(-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                )
                loss = -advantage.detach() * action_lp_grad
                fields["mirror_score_coefficient"] = -advantage
                fields["mirror_token_advantage"] = -advantage
            else:
                loss = fields["distillation_losses"]
            fields["mirror_current_advantage"] = advantage
            fields["mirror_current_advantage_abs"] = advantage.abs()
            moments = token_moments(fields, torch.ones_like(actions, dtype=torch.bool))
            for key, (reduction, value) in moments.items():
                if reduction == "sum":
                    sums[key] = sums.get(key, 0.0) + value
            if measure_distribution:
                positions = sequence["selected"]
                values = distribution_sums(
                    logits[positions],
                    sequence["q_logp_" + precision],
                    sequence["teacher_logp"],
                    actions[positions],
                    1.0 if config["method"] == "opd" else config["eta"],
                )
                values["anchor_student_tail_kl"] = float(
                    fields["mirror_student_center"][positions].double().sum()
                )
                values["anchor_teacher_tail_kl"] = float(
                    cache["mirror_teacher_center"][positions].double().sum()
                )
                values["anchor_tail_mass"] = float(
                    cache["mirror_anchor_tail_log_prob"][positions].double().exp().sum()
                )
                for key, value in values.items():
                    measured[key] = measured.get(key, 0.0) + value
            (loss.sum() * (dist.get_world_size() / total_tokens)).backward()
        del logits, fields, loss, cache
    norm = float(model.clip_grad_norm_(config["clip_grad"]))
    keys = sorted(sums)
    tensor = torch.tensor([sums[key] for key in keys], device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor)
    merged = dict(zip(keys, tensor.cpu().tolist(), strict=True))
    # finalize_token_moments also expects extrema; they are unnecessary for
    # this experiment's reduced means, so calculate the stored moments directly.
    n = merged["mirror/raw/count"]
    names = {key.split("/")[2] for key in merged if key.endswith("/sum")}
    metrics = {
        "global_tokens": total_tokens,
        "grad_norm": norm,
        "grad_clip_scale": min(1.0, config["clip_grad"] / max(norm, 1e-12)),
    }
    for name in names:
        metrics[name + "/mean"] = merged[f"mirror/raw/{name}/sum"] / n
        metrics[name + "/rms"] = (merged[f"mirror/raw/{name}/sum_sq"] / n) ** 0.5
    metrics["importance_ess_fraction"] = merged["mirror/raw/importance_ratio/sum"] ** 2 / (
        n * merged["mirror/raw/importance_ratio/sum_sq"]
    )
    if measure_distribution:
        diagnostic = reduce_sums(measured)
        diagnostic["student_tail_kl_contribution"] = (
            diagnostic["anchor_student_full_kl"] - diagnostic["anchor_student_tail_kl"]
        )
        diagnostic["teacher_tail_kl_contribution"] = (
            diagnostic["anchor_teacher_full_kl"] - diagnostic["anchor_teacher_tail_kl"]
        )
        diagnostic["student_tail_conditional_kl_mass_weighted"] = (
            diagnostic["student_tail_kl_contribution"] / diagnostic["anchor_tail_mass"]
        )
        diagnostic["teacher_tail_conditional_kl_mass_weighted"] = (
            diagnostic["teacher_tail_kl_contribution"] / diagnostic["anchor_tail_mass"]
        )
        metrics["native_reservoir"] = diagnostic
    inspected = {
        name: (parameter, parameter.detach().clone())
        for name, parameter in model.named_parameters()
        if (".layers.0." in name and name.endswith("self_attn.q_proj.weight"))
        or (".layers.12." in name and name.endswith("mlp.down_proj.weight"))
    }
    optimizer.step()
    rounding = torch.zeros(4, device="cuda", dtype=torch.float64)
    for parameter, previous in inspected.values():
        delta = parameter.detach() - previous
        rounding += torch.stack(
            (
                delta.new_tensor(delta.numel(), dtype=torch.float64),
                (delta != 0).sum().double(),
                (parameter.detach().bfloat16() != previous.bfloat16()).sum().double(),
                delta.double().square().sum(),
            )
        )
    dist.all_reduce(rounding)
    metrics["inspected_fp32_changed_fraction"] = float(rounding[1] / rounding[0])
    metrics["inspected_bf16_changed_fraction"] = float(rounding[2] / rounding[0])
    metrics["inspected_parameter_delta_rms"] = float((rounding[3] / rounding[0]).sqrt())
    if measure_distribution:
        moments = torch.zeros(3, device="cuda", dtype=torch.float64)
        for parameter, state in optimizer.state.items():
            moments[0] += parameter.numel()
            moments[1] += state["exp_avg"].double().square().sum()
            moments[2] += state["exp_avg_sq"].double().sum()
        dist.all_reduce(moments)
        metrics["adam_first_moment_rms"] = float((moments[1] / moments[0]).sqrt())
        metrics["adam_second_moment_mean"] = float(moments[2] / moments[0])
    return metrics


def train(root, output, cache_dir, config):
    """Run a full-size fixed or refreshed trajectory experiment."""
    rank = dist.get_rank()
    artifact = torch.load(cache_dir / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
    fixed_sequences = artifact["sequences"]
    diagnostic_sequences = [dict(sequence) for sequence in fixed_sequences]
    tokenizer, prompt, _ = one_prompt(root)
    precision = config["precision"]
    resume = (
        json.loads((output / "resume.json").read_text())
        if (output / "resume.json").exists()
        else None
    )
    base_path = Path(resume["checkpoint"]) if resume else model_path(root, "student")
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    )
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = "1"
    model = FullyShardedDataParallel(
        ResponseModel(hf, artifact["vocab"]),
        device_id=rank,
        auto_wrap_policy=ModuleWrapPolicy({Qwen2DecoderLayer}),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        use_orig_params=True,
        sync_module_states=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config["weight_decay"],
        foreach=False,
    )
    start = int(resume["update"]) if resume else 0
    if resume:
        optimizer.load_state_dict(
            torch.load(
                base_path / f"optimizer-rank-{rank}.pt", map_location="cuda", weights_only=False
            )
        )
        if config["anchor_interval"]:
            anchors = torch.load(
                base_path / f"anchor-rank-{rank}.pt", map_location="cpu", weights_only=False
            )
            for sequence, anchor in zip(fixed_sequences, anchors, strict=True):
                sequence.update(anchor)
    sequences = fixed_sequences
    teacher = (
        load_model(root, "teacher", torch.bfloat16)
        if config["rollout"] == "online" or config["anchor_interval"]
        else None
    )
    replica = load_model(root, "student", torch.bfloat16) if config["rollout"] == "online" else None
    checkpoint_steps = set(config["checkpoint_steps"])
    begin = time.monotonic()
    wandb_run = None
    if rank == 0 and config["wandb"]:
        import wandb

        wandb_run = wandb.init(
            entity="wncfht",
            project="mirror-opd-oneshot-absorption",
            id=hashlib.sha256(config["id"].encode()).hexdigest()[:8],
            name=config["id"],
            group="oneshot-effective-time-20261012",
            config=config,
            dir=str(output),
            resume="allow",
        )
        for scope in ("train", "fixed_fp32", "fixed_native", "native_final", "window_final"):
            wandb_run.define_metric(scope + "/optimizer_updates")
            wandb_run.define_metric(
                scope + "/*", step_metric=scope + "/optimizer_updates", step_sync=False
            )
        (output / "wandb-run.json").write_text(
            json.dumps({"id": wandb_run.id, "url": wandb_run.url}) + "\n"
        )
    if start == 0:
        record_fixed_readouts(model, diagnostic_sequences, config, output, 0, wandb_run)
    for update in range(start, config["updates"]):
        if config["rollout"] == "online":
            with FullyShardedDataParallel.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                state = {
                    key.removeprefix("hf."): value for key, value in model.state_dict().items()
                }
            replica.load_state_dict(state)
            del state
            sequences = generate_batch(
                replica,
                tokenizer,
                prompt,
                seed=config["seed"] + update * 1009,
                rank=rank,
                batch_size=config["batch_size"] // 4,
                response_length=config["response_length"],
            )
            select_positions(
                sequences,
                seed=config["seed"] + update * 1009,
                total_positions=config["diagnostic_positions"],
            )
            current_anchor(model, sequences, precision, config["topk"])
            score_teacher(teacher, sequences, vocab=artifact["vocab"])
        elif config["anchor_interval"] and update % config["anchor_interval"] == 0:
            # The states remain fixed; only the mirror anchor is reset. This
            # isolates anchor lifecycle from a changing rollout-state distribution.
            current_anchor(model, sequences, precision, config["topk"])
            score_teacher(teacher, sequences, vocab=artifact["vocab"])
        metrics = step_update(
            model,
            optimizer,
            sequences,
            config,
            measure_distribution=update % config["diagnostic_every"] == 0
            or config["method"] != "opd",
        )
        write_record(
            output / "updates.jsonl",
            {
                "update_before": update,
                "updates_completed": update + 1,
                "effective_time_before": config["lr"] / 1e-6 * update,
                "lr": config["lr"],
                "measurement": "native-pre-update",
                "seconds_elapsed": time.monotonic() - begin,
                "metrics": metrics,
            },
            wandb_run,
        )
        if config["anchor_interval"] and (update + 1) % config["anchor_interval"] == 0:
            write_record(
                output / "inner-window-final.jsonl",
                {
                    "update": update + 1,
                    "outer_window": update // config["anchor_interval"],
                    "inner_updates_completed": config["anchor_interval"],
                    "effective_time": config["lr"] / 1e-6 * (update + 1),
                    "measurement": "native-current-window-target-post-update-before-anchor-reset",
                    "metrics": evaluate_fixed(
                        model, sequences, precision, config["eta"], native=True
                    ),
                },
                wandb_run,
            )
        if (update + 1) % config["diagnostic_every"] == 0 or update + 1 == config["updates"]:
            record_fixed_readouts(
                model, diagnostic_sequences, config, output, update + 1, wandb_run
            )
        if update + 1 in checkpoint_steps:
            save_checkpoint(
                root, output, model, optimizer, tokenizer, sequences, update + 1, config
            )
        if rank == 0:
            (output / "progress.json").write_text(
                json.dumps(
                    {
                        "updates_completed": update + 1,
                        "updates_requested": config["updates"],
                        "elapsed_seconds": time.monotonic() - begin,
                        "state": "running",
                    }
                )
                + "\n"
            )
            print(
                json.dumps(
                    {
                        "update": update + 1,
                        "method": config["method"],
                        "precision": precision,
                        "grad_norm": metrics["grad_norm"],
                        "action_abs_gap": metrics["current_advantage_abs/mean"],
                    }
                ),
                flush=True,
            )
    write_record(
        output / "native-final.jsonl",
        {
            "update": config["updates"],
            "effective_time": config["lr"] / 1e-6 * config["updates"],
            "measurement": "native-last-batch-post-update",
            "metrics": measure_native(model, sequences, config),
        },
        wandb_run,
    )
    dist.barrier()
    if rank == 0:
        (output / "completion.json").write_text(
            json.dumps(
                {
                    "updates_completed": config["updates"],
                    "elapsed_seconds": time.monotonic() - begin,
                    "state": "completed",
                }
            )
            + "\n"
        )
        if wandb_run is not None:
            wandb_run.finish()
    del model, optimizer, teacher, replica
    gc.collect()


def main():
    """Initialize only GPUs 0--3 and dispatch one registered experiment job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--owners", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.empty(1, device="cuda")
    register_gpu_owner(args.owners, rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    if dist.get_world_size() != 4:
        raise RuntimeError("The registered experiment requires exactly four ranks")
    root = args.repo
    config = json.loads(args.config.read_text())
    torch.manual_seed(config["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        prepare_batch(root, args.output, config, rank)
    else:
        train(root, args.output, args.cache, config)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
