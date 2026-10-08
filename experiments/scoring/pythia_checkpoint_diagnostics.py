"""Score fixed teacher-forced prefixes; never generate or update model parameters."""

import argparse
import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch


def materialize_state_dict(state):
    """Materialize a world-size-one FSDP model state without silently dropping keys."""
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint model state must be a mapping")
    result = {}
    for key, checkpoint_value in state.items():
        value = checkpoint_value
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        elif hasattr(value, "local_shards"):
            shards = value.local_shards()
            shape = tuple(value.size())
            if not shards:
                raise ValueError(f"No local shard for {key}; only world-size-one is supported")
            full = torch.empty(shape, dtype=shards[0].tensor.dtype)
            covered = 0
            for shard in shards:
                offsets = shard.metadata.shard_offsets
                sizes = shard.metadata.shard_sizes
                slices = tuple(
                    slice(start, start + size) for start, size in zip(offsets, sizes, strict=True)
                )
                full[slices] = shard.tensor.cpu()
                covered += shard.tensor.numel()
            if covered != math.prod(shape):
                raise ValueError(f"Incomplete local shards for {key}: {covered}/{math.prod(shape)}")
            value = full
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor checkpoint field: {key}")
        result[key] = value.detach().cpu()
    return result


def load_actor_checkpoint(model, actor_dir, scratch_dir):
    actor_dir = Path(actor_dir)
    checkpoint = actor_dir / "model_world_size_1_rank_0.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Expected single-rank full-parameter FSDP checkpoint: {checkpoint}"
        )
    initialized_here = not torch.distributed.is_initialized()
    # ShardedTensor/DTensor deserialization requires a process group even for one rank.
    with tempfile.TemporaryDirectory(prefix="checkpoint-reader-", dir=scratch_dir) as scratch:
        if initialized_here:
            torch.distributed.init_process_group(
                "gloo", init_method=f"file://{scratch}/process-group", rank=0, world_size=1
            )
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(materialize_state_dict(state), strict=True)
        finally:
            if initialized_here:
                torch.distributed.destroy_process_group()
    return checkpoint


def distribution_metrics(student_logits, teacher_logits):
    """Exact full-vocabulary KL in both directions and entropy, all measured in nats."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher vocab/logit dimensions must match")
    student_logp = student_logits.double().log_softmax(dim=-1)
    teacher_logp = teacher_logits.double().log_softmax(dim=-1)
    student_p = student_logp.exp()
    teacher_p = teacher_logp.exp()
    return {
        "student_teacher_kl": (student_p * (student_logp - teacher_logp)).sum(-1),
        "teacher_student_kl": (teacher_p * (teacher_logp - student_logp)).sum(-1),
        "student_entropy": -(student_p * student_logp).sum(-1),
        "teacher_entropy": -(teacher_p * teacher_logp).sum(-1),
    }


def selected_positions(prompt_ids, continuation_ids, *, all_prompt_prefixes, max_prompt_prefixes):
    if not prompt_ids:
        raise ValueError("Each record requires a nonempty prompt_ids list")
    prompt_positions = (
        list(range(max(0, len(prompt_ids) - max_prompt_prefixes), len(prompt_ids)))
        if all_prompt_prefixes
        else [len(prompt_ids) - 1]
    )
    continuation_positions = list(
        range(len(prompt_ids) - 1, len(prompt_ids) + len(continuation_ids) - 1)
    )
    return sorted(set(prompt_positions + continuation_positions))


def read_records(path):
    data = json.loads(Path(path).read_text())
    records = data.get("records", data.get("examples")) if isinstance(data, dict) else data
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset JSON must be a nonempty list or contain records/examples")
    for index, record in enumerate(records):
        for field in ("prompt_ids", "continuation_ids"):
            if not isinstance(record.get(field), list) or any(
                not isinstance(token, int) or token < 0 for token in record[field]
            ):
                raise ValueError(f"Record {index} requires nonnegative integer {field}")
    return records


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows):
    fields = (
        "student_teacher_kl",
        "teacher_student_kl",
        "student_entropy",
        "teacher_entropy",
        "student_continuation_nll",
        "teacher_continuation_nll",
    )
    result = {}
    for field in fields:
        values = [row[field] for row in rows if field in row]
        if values:
            tensor = torch.tensor(values, dtype=torch.float64)
            result[field] = {
                "count": len(values),
                "mean": tensor.mean().item(),
                "std": tensor.std(correction=0).item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
            }
    return result


def score_records(student, teacher, records, args):
    rows = []
    device = next(student.parameters()).device
    with torch.inference_mode():
        for example_index, record in enumerate(records):
            prompt = record["prompt_ids"]
            continuation = record["continuation_ids"][: args.max_continuation_tokens]
            positions = selected_positions(
                prompt,
                continuation,
                all_prompt_prefixes=args.prompt_prefixes == "all",
                max_prompt_prefixes=args.max_prompt_prefixes,
            )
            ids = torch.tensor([prompt + continuation], device=device)
            # One bounded trajectory at a time; full logits are discarded before the next record.
            student_logits = student(input_ids=ids, use_cache=False).logits[0]
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits[0]
            for start in range(0, len(positions), args.prefix_chunk_size):
                selected = positions[start : start + args.prefix_chunk_size]
                student_selected = student_logits[selected]
                teacher_selected = teacher_logits[selected]
                values = distribution_metrics(student_selected, teacher_selected)
                student_logp = student_selected.double().log_softmax(-1)
                teacher_logp = teacher_selected.double().log_softmax(-1)
                for local_index, position in enumerate(selected):
                    row = {
                        "example_index": example_index,
                        "example_id": record.get("id", record.get("example_id", example_index)),
                        "prefix_length": position + 1,
                        "continuation_index": position - len(prompt) + 1,
                        **{key: value[local_index].item() for key, value in values.items()},
                    }
                    token_index = position - len(prompt) + 1
                    if 0 <= token_index < len(continuation):
                        target = continuation[token_index]
                        row["target_token"] = target
                        row["student_continuation_nll"] = -student_logp[local_index, target].item()
                        row["teacher_continuation_nll"] = -teacher_logp[local_index, target].item()
                    rows.append(row)
            del student_logits, teacher_logits
            print(
                f"scored example {example_index + 1}/{len(records)}: {len(positions)} prefixes",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="actor directory; omit for initial baseline"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--prompt-prefixes", choices=("last", "all"), default="last")
    parser.add_argument("--max-prompt-prefixes", type=int, default=64)
    parser.add_argument("--max-continuation-tokens", type=int, default=64)
    parser.add_argument("--prefix-chunk-size", type=int, default=8)
    args = parser.parse_args()
    if min(args.max_prompt_prefixes, args.max_continuation_tokens, args.prefix_chunk_size) < 1:
        parser.error("Prefix/continuation limits and chunk size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_records(args.dataset)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Exact token-level KL requires identical student/teacher token mapping")
    dtype = getattr(torch, args.dtype)
    student = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=dtype)
    checkpoint = None
    if args.checkpoint:
        checkpoint = load_actor_checkpoint(student, args.checkpoint, args.output)
    student.to(args.device).eval()
    teacher = (
        AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype)
        .to(args.device)
        .eval()
    )
    rows = score_records(student, teacher, records, args)
    with (args.output / "prefixes.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "measurement": "fixed-prefix teacher-forced full-vocabulary distributions",
        "units": "nats",
        "aggregation": "equal-prefix population moments; NLL equal-token",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "examples": len(records),
        "prefixes": len(rows),
        "dtype": args.dtype,
        "prompt_prefixes": args.prompt_prefixes,
        "max_prompt_prefixes": args.max_prompt_prefixes,
        "max_continuation_tokens": args.max_continuation_tokens,
        "torch_version": torch.__version__,
        "statistics": summarize(rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
