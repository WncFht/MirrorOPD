"""Native one-shot trajectories, compact caches, and fixed diagnostic positions."""

import gc
import gzip
import hashlib
import json

import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from mirroropd.training.cache import build_mirror_anchor_cache, complete_mirror_teacher_cache


class RolloutVocabulary(LogitsProcessor):
    """Match the tokenizer-supported action vocabulary used by cache softmax."""

    def __init__(self, vocab):
        """Retain the contiguous valid vocabulary size."""
        self.vocab = vocab

    def __call__(self, input_ids, scores):  # noqa: ARG002 -- HF LogitsProcessor protocol
        """Exclude the model's padded output columns before sampling."""
        scores[:, self.vocab :] = float("-inf")
        return scores


def model_path(root, role):
    """Resolve the original model pair used by the historical math experiments."""
    name = "DeepSeek-R1-Distill-Qwen-1.5B" if role == "student" else "JustRL-DeepSeek-1.5B"
    return root.parent / "models" / name


def load_model(root, role, dtype):
    """Use SDPA for both precisions, retaining identical attention semantics."""
    return (
        AutoModelForCausalLM.from_pretrained(
            model_path(root, role),
            torch_dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )


def one_prompt(root):
    """Verify and return the original t22 prompt without truncation."""
    path = root.parent / "MD-OPD/data/math/DAPO-oneshot/t22.parquet"
    frame = pd.read_parquet(path)
    prompts = [json.dumps(list(value), sort_keys=True) for value in frame.prompt]
    if len(frame) != 64 or len(set(prompts)) != 1:
        raise ValueError("t22 must contain 64 copies of one prompt")
    tokenizer = AutoTokenizer.from_pretrained(model_path(root, "student"), local_files_only=True)
    ids = tokenizer.apply_chat_template(
        list(frame.prompt.iloc[0]), tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    return tokenizer, ids, hashlib.sha256(path.read_bytes()).hexdigest()


def generate_batch(model, tokenizer, prompt, *, seed, rank, batch_size, response_length):
    """Sample true EOS-terminated responses with unchanged native prompt tokens."""
    torch.manual_seed(seed + 100003 * rank)
    inputs = prompt.cuda().expand(batch_size, -1)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=response_length,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[RolloutVocabulary(len(tokenizer))],
        )
    start = prompt.shape[-1] - 1
    sequences = []
    for row in output:
        response = row[prompt.shape[-1] :]
        endings = (response == tokenizer.eos_token_id).nonzero().flatten()
        length = int(endings[0]) + 1 if endings.numel() else response.numel()
        sequences.append(
            {
                "ids": row[: prompt.shape[-1] + length].cpu(),
                "start": start,
                "response_tokens": length,
                "budget_exhausted": not bool(endings.numel()) and length == response_length,
            }
        )
    return sequences


def select_positions(sequences, *, seed, total_positions):
    """Draw a uniform global sample of response prediction positions."""
    local_lengths = [sequence["response_tokens"] for sequence in sequences]
    lengths = [None] * dist.get_world_size()
    dist.all_gather_object(lengths, local_lengths)
    flat_lengths = [length for rank_lengths in lengths for length in rank_lengths]
    count = sum(flat_lengths)
    generator = torch.Generator().manual_seed(seed + 99991)
    selected = (
        torch.randperm(count, generator=generator)[: min(total_positions, count)].sort().values
    )
    offset = sum(sum(value) for value in lengths[: dist.get_rank()])
    for sequence in sequences:
        length = sequence["response_tokens"]
        sequence["selected"] = (
            selected[(selected >= offset) & (selected < offset + length)] - offset
        )
        offset += length
    return count, min(total_positions, count)


def prediction_hidden(model, sequence):
    """Score response actions at predecessor positions, including terminal actions."""
    ids = sequence["ids"].cuda().unsqueeze(0)
    with torch.no_grad():
        return model.model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).last_hidden_state[0, sequence["start"] : -1]


def score_anchor(model, sequences, *, precision, vocab, topk):
    """Store compact anchor probabilities and full probabilities only at the reservoir."""
    for sequence in sequences:
        hidden = prediction_hidden(model, sequence)
        actions = sequence["ids"][sequence["start"] + 1 :].cuda()
        parts = []
        selected = []
        for start in range(0, len(actions), 128):
            stop = min(start + 128, len(actions))
            with torch.no_grad():
                logits = model.lm_head(hidden[start:stop])[..., :vocab]
                cache = build_mirror_anchor_cache(
                    logits,
                    actions[start:stop],
                    torch.ones_like(actions[start:stop], dtype=torch.bool),
                    topk,
                )
                parts.append({key: value.cpu() for key, value in cache.items()})
                positions = sequence["selected"]
                positions = positions[(positions >= start) & (positions < stop)] - start
                selected.append(logits.float().log_softmax(-1)[positions].cpu())
        sequence["anchor_" + precision] = {
            key: torch.cat([part[key] for part in parts]) for key in parts[0]
        }
        sequence["q_logp_" + precision] = torch.cat(selected)


def score_teacher(model, sequences, *, vocab):
    """Freeze one BF16 teacher distribution across student-precision conditions."""
    for sequence in sequences:
        hidden = prediction_hidden(model, sequence)
        selected = []
        completed = {
            precision: [] for precision in ("bf16", "fp32") if "anchor_" + precision in sequence
        }
        for start in range(0, len(hidden), 128):
            stop = min(start + 128, len(hidden))
            with torch.no_grad():
                logits = model.lm_head(hidden[start:stop])[..., :vocab]
                for precision, parts in completed.items():
                    anchor = {
                        key: value[start:stop].cuda()
                        for key, value in sequence["anchor_" + precision].items()
                    }
                    cache = complete_mirror_teacher_cache(logits, anchor)
                    parts.append({key: value.cpu() for key, value in cache.items()})
                positions = sequence["selected"]
                positions = positions[(positions >= start) & (positions < stop)] - start
                selected.append(logits.float().log_softmax(-1)[positions].cpu())
        for precision, parts in completed.items():
            sequence["cache_" + precision] = {
                key: torch.cat([part[key] for part in parts]) for key in parts[0]
            }
        sequence["teacher_logp"] = torch.cat(selected)
        for precision in completed:
            del sequence["anchor_" + precision]


def prepare_batch(root, output, config, rank):
    """Prepare matched 64-trajectory inputs before any training arm starts."""
    tokenizer, prompt, data_hash = one_prompt(root)
    initial = load_model(root, "student", torch.bfloat16)
    raw = output / "trajectories.jsonl"
    archive = output / "trajectories.jsonl.gz"
    if raw.exists() or archive.exists():
        with raw.open() if raw.exists() else gzip.open(archive, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        sequences = [
            {
                "ids": torch.tensor(row["ids"]),
                "start": prompt.shape[-1] - 1,
                "response_tokens": row["response_tokens"],
                "budget_exhausted": row["budget_exhausted"],
                "selected": torch.tensor(row["selected_positions"], dtype=torch.long),
            }
            for row in rows
            if row["rank"] == rank
        ]
        assert len(sequences) == config["batch_size"] // 4
        total_tokens = sum(row["response_tokens"] for row in rows)
        diagnostic_tokens = sum(len(row["selected_positions"]) for row in rows)
    else:
        sequences = generate_batch(
            initial,
            tokenizer,
            prompt,
            seed=config["seed"],
            rank=rank,
            batch_size=config["batch_size"] // 4,
            response_length=config["response_length"],
        )
        total_tokens, diagnostic_tokens = select_positions(
            sequences, seed=config["seed"], total_positions=config["diagnostic_positions"]
        )
    score_anchor(initial, sequences, precision="bf16", vocab=len(tokenizer), topk=config["topk"])
    del initial
    gc.collect()
    torch.cuda.empty_cache()
    initial = load_model(root, "student", torch.float32)
    score_anchor(initial, sequences, precision="fp32", vocab=len(tokenizer), topk=config["topk"])
    del initial
    gc.collect()
    torch.cuda.empty_cache()
    teacher = load_model(root, "teacher", torch.bfloat16)
    score_teacher(teacher, sequences, vocab=len(tokenizer))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequences": sequences,
            "config": config,
            "prompt_ids": prompt,
            "vocab": len(tokenizer),
            "total_tokens": total_tokens,
            "diagnostic_tokens": diagnostic_tokens,
            "data_sha256": data_hash,
        },
        output / f"rank-{rank}.pt",
    )
    local_rows = [
        {
            "rank": rank,
            "sample_index": rank * len(sequences) + index,
            "response_tokens": sequence["response_tokens"],
            "budget_exhausted": sequence["budget_exhausted"],
            "selected_positions": sequence["selected"].tolist(),
            "ids": sequence["ids"].tolist(),
        }
        for index, sequence in enumerate(sequences)
    ]
    rows = [None] * 4
    dist.all_gather_object(rows, local_rows)
    if rank == 0:
        (output / "trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for part in rows for row in part)
        )
        manifest = {
            "config": config,
            "data_sha256": data_hash,
            "total_tokens": total_tokens,
            "diagnostic_tokens": diagnostic_tokens,
            "budget_exhausted_count": sum(row["budget_exhausted"] for part in rows for row in part),
            "rank_sha256": {
                str(index): hashlib.sha256((output / f"rank-{index}.pt").read_bytes()).hexdigest()
                for index in range(4)
            },
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
