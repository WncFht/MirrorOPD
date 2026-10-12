"""Frozen, unpadded multi-prompt pools for scalar BPO confirmation."""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from training.scalar_large_gpu import register_owner

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_student_helper():
    sys.path.insert(0, str(ROOT / "experiments/toy"))
    from scalar_schedule_lm import load_model

    return load_model


def response_logits(model, sequence, positions=None, *, precision="fp32"):
    ids = sequence["ids"].cuda().unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        hidden = model.model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).last_hidden_state[0, sequence["start"] : -1]
        if positions is not None:
            hidden = hidden[positions.cuda()]
        return model.lm_head(hidden).float()


def select_prompts(tokenizer, seed, count):
    frame = pd.read_parquet(ROOT.parent / "MD-OPD/data/math/DAPO/train.parquet")
    # Calibration and confirmation draw from disjoint row ranges.
    indices = list(range(min(512, len(frame)))) if seed == 1042 else list(range(512, len(frame)))
    random.Random(seed).shuffle(indices)
    prompts = []
    for index in indices:
        text = tokenizer.apply_chat_template(
            list(frame.prompt.iloc[index]), tokenize=False, add_generation_prompt=True
        )
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if 1 <= len(token_ids) <= 512:
            prompts.append({"source_row": index, "text": text, "prompt_ids": token_ids})
        if len(prompts) == count:
            return prompts
    raise ValueError("not enough eligible complete prompts")


def generate_records(model, tokenizer, prompts, seed, response_length):
    generator_config = copy.deepcopy(model.generation_config)
    eos = generator_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    records = []
    torch.manual_seed(seed)
    for start in range(0, len(prompts), 8):
        selected = prompts[start : start + 8]
        inputs = tokenizer(
            [row["text"] for row in selected],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=1,
                top_p=1,
                top_k=0,
                max_new_tokens=response_length,
                generation_config=generator_config,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row, output in zip(selected, ids.cpu(), strict=True):
            response = output[inputs.input_ids.shape[1] :].tolist()
            terminal = next((i for i, token in enumerate(response) if token in eos_ids), None)
            if terminal is not None:
                response = response[: terminal + 1]
            prefix = row["prompt_ids"]
            records.append(
                {
                    "ids": torch.tensor(prefix + response, dtype=torch.long),
                    "start": len(prefix) - 1,
                    "response_tokens": len(response),
                    "truncated": terminal is None and len(response) == response_length,
                    "source_row": row["source_row"],
                    "text": row["text"],
                }
            )
        print(json.dumps({"generated": len(records), "seed": seed}), flush=True)
    return records


def prepare(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.device) or args.device not in range(4):
        raise RuntimeError("only physical GPUs 0--3 are authorized")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        print(json.dumps({"already_prepared": str(manifest_path)}), flush=True)
        return
    load_model = load_student_helper()
    student_path = ROOT.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B"
    teacher_path = ROOT.parent / "models/JustRL-DeepSeek-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(
        student_path, local_files_only=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    records_path = args.output / "tokens.pt"
    student = load_model(student_path)
    register_owner(args.device)
    if records_path.exists():
        records = torch.load(records_path, weights_only=False)
    else:
        prompts = select_prompts(
            tokenizer, args.seed, args.train_sequences + args.validation_sequences
        )
        records = generate_records(student, tokenizer, prompts, args.seed, args.response_length)
        torch.save(records, records_path)
    for index, sequence in enumerate(records):
        target = args.output / f"sequence-{index:04d}.pt"
        if target.exists():
            continue
        with torch.no_grad():
            qlog = response_logits(student, sequence).log_softmax(-1).cpu()
        torch.save({"qlog": qlog}, target)
    del student
    torch.cuda.empty_cache()
    teacher = load_model(teacher_path)
    entries = []
    for index, sequence in enumerate(records):
        target = args.output / f"sequence-{index:04d}.pt"
        cached = torch.load(target, weights_only=True, mmap=True)
        if "tlog" not in cached:
            with torch.no_grad():
                tlog = response_logits(teacher, sequence).log_softmax(-1)
                qlog = cached["qlog"].cuda()
                actions = sequence["ids"][sequence["start"] + 1 :].cuda()
                caches = {
                    str(k): {
                        key: value.cpu()
                        for key, value in build_mirror_opd_cache_from_full_logits(
                            qlog, tlog, actions, torch.ones_like(actions, dtype=torch.bool), k
                        ).items()
                    }
                    for k in (16, 64)
                }
                teacher_full_center = (qlog.exp() * (qlog - tlog)).sum(-1).cpu()
            complete = dict(
                sequence,
                qlog=cached["qlog"].clone(),
                tlog=tlog.cpu(),
                caches=caches,
                teacher_full_center=teacher_full_center,
            )
            temporary = target.with_suffix(".tmp")
            torch.save(complete, temporary)
            temporary.replace(target)
            del qlog, tlog, complete
        entries.append(
            {
                "file": target.name,
                "response_tokens": sequence["response_tokens"],
                "source_row": sequence["source_row"],
                "truncated": sequence["truncated"],
            }
        )
        if index % 8 == 0:
            print(json.dumps({"scored": index + 1, "seed": args.seed}), flush=True)
    manifest = {
        "seed": args.seed,
        "train_sequences": args.train_sequences,
        "validation_sequences": args.validation_sequences,
        "response_length": args.response_length,
        "temperature": 1,
        "top_p": 1,
        "top_k": 0,
        "prompt_truncation": False,
        "source_sha256": SOURCE_SHA256,
        "trajectory_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "student": str(student_path),
        "teacher": str(teacher_path),
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"prepared": str(manifest_path), "tokens": sum(e["response_tokens"] for e in entries)}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--train-sequences", type=int, default=128)
    parser.add_argument("--validation-sequences", type=int, default=32)
    parser.add_argument("--response-length", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
