"""Fail before GPU allocation on incompatible templates or context budgets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from mirroropd.training.tokenization import assert_shared_tokenization


def audit(config):
    model = config.actor_rollout_ref.model
    student = AutoTokenizer.from_pretrained(model.path)
    teacher_path = config.distillation.mirror_teacher_model_path
    teacher = AutoTokenizer.from_pretrained(teacher_path)
    report = assert_shared_tokenization(student, teacher, require_chat=True)
    if model.custom_chat_template is not None:
        raise ValueError("GSM8K native-template setting requires custom_chat_template=null")
    data = config.data
    context = config.actor_rollout_ref.rollout.max_model_len
    if context < data.max_prompt_length + data.max_response_length:
        raise ValueError("max_model_len must cover prompt budget + full response budget")
    report["model_configs"] = {}
    for name in (model.path, teacher_path):
        hf = AutoConfig.from_pretrained(name)
        if hf.max_position_embeddings < context:
            raise ValueError(f"{name}: context exceeds max_position_embeddings")
        report["model_configs"][name] = {
            "revision": hf._commit_hash,  # noqa: SLF001
            "max_position_embeddings": hf.max_position_embeddings,
        }
    report["native_template_sha256"] = hashlib.sha256(student.chat_template.encode()).hexdigest()
    if config.actor_rollout_ref.actor.ppo_mini_batch_size != data.train_batch_size:
        raise ValueError("MDPO full-batch setting requires mini batch = prompt batch")
    actor = config.actor_rollout_ref.actor
    if actor.optim.clip_grad != actor.grad_clip:
        raise ValueError(
            "FSDP engine uses optim.clip_grad; it must match the declared actor.grad_clip"
        )
    if config.actor_rollout_ref.rollout.n != 1:
        raise ValueError("MDPO setting expects one response per prompt")
    report.update(student=model.path, teacher=teacher_path, context=context, splits={})
    for split, files in (("train", data.train_files), ("test", data.val_files)):
        paths = [files] if isinstance(files, str) else list(files)
        lengths, count = [], 0
        hashes = {}
        for file in paths:
            frame = pd.read_parquet(file)
            hashes[str(file)] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            for row in frame.itertuples():
                messages = list(row.prompt)
                if len(messages) != 1 or messages[0]["role"] != "user":
                    raise ValueError("Expected a single user question, without reference solution")
                if row.extra_info["split"] != split:
                    raise ValueError(f"Incorrect split in {file}")
                ids = student.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                text = student.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Native apply_chat_template(tokenize=True) must not duplicate BOS.
                if ids != student.encode(text, add_special_tokens=False):
                    raise ValueError("Chat tokenization introduces extra special tokens")
                lengths.append(len(ids))
            count += len(frame)
        if max(lengths) > data.max_prompt_length:
            raise ValueError(f"{split}: overlong prompt {max(lengths)} > {data.max_prompt_length}")
        report["splits"][split] = {
            "available_examples": count,
            "configured_max_samples": data.get(
                f"{split if split == 'train' else 'val'}_max_samples", -1
            ),
            "prompt_tokens_mean": float(np.mean(lengths)),
            "prompt_tokens_max": max(lengths),
            "sha256": hashes,
        }
    first = pd.read_parquet(paths[0]).iloc[0]
    report["serialized_test_prompt"] = student.apply_chat_template(
        list(first.prompt), tokenize=False, add_generation_prompt=True
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(OmegaConf.load(args.config))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
