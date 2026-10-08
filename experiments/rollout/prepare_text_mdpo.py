"""Freeze natural-text splits, token prefixes and shared model vocabulary."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer

IDENTITY_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
MODEL_REVISIONS = {
    "EleutherAI/pythia-14m": "cf967c0a9a04383db6f7b1108d86b2962634b4ac",
    "EleutherAI/pythia-70m": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
}
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def prepare(output, train_count=512, validation_count=64, seed=20261008):
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    paths = {}
    revisions = {}
    tokenizers = []
    for role, name in (("student", "EleutherAI/pythia-14m"), ("teacher", "EleutherAI/pythia-70m")):
        revision = MODEL_REVISIONS[name]
        assert api.model_info(name, revision=revision).sha == revision
        path = snapshot_download(
            name, revision=revision, allow_patterns=["*.json", "*.txt", "*.bin"]
        )
        paths[role] = path
        revisions[role] = {"name": name, "revision": revision}
        tokenizers.append(AutoTokenizer.from_pretrained(path))
    tokenizer, teacher_tokenizer = tokenizers
    assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
    assert tokenizer.eos_token_id == teacher_tokenizer.eos_token_id
    for text in ("The quick brown fox jumps over the lazy dog.", "\n = Mathematics = \n", "x² = 4"):
        assert tokenizer.encode(text) == teacher_tokenizer.encode(text)
    tokenizer.chat_template = IDENTITY_TEMPLATE
    vocab_sizes = [AutoConfig.from_pretrained(paths[r]).vocab_size for r in ("student", "teacher")]
    assert vocab_sizes[0] == vocab_sizes[1]
    dataset_revision = DATASET_REVISION
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=dataset_revision)
    manifest = {
        "dataset": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "dataset_revision": dataset_revision,
        "seed": seed,
        "prefix_length": 64,
        "continuation_length": 64,
        "selection": "shuffle eligible source rows independently per split; first requested rows",
        "eligibility": "nonempty, nonheading row; first 64 token IDs decode and re-encode identically; >=128 tokens",
        "models": revisions,
        "model_paths": paths,
        "vocab_size": vocab_sizes[0],
        "tokenizer_length": len(tokenizer),
        "shared_tokenizer": "student tokenizer for actor, teacher and rollout; equal vocabulary maps and test encodings",
        "template": IDENTITY_TEMPLATE,
        "splits": {},
    }
    for split, count in (("train", train_count), ("validation", validation_count)):
        candidates = []
        for index, item in enumerate(dataset[split]):
            text = item["text"]
            if not text.strip() or text.lstrip().startswith("="):
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 128:
                continue
            prompt_ids = ids[:64]
            prefix = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=False)
            if tokenizer.encode(prefix, add_special_tokens=False) != prompt_ids:
                continue
            assert (
                tokenizer.apply_chat_template([{"role": "user", "content": prefix}]) == prompt_ids
            )
            candidates.append(
                {
                    "source_row": index,
                    "split": split,
                    "prompt": prefix,
                    "prompt_ids": prompt_ids,
                    "continuation_ids": ids[64:128],
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
        random.Random(seed + int(split == "validation")).shuffle(candidates)
        assert len(candidates) >= count, (split, len(candidates), count)
        records = candidates[:count]
        target = output / f"{split}-prefixes.json"
        target.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n")
        rows = [
            {
                "data_source": "wikitext-mdpo",
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "ability": "text",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"index": i, "source_row": r["source_row"], "split": split},
            }
            for i, r in enumerate(records)
        ]
        parquet = output / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(parquet, index=False)
        manifest["splits"][split] = {
            "count": count,
            "source_rows": [r["source_row"] for r in records],
            "json_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["models"], indent=2))
    print("Prepared immutable train/validation prefixes:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    prepare(args.output, args.train_count, args.validation_count, args.seed)
