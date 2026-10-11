"""Measure padded-output probability mass at fixed native DAPO prefixes."""

import gc
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    indices = [0, 21, 100, 1000]
    rows = pd.read_parquet(workspace / "MD-OPD/data/math/DAPO/train.parquet").iloc[indices]
    report = {
        "device": "physical GPU 0",
        "dtype": "BF16 forward",
        "input": "DAPO native prompt final position",
        "indices": indices,
        "models": {},
    }
    for name in ["DeepSeek-R1-Distill-Qwen-1.5B", "JustRL-DeepSeek-1.5B"]:
        path = workspace / "models" / name
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = (
            AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        masses = []
        for row in rows.itertuples():
            ids = tokenizer.apply_chat_template(
                list(row.prompt), tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                logits = model(ids).logits[0, -1].double()
                probability = logits.softmax(-1)
                masses.append(float(probability[len(tokenizer) :].sum()))
        report["models"][name] = {
            "output_vocab": model.config.vocab_size,
            "action_vocab": len(tokenizer),
            "padded_mass": masses,
            "hf_minus_rollout_logprob": [
                float(torch.log1p(torch.tensor(-value, dtype=torch.float64))) for value in masses
            ],
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()
    output = root / "results/training/math-4gpu/support-mass-probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
