"""Rebuild frozen full-vocabulary caches from the versioned token IDs."""

import argparse
import gzip
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "experiments/toy"))
    from scalar_schedule_lm import load_model, prediction_logits

    with gzip.open(args.prefixes, "rt") as stream:
        record = json.load(stream)
    ids = torch.tensor(record["ids"], dtype=torch.long).cuda()
    mask = torch.tensor(record["attention_mask"], dtype=torch.long).cuda()
    artifact = dict(record["metadata"])
    artifact["student"] = str(root.parent / "models/DeepSeek-R1-Distill-Qwen-1.5B")
    artifact["teacher"] = str(root.parent / "models/JustRL-DeepSeek-1.5B")
    for role, field in (("student", "q_logits"), ("teacher", "t_logits")):
        model = load_model(artifact[role])
        with torch.no_grad():
            artifact[field] = prediction_logits(
                model, ids, artifact["start"], mixed_precision=False, attention_mask=mask
            ).cpu()
        del model
        torch.cuda.empty_cache()
    artifact["ids"] = ids.cpu()
    artifact["attention_mask"] = mask.cpu()
    artifact["actions"] = ids[:, artifact["prompt_length"] :].flatten().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(
        json.dumps(
            {"cache": str(args.output), "prompts": len(ids), "prefixes": len(artifact["actions"])}
        )
    )


if __name__ == "__main__":
    main()
