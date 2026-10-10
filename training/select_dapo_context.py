"""Choose a response budget from the full training-set paired baselines."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo / "runs/dapo-context-baselines"
    summaries = {
        label: json.loads((root / label / "dapo-math-17k-train-summary.json").read_text())["all"]
        for label in ("student", "teacher")
    }
    if any(s["count"] != 17917 for s in summaries.values()):
        raise ValueError("Context selection requires complete training-set baselines")
    sampled = {
        label: json.loads((root / label / "sampled/dapo-math-17k-train-summary.json").read_text())[
            "all"
        ]
        for label in ("student", "teacher")
    }
    if any(s["count"] != 1024 for s in sampled.values()):
        raise ValueError("Context selection requires complete T=1 length pilots")
    checks = [*summaries.values(), *sampled.values()]
    prompt = ((max(s["prompt_max"] for s in summaries.values()) + 127) // 128) * 128
    acceptable = [
        b
        for b in (512, 1024, 2048, 4096)
        if all(
            s["budget_overflow_rate"][str(b)] <= 0.01
            and s["accuracy"] - s["budget_accuracy"][str(b)] <= 0.005
            for s in checks
        )
        and prompt + b <= 8192
    ]
    selected = acceptable[0] if acceptable else None
    report = {
        "max_prompt_length": prompt,
        "max_response_length": selected,
        "max_model_len": prompt + selected if selected else None,
        "criterion": "both complete train sets: overflow <=1%, replay accuracy loss <=0.5 percentage points",
        "status": "ready_for_training_memory_canary" if selected else "needs_length_analysis",
        "next_stage": "Memory canary, then production training only after successful canary.",
        "baselines": summaries,
        "sampled_length_pilots": sampled,
    }
    (root / "context-selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
