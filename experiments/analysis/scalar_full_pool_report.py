"""Both scalar candidates versus each locked baseline in the three-seed full-pool study."""

import argparse
import json
from pathlib import Path

from experiments.analysis.scalar_large_report import summarize


def report(root, output, selection):
    """Use pilot-selected constants; never choose a baseline from confirmation results."""
    for index, baseline in enumerate(selection["baselines"], start=1):
        for method in ("search", "feedback"):
            candidate = {
                "method": method,
                "search_every": 4 if method == "search" else 1,
                "target_rate": 0.01,
            }
            summarize(
                root,
                output / f"{method}-vs-constant{index}",
                {"baseline_configs": [baseline], "primary_scalar_config": candidate},
                expected_seeds=(42, 142, 242),
            )
    (output / "locked-comparisons.json").write_text(
        json.dumps(
            {
                "baselines": selection["baselines"],
                "seeds": [42, 142, 242],
                "primary_baseline": 1,
                "candidates": ["search every 4", "feedback target 0.01"],
                "criterion": "all paired seeds win; median geometric error gain >= 0.1; validation and final error gates",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report(args.root, args.output, json.loads(args.selection.read_text()))


if __name__ == "__main__":
    main()
