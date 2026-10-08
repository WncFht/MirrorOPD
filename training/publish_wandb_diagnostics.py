"""Publish completed inner JSONL and available independent evaluations to W&B."""

import argparse
import json
from pathlib import Path

from mirroropd.training.tracking import configure_wandb, grouped_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--include-outer", action="store_true", help="Regroup a legacy run")
    args = parser.parse_args()
    reference = json.loads((args.run / "wandb-run.json").read_text())
    import wandb

    run = wandb.init(
        **{k: reference[k] for k in ("entity", "project", "id")}, resume="must", mode="online"
    )
    configure_wandb(run, {})
    # W&B has one internal history step per run.  The displayed axes are
    # separate (outer_step vs inner_update), but their writes still need a
    # monotonically increasing internal step or later inner rows overwrite
    # earlier outer rows in the web UI.
    # Resumed native trainer runs already have history steps (usually 1--5),
    # while older publisher versions may have written a different sequence.
    # A large monotone offset makes this repair append-only for every run.
    internal_step = max(int(getattr(run, "step", 0) or 0), 1_000_000)
    if args.include_outer:
        for line in (args.run / "outer.jsonl").read_text().splitlines():
            record = json.loads(line)
            internal_step += 1
            run.log(
                {"00_axes/outer_step": record["step"], **grouped_metrics(record["data"], "outer")},
                step=internal_step,
            )
    records = [json.loads(line) for line in (args.run / "inner.jsonl").read_text().splitlines()]
    for index, record in enumerate(records):
        internal_step += 1
        run.log(
            {
                "00_axes/inner_update": index,
                "00_axes/inner_outer": record["outer_call"],
                "00_axes/inner_epoch_index": record["inner_step"],
                **grouped_metrics(record["metrics"], "inner"),
            },
            step=internal_step,
        )
    evaluations = sorted(
        (args.run / "evaluation").glob("step-*/summary.json"),
        key=lambda p: int(p.parent.name.removeprefix("step-")),
    )
    for path in evaluations:
        result = json.loads(path.read_text())
        internal_step += 1
        run.log(
            {
                "00_axes/evaluation_outer": int(path.parent.name.removeprefix("step-")),
                **grouped_metrics(
                    {f"evaluation/{k}": v["mean"] for k, v in result["statistics"].items()},
                    "evaluation",
                ),
            },
            step=internal_step,
        )
    run.summary["diagnostics/inner_records"] = len(records)
    run.summary["diagnostics/independent_evaluations"] = len(evaluations)
    run.finish()


if __name__ == "__main__":
    main()
