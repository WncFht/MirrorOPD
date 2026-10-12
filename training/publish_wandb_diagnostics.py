"""Publish completed inner JSONL and available independent evaluations to W&B."""

import argparse
import json
from pathlib import Path

from mirroropd.training.tracking import configure_wandb, grouped_metrics, metric_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--include-outer", action="store_true", help="Regroup a legacy run")
    args = parser.parse_args()
    reference = json.loads((args.run / "wandb-run.json").read_text())
    import wandb

    history = list(
        wandb.Api()
        .run(f"{reference['entity']}/{reference['project']}/{reference['id']}")
        .scan_history()
    )
    seen_inner = {row["00_axes/inner_update"] for row in history if "00_axes/inner_update" in row}
    seen_outer = {row["00_axes/outer_step"] for row in history if "00_axes/outer_step" in row}
    seen_independent = {
        row["00_axes/evaluation_outer"]
        for row in history
        if "00_axes/evaluation_outer" in row
        and any(
            key.startswith("01_quality/")
            and "/val-" not in key
            and key != "01_quality/generated_examples"
            for key in row
        )
    }
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
    # Continue from the actual server history, including resumed sessions.
    # An arbitrary offset pollutes the default Step axis and hides progress.
    internal_step = max(
        int(getattr(run, "step", 0) or 0),
        max((int(row.get("_step", -1)) + 1 for row in history), default=0),
    )
    if args.include_outer:
        for line in (args.run / "outer.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["step"] in seen_outer:
                continue
            internal_step += 1
            payload = {
                "00_axes/outer_step": record["step"],
                **{
                    metric_name(
                        key,
                        "aggregate"
                        if key.startswith("actor/mirror/") or key == "actor/grad_norm"
                        else "outer",
                    ): value
                    for key, value in record["data"].items()
                },
            }
            if any(
                key.startswith(("val-core/", "val-aux/", "evaluation/")) for key in record["data"]
            ):
                payload["00_axes/evaluation_outer"] = record["step"]
            run.log(
                payload,
                step=internal_step,
            )
    records = [json.loads(line) for line in (args.run / "inner.jsonl").read_text().splitlines()]
    for index, record in enumerate(records):
        if index in seen_inner:
            continue
        internal_step += 1
        run.log(
            {
                "00_axes/inner_update": index,
                "00_axes/inner_outer": record["outer_call"],
                "00_axes/inner_step_index": record["inner_step"],
                **grouped_metrics(record["metrics"], "inner"),
            },
            step=internal_step,
        )
    evaluations = sorted(
        (args.run / "evaluation").glob("step-*/summary.json"),
        key=lambda p: int(p.parent.name.removeprefix("step-")),
    )
    for path in evaluations:
        if int(path.parent.name.removeprefix("step-")) in seen_independent:
            continue
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
    exit_file = args.run / "exit-code.txt"
    exit_code = int(exit_file.read_text()) if exit_file.exists() else 0
    run.summary["training/process_exit_code"] = exit_code
    run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
