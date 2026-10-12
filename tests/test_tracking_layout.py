"""Check lossless W&B partitioning and axis isolation from local raw metrics."""

import json
from pathlib import Path

from mirroropd.training.tracking import configure_wandb, grouped_metrics


class FakeRun:
    def __init__(self):
        self.summary = {"setting/topk": 16}
        self.axes = []

    def define_metric(self, name, **kwargs):
        self.axes.append((name, kwargs))


def main():
    records = Path("results/training/setting-k16-mdpo-b32-j32/inner.jsonl")
    original = json.loads(records.read_text().splitlines()[0])["metrics"]
    grouped = grouped_metrics(original, "inner")
    assert len(grouped) == len(original), "Renaming must not collide or drop fields"
    assert sorted(grouped.values()) == sorted(original.values())
    assert "02_offpolicy/inner_importance_ess_fraction" in grouped
    assert "90_detail_extrema/inner_score_coefficient/max" in grouped
    assert "02_offpolicy/inner_importance_ratio/max" in grouped
    run = FakeRun()
    configure_wandb(run, {})
    assert run.summary["setting/topk"] == 16, "Publishing must preserve settings"
    assert (
        "01_quality/*",
        {"step_metric": "00_axes/evaluation_outer", "step_sync": False},
    ) in run.axes
    print("W&B grouping preserves all metrics and separates inner/evaluation axes")


if __name__ == "__main__":
    main()
