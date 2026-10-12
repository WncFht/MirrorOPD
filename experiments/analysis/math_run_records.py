"""Validated legacy math-run records shared by fixed reviews and live plots."""

import csv
import json
from pathlib import Path

import numpy as np

RUNS = {
    "OPD": "dapo-deepseek-opd-o50-j1",
    "MDPO": "dapo-deepseek-mdpo-o50-j16",
    "BPO": "dapo-deepseek-bpo-o50-j16",
}
BENCHMARKS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}
COLORS = {"OPD": "#0072B2", "MDPO": "#D55E00", "BPO": "#009E73"}


def read_rows(path):
    """Read a frozen JSONL source; live sources must be snapshotted first."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    """Export heterogeneous event records with their complete field union."""
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, key):
    """Return one globally token-reduced legacy metric per inner event."""
    return np.asarray([r["metrics"]["mirror/" + key] for r in rows])


def load_run(root, name):
    """Validate completed outers of the B=mini benchmark, excluding pending rows."""
    directory = Path(root) / name
    all_inner = read_rows(directory / "inner.jsonl")
    outer = read_rows(directory / "outer.jsonl")
    training = [r for r in outer if "timing_s/step" in r["data"]]
    completed = len(training)
    inner = [r for r in all_inner if r["outer_call"] < completed]
    assert [r["step"] for r in training] == list(range(1, completed + 1))
    epochs = inner[0]["epochs"]
    assert [(r["outer_call"], r["inner_step"]) for r in inner] == [
        (k, n) for k in range(completed) for n in range(epochs)
    ]
    times = np.concatenate(([0.0], np.cumsum([r["data"]["timing_s/step"] for r in training])))
    evaluation = [r for r in outer if "evaluation/generated_examples" in r["data"]]
    first_evaluation = evaluation[0]["data"]
    if "evaluation/unique_questions" in first_evaluation:
        repeats = int(first_evaluation["evaluation/generations_per_question"])
        protocol = first_evaluation["evaluation/protocol_id"]
        for row in evaluation:
            data = row["data"]
            assert data["evaluation/unique_questions"] == 570
            assert data["evaluation/generations_per_question"] == repeats
            assert data["evaluation/generated_examples"] == 570 * repeats
            assert data["evaluation/protocol_id"] == protocol
    else:
        # The completed benchmark predates the UID/repeat protocol metadata.
        assert all(r["data"]["evaluation/generated_examples"] == 570 for r in evaluation)
        repeats, protocol = 1, "legacy-greedy-l7168"
    groups = [inner[k * epochs : (k + 1) * epochs] for k in range(completed)]
    return {
        "name": name,
        "directory": directory,
        "N": epochs,
        "completed": completed,
        "inner": inner,
        "pending_inner": len(all_inner) - len(inner),
        "outer": outer,
        "training": training,
        "evaluation": evaluation,
        "groups": groups,
        "first": [g[0] for g in groups],
        "last": [g[-1] for g in groups],
        "times": times,
        "R": repeats,
        "protocol": protocol,
    }
