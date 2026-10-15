"""Dataset normalization and scorer plugins are independent of model and verl imports."""

import json
import sys

import pandas as pd
import pytest

from mirroropd.runtime.config import Columns, Dataset, Scorer
from mirroropd.runtime.data import evaluation_questions, prepare_data
from mirroropd.scoring import load_scorer, score, scorer_identity


def test_jsonl_csv_and_canonical_parquet_share_one_preparation_contract(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "test.csv"
    train.write_text(json.dumps({"question": "unlabeled train"}) + "\n")
    pd.DataFrame([{"question": "q1", "answer": "a1"}, {"question": "q1", "answer": "a1"}]).to_csv(
        validation, index=False
    )
    dataset = Dataset(
        [str(train)],
        [str(validation)],
        columns=Columns("question", "answer", "unused"),
        source_name="custom",
    )
    manifest = prepare_data(dataset, tmp_path / "prepared")
    assert manifest["splits"]["train"]["labeled_rows"] == 0
    assert manifest["splits"]["validation"]["labeled_rows"] == 2
    rows = pd.read_parquet(manifest["splits"]["validation"]["paths"][0]).to_dict(orient="records")
    assert len({row["uid"] for row in rows}) == 2
    assert rows[0]["reward_model"]["ground_truth"] == "a1"
    assert rows[0]["prompt"][0]["content"] == "q1"
    assert evaluation_questions(manifest, None) == 2
    assert evaluation_questions(manifest, 1) == 1
    rerun = prepare_data(dataset, tmp_path / "again")
    assert rerun["dataset_id"] == manifest["dataset_id"]
    canonical = Dataset(
        manifest["splits"]["train"]["paths"], manifest["splits"]["validation"]["paths"]
    )
    repeated = prepare_data(canonical, tmp_path / "canonical")
    assert repeated["splits"]["validation"]["rows"] == 2
    assert repeated["splits"]["train"]["labeled_rows"] == 0


def test_missing_validation_answers_and_configured_columns_fail(tmp_path):
    file = tmp_path / "data.jsonl"
    file.write_text(json.dumps({"question": "q"}) + "\n")
    dataset = Dataset(
        [str(file)],
        [str(file)],
        columns=Columns("question", "answer", "unused"),
        source_name="custom",
    )
    with pytest.raises(ValueError, match="configured field"):
        prepare_data(dataset, tmp_path / "prepared")


@pytest.mark.parametrize(
    ("text", "truth", "correct"),
    [
        ("reasoning #### 1,234", "1234", 1),
        (r"x \boxed{-.5}", "-0.5", 1),
        ("final 8", "7", 0),
        ("unknown", "7", 0),
    ],
)
def test_numeric_scoring(text, truth, correct):
    assert score(Scorer("gsm8k"), "gsm8k", text, truth)["score"] == correct


@pytest.mark.parametrize(
    ("text", "truth", "correct", "formatted"),
    [
        (r"The answer is \boxed{37}", "37", 1, 1),
        (r"The answer is \boxed{38}", "37", 0, 1),
        (r"The answer is \boxed{\frac{1}{2}}", "0.5", 1, 1),
        ("The final answer is 37.", "37", 1, 0),
        ("I cannot solve this problem.", "37", 0, 0),
    ],
)
def test_pinned_math_protocol_runs_without_a_verl_checkout(text, truth, correct, formatted):
    result = score(Scorer("math"), "math", text, truth)
    assert result["score"] == correct
    assert result["format_score"] == formatted


def test_custom_scorer_parameters_and_domains_are_versioned(tmp_path, monkeypatch):
    module = tmp_path / "test_scorer_plugin.py"
    module.write_text(
        "def score(source,text,truth,extra_info=None,scale=1):\n    return {'score':len(text)*scale}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = Scorer("test_scorer_plugin:score", "scalar", {"scale": 0.5})
    try:
        assert score(spec, "source", "abc", "ignored")["score"] == 1.5
        assert scorer_identity(spec) != scorer_identity(Scorer(spec.name, "scalar", {"scale": 1}))
        with pytest.raises(ValueError, match="declared domain"):
            score(Scorer(spec.name, "binary"), "source", "abc", "ignored")
        with pytest.raises(TypeError, match="unexpected keyword"):
            load_scorer(Scorer("exact", parameters={"typo": 1}))
    finally:
        sys.modules.pop("test_scorer_plugin", None)
