"""570-by-16 completeness, continuous scorers and actual generation metadata."""

from dataclasses import replace

import pytest

from mirroropd.runtime.evaluation import (
    Protocol,
    audit_rows,
    complete_metrics,
    question_uids,
    request_seed,
)


def protocol(kind="binary"):
    return Protocol(16, 0.7, 0.9, -1, 4096, 31744, 35840, 42, "test-scorer-sha", kind)


def generated_rows(sources=(("math", 500), ("amc", 40), ("aime", 30))):
    spec = protocol()
    protocol_id = spec.protocol_id
    rows = []
    for source, count in sources:
        identities = question_uids([source] * count, [f"question-{i}" for i in range(count)])
        for uid in identities:
            rows.extend(
                {
                    "schema_version": 2,
                    "protocol_id": protocol_id,
                    "model_version": 5,
                    "data_source": source,
                    "question_uid": uid,
                    "sample_index": sample,
                    "request_seed": request_seed(42, "evaluation", 5, uid, sample),
                    "score": float(sample % 2),
                    "response_tokens": 31744 if sample == 15 else 8000,
                    "finish_reason": "length" if sample == 15 else "stop",
                }
                for sample in range(16)
            )
    return rows


def test_all_570_questions_have_exactly_16_preserved_samples():
    rows = generated_rows()
    measured = audit_rows(rows, protocol(), 570, model_version=5)
    assert measured["evaluation/generated_examples"] == 9120
    assert measured["evaluation/unique_questions"] == 570
    for source in ("math", "amc", "aime"):
        assert measured[f"evaluation/{source}/mean@16"] == 0.5
        assert measured[f"evaluation/{source}/pass@16"] == 1.0


@pytest.mark.parametrize(
    "mutation",
    ["drop", "duplicate", "index", "seed", "version", "protocol", "budget", "finish", "score"],
)
def test_partial_or_mislabeled_evaluations_are_rejected(mutation):
    rows = generated_rows((("math", 1),))
    if mutation == "drop":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = dict(rows[0])
    elif mutation == "index":
        rows[-1]["sample_index"] = 1.5
    elif mutation == "seed":
        rows[-1]["request_seed"] += 1
    elif mutation == "version":
        rows[-1]["model_version"] = 4
    elif mutation == "protocol":
        rows[-1]["protocol_id"] = "old"
    elif mutation == "budget":
        rows[-1]["response_tokens"] = 7168
    elif mutation == "finish":
        rows[-1]["finish_reason"] = "abort"
    elif mutation == "score":
        rows[-1]["score"] = 0.5
    with pytest.raises((ValueError, TypeError)):
        audit_rows(rows, protocol(), 1, model_version=5)


def test_continuous_rewards_have_means_without_fake_pass_accuracy():
    spec = replace(protocol("scalar"), samples_per_question=2)
    measured = complete_metrics(["text", "text"], ["q", "q"], [-1.5, 2.5], [0, 1], spec, 1)
    assert measured["evaluation/text/mean@2"] == 0.5
    assert not any("pass@" in key for key in measured)
    assert spec.protocol_id != replace(spec, seed=43).protocol_id


def test_request_seeds_separate_phases_and_do_not_depend_on_request_order():
    a = request_seed(42, "evaluation", 5, "q", 0)
    values = [
        request_seed(42, phase, outer, uid, sample)
        for phase, outer, uid, sample in [
            ("train", 5, "q", 0),
            ("evaluation", 6, "q", 0),
            ("evaluation", 5, "q", 1),
            ("evaluation", 5, "other", 0),
        ]
    ]
    assert a not in values
    assert request_seed(42, "evaluation", 5, "q", 0) == a
    assert Protocol.from_dict(protocol().to_dict()) == protocol()
