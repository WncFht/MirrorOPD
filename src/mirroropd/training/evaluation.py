"""Validation budgets and complete per-question Monte Carlo evidence.

This module is independent of Ray, CUDA and verl. Length selection follows
MD-OPD's effective_response_length contract; validation never changes the
training response budget.
"""

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _get(config, key, default=None):
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def effective_response_length(rollout_config, validate):
    if validate:
        value = _get(_get(rollout_config, "val_kwargs", {}), "max_tokens")
        if value is not None:
            if int(value) < 1:
                raise ValueError("Validation max_tokens must be positive")
            return int(value)
    return int(_get(rollout_config, "response_length"))


def effective_prompt_length(data_config, validate):
    value = _get(data_config, "val_max_prompt_length") if validate else None
    return int(value if value is not None else _get(data_config, "max_prompt_length"))


def phase_data_config(data_config, validate):
    """Clone the dataset config so validation cannot mutate training budgets."""
    result = copy.deepcopy(data_config)
    result["max_prompt_length"] = effective_prompt_length(data_config, validate)
    return result


def evaluation_protocol(config):
    rollout = _get(_get(config, "actor_rollout_ref"), "rollout")
    data = _get(config, "data")
    kwargs = _get(rollout, "val_kwargs")
    repeats = int(_get(kwargs, "n", 1))
    if repeats < 1:
        raise ValueError("Validation n must be positive")
    protocol = {
        "samples_per_question": repeats,
        "do_sample": bool(_get(kwargs, "do_sample", True)),
        "temperature": float(_get(kwargs, "temperature", 1.0)),
        "top_p": float(_get(kwargs, "top_p", 1.0)),
        "top_k": int(_get(kwargs, "top_k", -1)),
        "max_response_tokens": effective_response_length(rollout, True),
        "max_prompt_tokens": effective_prompt_length(data, True),
        "max_model_len": int(_get(rollout, "max_model_len")),
    }
    if protocol["max_model_len"] < protocol["max_prompt_tokens"] + protocol["max_response_tokens"]:
        raise ValueError("Validation context does not cover prompt plus full response budget")
    protocol["protocol_id"] = (
        "math-eval-"
        + hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    )
    return protocol


def stable_question_uids(data_sources, raw_prompts, existing=None):
    """Namespace supplied UIDs, or hash source and raw question deterministically."""
    result = []
    if existing is None:
        existing = [None] * len(raw_prompts)
    for source, prompt, uid in zip(data_sources, raw_prompts, existing, strict=True):
        question_uid = uid
        if question_uid is None:
            prompt_value = prompt.tolist() if hasattr(prompt, "tolist") else prompt
            payload = json.dumps(prompt_value, sort_keys=True, ensure_ascii=False, default=str)
            question_uid = hashlib.sha256(payload.encode()).hexdigest()
        result.append(f"{source}:{question_uid}")
    if len(set(result)) != len(result):
        raise ValueError("Validation question UIDs must be unique before repetition")
    return result


def complete_sample_metrics(
    data_sources, uids, scores, repeats, expected_questions, sample_indices=None
):
    """Reject incomplete groups and report mean@R and empirical pass@R.

    Every UID must occur exactly R times. pass@R is the fraction of questions
    with at least one correct output, not an average of individual outputs.
    """
    if repeats < 1 or expected_questions < 1:
        raise ValueError("Positive sample and question counts are required")
    groups = defaultdict(list)
    indices = defaultdict(list)
    for position, (source, uid, score) in enumerate(zip(data_sources, uids, scores, strict=True)):
        score_value = float(score)
        if not math.isfinite(score_value) or score_value not in (0.0, 1.0):
            raise ValueError("Math validation scores must be finite binary verifier results")
        groups[(str(source), str(uid))].append(score_value)
        if sample_indices is not None:
            indices[(str(source), str(uid))].append(int(sample_indices[position]))
    if len(groups) != expected_questions or len(set(map(str, uids))) != expected_questions:
        raise ValueError("Incorrect number of unique validation question UIDs")
    for key, values in groups.items():
        if len(values) != repeats:
            raise ValueError(f"Incomplete validation samples for {key}: {len(values)} != {repeats}")
        if sample_indices is not None and sorted(indices[key]) != list(range(repeats)):
            raise ValueError(f"Duplicated or missing validation sample index for {key}")
    metrics = {
        "evaluation/generated_examples": len(uids),
        "evaluation/unique_questions": len(groups),
        "evaluation/samples_per_question": repeats,
        "evaluation/generations_per_question": repeats,
    }
    by_source = defaultdict(list)
    for (source, _), values in groups.items():
        by_source[source].append(values)
    for source, values in by_source.items():
        metrics[f"val-core/{source}/acc/mean@{repeats}"] = sum(map(sum, values)) / (
            len(values) * repeats
        )
        metrics[f"val-core/{source}/acc/pass@{repeats}"] = sum(max(v) for v in values) / len(values)
    return metrics


def generation_metadata(uids, data_sources, finish_reasons):
    """Align stable UID, zero-based sample index, source and real engine finish reason."""
    counter = Counter()
    sample_indices = []
    for uid in uids:
        sample_indices.append(counter[str(uid)])
        counter[str(uid)] += 1
    if len(finish_reasons) != len(uids) or any(
        reason in (None, "abort", "aborted") for reason in finish_reasons
    ):
        raise ValueError("Validation requires completed generation finish reasons")
    return {
        "uid": list(map(str, uids)),
        "sample_index": sample_indices,
        "data_source": list(map(str, data_sources)),
        "finish_reason": list(map(str, finish_reasons)),
    }


def audit_generation_rows(rows, protocol, expected_questions):
    metrics = complete_sample_metrics(
        [row["data_source"] for row in rows],
        [row["uid"] for row in rows],
        [row["score"] for row in rows],
        protocol["samples_per_question"],
        expected_questions,
        [row["sample_index"] for row in rows],
    )
    for row in rows:
        if row.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError("Generation protocol differs from resolved config")
        if row["finish_reason"] not in ("stop", "length"):
            raise ValueError("Unfinished validation output")
        if not 0 < row["response_tokens"] <= protocol["max_response_tokens"]:
            raise ValueError("Invalid validation response length")
        if (
            row["finish_reason"] == "length"
            and row["response_tokens"] != protocol["max_response_tokens"]
        ):
            raise ValueError("Length-terminated validation output did not use the full budget")
    return metrics


def audit_evaluation_records(run_dir, config, evaluations, expected_questions):
    """Keep historical n1 summaries readable; require saved UID evidence for n>1."""
    protocol = evaluation_protocol(config)
    repeats = protocol["samples_per_question"]
    expected_examples = expected_questions * repeats
    question_uids = None
    save_dir = _get(_get(config, "trainer"), "validation_data_dir")
    for evaluation in evaluations:
        metrics = evaluation["data"]
        if metrics["evaluation/generated_examples"] != expected_examples:
            raise ValueError("Incomplete evaluation dataset or repetitions")
        if repeats > 1:
            if metrics.get("evaluation/unique_questions") != expected_questions:
                raise ValueError("Incomplete unique validation questions")
            if metrics.get("evaluation/generations_per_question") != repeats:
                raise ValueError("Incorrect evaluation repetitions")
            if metrics.get("evaluation/protocol_id") != protocol["protocol_id"]:
                raise ValueError("Evaluation protocol differs from resolved config")
            if not save_dir:
                raise ValueError("Multi-sample validation requires saved per-question outputs")
            # Archived runs can be moved: use the original path when present,
            # otherwise the launcher-standard validation subdirectory.
            directory = Path(save_dir)
            if not directory.exists():
                directory = Path(run_dir) / "validation"
            path = directory / f"{evaluation['step']}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if any(row["step"] != evaluation["step"] for row in rows):
                raise ValueError("Generation artifact has the wrong evaluation step")
            measured = audit_generation_rows(rows, protocol, expected_questions)
            for key, value in measured.items():
                if key not in metrics or not math.isclose(metrics[key], value, abs_tol=1e-12):
                    raise ValueError(f"Generation evidence disagrees with aggregate metric {key}")
            current_uids = {row["uid"] for row in rows}
            if question_uids is not None and current_uids != question_uids:
                raise ValueError("Validation question UIDs changed across evaluation rounds")
            question_uids = current_uids
    return {
        "protocol": protocol,
        "evaluation_questions": expected_questions,
        "evaluation_examples": expected_examples,
    }
