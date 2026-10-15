"""Complete sampled evaluation, explicit scoring domains and request identities."""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from numbers import Integral

from mirroropd.core.config import API_VERSION, decode
from mirroropd.identity import canonical_json, content_id


@dataclass(frozen=True, slots=True)
class Protocol:
    samples_per_question: int
    temperature: float
    top_p: float
    top_k: int
    prompt_tokens: int
    response_tokens: int
    max_context_tokens: int
    seed: int
    scorer_id: str
    score_kind: str
    schema_version: int = API_VERSION

    def __post_init__(self):
        if self.schema_version != API_VERSION:
            raise ValueError("Unsupported evaluation schema")
        if min(self.samples_per_question, self.prompt_tokens, self.response_tokens) < 1:
            raise ValueError("Evaluation counts and token budgets must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Sampled evaluation requires positive temperature and top_p in (0,1]")
        if self.max_context_tokens < self.prompt_tokens + self.response_tokens:
            raise ValueError(
                "Evaluation context does not cover the independent full response budget"
            )
        if self.score_kind not in {"binary", "scalar"} or not self.scorer_id:
            raise ValueError("Evaluation requires an explicit scorer identity and score domain")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("Evaluation top_k must be -1 or positive")

    @property
    def protocol_id(self):
        return content_id(asdict(self))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return decode(cls, value)

    @classmethod
    def from_job(cls, job, scorer_id):
        setting = job.evaluation
        return cls(
            setting.samples_per_question,
            setting.temperature,
            setting.top_p,
            setting.top_k,
            setting.prompt_tokens,
            setting.response_tokens,
            job.max_context_tokens,
            setting.seed,
            scorer_id,
            job.data.scorer.kind,
        )


def request_seed(base_seed, phase, outer, question_uid, sample_index):
    if phase not in {"train", "evaluation"} or outer < 0 or sample_index < 0:
        raise ValueError("Request seed requires valid phase/outer/sample coordinates")
    value = canonical_json([API_VERSION, base_seed, phase, outer, str(question_uid), sample_index])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**31)


def question_uids(data_sources, raw_prompts, existing=None):
    identities = existing if existing is not None else [None] * len(raw_prompts)
    result = []
    for source, prompt, identity in zip(data_sources, raw_prompts, identities, strict=True):
        question_identity = identity
        if identity is None:
            normalized = prompt.tolist() if hasattr(prompt, "tolist") else prompt
            question_identity = canonical_json(normalized)
        result.append(content_id({"source": str(source), "question": str(question_identity)}))
    if len(set(result)) != len(result):
        raise ValueError("Validation rows need distinct question UIDs before repetition")
    return result


def complete_metrics(data_sources, uids, scores, sample_indices, protocol, expected_questions):
    if expected_questions < 1:
        raise ValueError("Evaluation must contain questions")
    groups = defaultdict(dict)
    for source, uid, score, sample in zip(data_sources, uids, scores, sample_indices, strict=True):
        score_value = float(score)
        if not math.isfinite(score_value) or (
            protocol.score_kind == "binary" and score_value not in {0.0, 1.0}
        ):
            raise ValueError("Score is outside the declared evaluation domain")
        key = (str(source), str(uid))
        if not isinstance(sample, Integral) or isinstance(sample, bool):
            raise TypeError("Sample indices must be integers")
        if sample in groups[key]:
            raise ValueError("Duplicated question/sample identity")
        groups[key][int(sample)] = score_value
    if len(groups) != expected_questions or len(set(map(str, uids))) != expected_questions:
        raise ValueError("Evaluation has the wrong number of distinct question UIDs")
    expected = set(range(protocol.samples_per_question))
    by_source = defaultdict(list)
    for (source, _uid), samples in groups.items():
        if set(samples) != expected:
            raise ValueError("Incomplete or out-of-range evaluation samples")
        by_source[source].append(list(samples.values()))
    metrics = {
        "evaluation/generated_examples": len(scores),
        "evaluation/unique_questions": len(groups),
        "evaluation/samples_per_question": protocol.samples_per_question,
    }
    for source, samples in by_source.items():
        metrics[f"evaluation/{source}/mean@{protocol.samples_per_question}"] = sum(
            map(sum, samples)
        ) / (len(samples) * protocol.samples_per_question)
        if protocol.score_kind == "binary":
            metrics[f"evaluation/{source}/pass@{protocol.samples_per_question}"] = sum(
                max(values) for values in samples
            ) / len(samples)
    return metrics


def audit_rows(rows, protocol, expected_questions, *, model_version):
    metrics = complete_metrics(
        [r["data_source"] for r in rows],
        [r["question_uid"] for r in rows],
        [r["score"] for r in rows],
        [r["sample_index"] for r in rows],
        protocol,
        expected_questions,
    )
    for row in rows:
        if (
            row.get("schema_version") != API_VERSION
            or row.get("protocol_id") != protocol.protocol_id
        ):
            raise ValueError("Generation artifact schema/protocol mismatch")
        if row.get("model_version") != model_version:
            raise ValueError("Generation artifact has a stale student weight version")
        if row["finish_reason"] not in {"stop", "length"}:
            raise ValueError("Evaluation contains unfinished generation")
        count = row["response_tokens"]
        if not isinstance(count, int) or not 0 < count <= protocol.response_tokens:
            raise ValueError("Invalid response token count")
        if row["finish_reason"] == "length" and count != protocol.response_tokens:
            raise ValueError("Length-terminated output did not preserve the full budget")
        expected = request_seed(
            protocol.seed, "evaluation", model_version, row["question_uid"], row["sample_index"]
        )
        if row.get("request_seed") != expected:
            raise ValueError("Generation did not report the requested deterministic sample seed")
    return metrics


def audit_file(path, protocol, expected_questions, *, model_version):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return audit_rows(rows, protocol, expected_questions, model_version=model_version)
