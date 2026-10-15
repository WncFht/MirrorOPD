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
    audit = EvaluationAudit(protocol, expected_questions, model_version)
    for row in rows:
        audit.add(row)
    return audit.finish()


def audit_file(path, protocol, expected_questions, *, model_version):
    with path.open() as stream:
        return audit_rows(
            (json.loads(line) for line in stream if line.strip()),
            protocol,
            expected_questions,
            model_version=model_version,
        )


class EvaluationAudit:
    """Streaming artifact validation retains scores/identities, never response text."""

    def __init__(self, protocol, expected_questions, model_version):
        self.protocol = protocol
        self.expected_questions = expected_questions
        self.model_version = model_version
        self.groups = defaultdict(dict)
        self.lengths = defaultdict(lambda: [0, 0, 0, 0])
        self.protocol_id = protocol.protocol_id

    def add(self, row):
        if row.get("schema_version") != API_VERSION or row.get("protocol_id") != self.protocol_id:
            raise ValueError("Generation artifact schema/protocol mismatch")
        if type(row.get("model_version")) is not int or row["model_version"] != self.model_version:
            raise ValueError("Generation artifact has a stale student weight version")
        sample = row["sample_index"]
        if not isinstance(sample, Integral) or isinstance(sample, bool):
            raise TypeError("Sample indices must be integers")
        key = (str(row["data_source"]), str(row["question_uid"]))
        if sample in self.groups[key]:
            raise ValueError("Duplicated question/sample identity")
        score = float(row["score"])
        if not math.isfinite(score) or (
            self.protocol.score_kind == "binary" and score not in {0.0, 1.0}
        ):
            raise ValueError("Score is outside the declared evaluation domain")
        count = row["response_tokens"]
        if type(count) is not int or not 0 < count <= self.protocol.response_tokens:
            raise ValueError("Invalid response token count")
        reason = row["finish_reason"]
        if reason not in {"stop", "length"}:
            raise ValueError("Evaluation contains unfinished generation")
        if reason == "length" and count != self.protocol.response_tokens:
            raise ValueError("Length-terminated output did not preserve the full budget")
        expected = request_seed(
            self.protocol.seed, "evaluation", self.model_version, key[1], int(sample)
        )
        if row.get("request_seed") != expected:
            raise ValueError("Generation did not report its requested deterministic sample seed")
        self.groups[key][int(sample)] = score
        stats = self.lengths[key[0]]
        stats[0] += 1
        stats[1] += count
        stats[2] += int(reason == "length")
        stats[3] += int(row.get("eos_terminated", False))

    def finish(self):
        sources, uids, scores, indices = [], [], [], []
        for (source, uid), samples in self.groups.items():
            for index, score in samples.items():
                sources.append(source)
                uids.append(uid)
                scores.append(score)
                indices.append(index)
        result = complete_metrics(
            sources, uids, scores, indices, self.protocol, self.expected_questions
        )
        for source, (count, total, length, eos) in self.lengths.items():
            result.update(
                {
                    f"evaluation/{source}/response_tokens/mean": total / count,
                    f"evaluation/{source}/budget_exhausted_fraction": length / count,
                    f"evaluation/{source}/eos_fraction": eos / count,
                }
            )
        return result
