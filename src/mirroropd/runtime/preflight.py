"""Model/data contracts are checked before reserving GPUs or creating workers."""

from dataclasses import replace
from pathlib import Path

from mirroropd.core.vocabulary import validate_tokenizers
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.data import evaluation_questions, prepare_data, read_frame
from mirroropd.runtime.evaluation import Protocol
from mirroropd.runtime.memory import check_cache_capacity
from mirroropd.runtime.provenance import FingerprintCache, model_artifact
from mirroropd.scoring import load_scorer, scorer_identity


def audit_lengths(tokenizer, paths, limit, *, question_limit=None):
    lengths = []
    remaining = question_limit
    for path in paths:
        prompts = read_frame(Path(path))["prompt"].tolist()
        if remaining is not None:
            prompts = prompts[:remaining]
            remaining -= len(prompts)
        for start in range(0, len(prompts), 128):
            group = [
                value.tolist() if hasattr(value, "tolist") else value
                for value in prompts[start : start + 128]
            ]
            encoded = tokenizer.apply_chat_template(
                group, tokenize=True, add_generation_prompt=True
            )
            lengths.extend(map(len, encoded))
        if remaining == 0:
            break
    if not lengths or max(lengths) > limit:
        raise ValueError(
            f"Prompt lengths exceed configured budget {limit}: maximum={max(lengths, default=0)}"
        )
    return {
        "questions": len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
    }


def prepare_inputs(job, directory, *, fingerprint_cache):
    from transformers import AutoConfig, AutoTokenizer

    directory = Path(directory)
    cache = FingerprintCache(fingerprint_cache)
    models = {
        role: model_artifact(
            getattr(job.models, role), cache, trust_remote_code=job.models.trust_remote_code
        )
        for role in ("student", "teacher")
    }
    tokenizers = {
        role: AutoTokenizer.from_pretrained(
            model["path"], local_files_only=True, trust_remote_code=job.models.trust_remote_code
        )
        for role, model in models.items()
    }
    if job.models.chat_template is not None:
        for tokenizer in tokenizers.values():
            tokenizer.chat_template = job.models.chat_template
    vocabulary_id = validate_tokenizers(tokenizers["student"], tokenizers["teacher"])
    vocab_size = len(tokenizers["student"])
    if set(tokenizers["student"].get_vocab().values()) != set(range(vocab_size)):
        raise ValueError("Shared vocabulary IDs must form the dense legal output prefix")
    if job.objective.estimator.topj > vocab_size:
        raise ValueError("Configured topj exceeds the shared vocabulary")
    student_config = AutoConfig.from_pretrained(
        models["student"]["path"],
        local_files_only=True,
        trust_remote_code=job.models.trust_remote_code,
    )
    memory_plan = check_cache_capacity(
        job.objective,
        job.memory,
        prefixes=(
            job.training.batch_size
            * job.training.responses_per_prompt
            * job.training.response_tokens
        ),
        vocab_size=vocab_size,
        hidden_size=getattr(student_config, "hidden_size", None),
        bytes_per_hidden=2 if job.precision.forward == "bf16" else 4,
        sequence_slots=job.training.batch_size
        * job.training.responses_per_prompt
        * (job.training.prompt_tokens + job.training.response_tokens),
    )
    for role, model in models.items():
        config = AutoConfig.from_pretrained(
            model["path"], local_files_only=True, trust_remote_code=job.models.trust_remote_code
        )
        needed = (
            job.max_context_tokens
            if role == "student"
            else job.training.prompt_tokens + job.training.response_tokens
        )
        capacity = getattr(config, "max_position_embeddings", getattr(config, "n_positions", None))
        if capacity is not None and capacity < needed:
            raise ValueError(f"{role} model context {capacity} is smaller than required {needed}")
        if getattr(config, "vocab_size", vocab_size) < vocab_size:
            raise ValueError("Model output head cannot cover the declared vocabulary")
    load_scorer(job.data.scorer)
    scorer_id = scorer_identity(job.data.scorer)
    dataset = prepare_data(job.data, directory / "data")
    if dataset["splits"]["train"]["rows"] < job.training.batch_size:
        raise ValueError("Training dataset is smaller than one full prompt batch")
    lengths = {
        "train": audit_lengths(
            tokenizers["student"], dataset["splits"]["train"]["paths"], job.training.prompt_tokens
        ),
        "validation": audit_lengths(
            tokenizers["student"],
            dataset["splits"]["validation"]["paths"],
            job.evaluation.prompt_tokens,
            question_limit=job.evaluation.question_limit,
        ),
    }
    job = replace(
        job,
        models=replace(
            job.models, student=models["student"]["path"], teacher=models["teacher"]["path"]
        ),
        data=replace(
            job.data,
            train=[item["source_path"] for item in dataset["files"] if item["split"] == "train"],
            validation=[
                item["source_path"] for item in dataset["files"] if item["split"] == "validation"
            ],
        ),
    )
    protocol = Protocol.from_job(job, scorer_id)
    evidence = {
        "schema_version": 2,
        "models": models,
        "vocabulary_id": vocabulary_id,
        "vocab_size": vocab_size,
        "memory_plan": memory_plan,
        "dataset": dataset,
        "prompt_lengths": lengths,
        "scorer_id": scorer_id,
        "protocol": protocol.to_dict(),
        "protocol_id": protocol.protocol_id,
        "evaluation_questions": evaluation_questions(dataset, job.evaluation.question_limit),
    }
    cache.save()
    atomic_json(directory / "inputs.json", evidence)
    atomic_json(directory / "job.json", job.to_dict())
    return job, evidence
