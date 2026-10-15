"""Native rollout orchestration with one complete, versioned evaluation artifact."""

import copy
import os
import tempfile
from pathlib import Path

import numpy as np

from mirroropd.core.config import API_VERSION
from mirroropd.identity import canonical_json
from mirroropd.integrations.verl.bridge import validate_rollout_versions
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.evaluation import EvaluationAudit, Protocol, question_uids
from mirroropd.runtime.events import environment_writer, phase


def response_tokens(rollout, validate):
    value = rollout.val_kwargs.get("max_tokens") if validate else None
    return int(value if value is not None else rollout.response_length)


def prompt_tokens(data, validate):
    value = data.get("val_max_prompt_length") if validate else None
    return int(value if value is not None else data.max_prompt_length)


def phase_data(data, validate):
    result = copy.deepcopy(data)
    result["max_prompt_length"] = prompt_tokens(data, validate)
    return result


def _protocol(config):
    from omegaconf import OmegaConf

    value = config.trainer.mirror_protocol
    return Protocol.from_dict(
        OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    )


def token_metadata(ids, mask, eos_token_id, finish_reason, budget):
    values = [int(token) for token, valid in zip(ids, mask, strict=True) if valid]
    eos = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    if not 0 < len(values) <= budget:
        raise ValueError("Generated response violates its explicit token budget")
    return {
        "response_tokens": len(values),
        "eos_terminated": bool(values[-1] in eos),
        "finish_reason": str(finish_reason),
    }


def evaluate(trainer):
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.trainer.ppo.reward import extract_reward

    protocol = _protocol(trainer.config)
    version = trainer.global_steps
    audit = EvaluationAudit(
        protocol, int(trainer.config.trainer.mirror_expected_questions), version
    )
    directory = Path(trainer.config.trainer.validation_data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{version}.jsonl"
    if destination.exists():
        raise ValueError("Completed evaluation artifact already exists; refusing to replace it")
    temporary = None
    try:
        with (
            phase("evaluation", model_version=version, protocol_id=protocol.protocol_id),
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                prefix=f"{version}.",
                suffix=".partial",
                delete=False,
            ) as stream,
        ):
            temporary = Path(stream.name)
            for values in trainer.val_dataloader:
                batch = DataProto.from_single_dict(values)
                identities = question_uids(
                    batch.non_tensor_batch["data_source"],
                    batch.non_tensor_batch["raw_prompt"],
                    batch.non_tensor_batch.get("uid"),
                )
                batch.non_tensor_batch["uid"] = np.array(identities, dtype=object)
                batch = batch.repeat(repeat_times=protocol.samples_per_question, interleave=True)
                samples = np.tile(np.arange(protocol.samples_per_question), len(identities))
                batch.non_tensor_batch["sample_index"] = samples
                generated = trainer._get_gen_batch(batch)  # noqa: SLF001 -- pinned native generation contract
                generated.meta_info = {
                    "eos_token_id": trainer.tokenizer.eos_token_id,
                    "pad_token_id": trainer.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "validate": True,
                    "global_steps": version,
                    "mirror_generation_version": version,
                }
                padded, pad_size = pad_dataproto_to_divisor(
                    generated, trainer.config.actor_rollout_ref.rollout.agent.num_workers
                )
                output = unpad_dataproto(
                    trainer.async_rollout_manager.generate_sequences(padded), pad_size=pad_size
                )
                validate_rollout_versions(output.non_tensor_batch, version, len(batch))
                combined = batch.union(output)
                combined.meta_info["validate"] = True
                rewards, _extra = extract_reward(combined)
                scores = rewards.sum(-1).cpu().tolist()
                responses = output.batch["responses"].cpu()
                mask = output.batch["response_mask"].cpu()
                inputs = trainer.tokenizer.batch_decode(
                    output.batch["prompts"], skip_special_tokens=True
                )
                texts = trainer.tokenizer.batch_decode(responses, skip_special_tokens=True)
                for index, (identity, sample, score) in enumerate(
                    zip(batch.non_tensor_batch["uid"], samples, scores, strict=True)
                ):
                    row = {
                        "schema_version": API_VERSION,
                        "protocol_id": protocol.protocol_id,
                        "model_version": version,
                        "question_uid": str(identity),
                        "sample_index": int(sample),
                        "data_source": str(batch.non_tensor_batch["data_source"][index]),
                        "request_seed": int(output.non_tensor_batch["generation_seed"][index]),
                        "input": inputs[index],
                        "output": texts[index],
                        "ground_truth": batch.non_tensor_batch["reward_model"][index][
                            "ground_truth"
                        ],
                        "score": float(score),
                        **token_metadata(
                            responses[index].tolist(),
                            mask[index].tolist(),
                            trainer.tokenizer.eos_token_id,
                            output.non_tensor_batch["generation_finish_reason"][index],
                            protocol.response_tokens,
                        ),
                    }
                    audit.add(row)
                    stream.write(canonical_json(row) + "\n")
            metrics = audit.finish()
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        result = {
            "schema_version": API_VERSION,
            "model_version": version,
            "protocol_id": protocol.protocol_id,
            "artifact": str(destination),
            "metrics": metrics,
            "passed": True,
        }
        atomic_json(directory / f"{version}.summary.json", result)
        writer = environment_writer("driver")
        if writer:
            writer.emit("evaluation.completed", **result)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metrics
