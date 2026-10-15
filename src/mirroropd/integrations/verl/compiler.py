"""Compile one validated job to resolved backend YAML without shell interpolation."""

from dataclasses import asdict
from pathlib import Path


def backend_values(job, evidence, run_dir, source_root, environment):
    run_dir, source_root = Path(run_dir), Path(source_root)
    training, evaluation, resources = job.training, job.evaluation, job.resources
    precision = job.precision
    model = {
        "path": job.models.student,
        "trust_remote_code": job.models.trust_remote_code,
        "custom_chat_template": job.models.chat_template,
        "lora_rank": 0,
        "enable_gradient_checkpointing": training.gradient_checkpointing,
        "use_remove_padding": True,
        "use_fused_kernels": False,
        "mirror_policy": True,
    }
    mixed = {
        "param_dtype": precision.forward,
        "reduce_dtype": precision.reduction,
        "buffer_dtype": precision.buffers,
    }
    micro = {
        "log_prob_micro_batch_size_per_gpu": training.micro_batch_size,
        "log_prob_use_dynamic_bsz": False,
    }
    actor = {
        "ppo_mini_batch_size": training.batch_size,
        "ppo_micro_batch_size_per_gpu": training.micro_batch_size,
        "ppo_epochs": training.inner_steps,
        "use_dynamic_bsz": False,
        "entropy_coeff": 0.0,
        "use_kl_loss": False,
        "grad_clip": training.clip_grad,
        "optim": {
            **{key: value for key, value in asdict(training.optim).items() if key != "eps"},
            "lr": training.learning_rate,
            "weight_decay": training.weight_decay,
            "clip_grad": training.clip_grad,
            "override_optimizer_config": {"eps": training.optim.eps},
        },
        "fsdp_config": {
            "model_dtype": precision.parameters,
            "param_offload": training.parameter_offload,
            "optimizer_offload": training.optimizer_offload,
            "use_torch_compile": False,
            "mixed_precision": mixed,
        },
        "checkpoint": {
            "save_contents": job.checkpoints.save_contents if job.checkpoints.every else []
        },
    }
    reference = micro | {
        "fsdp_config": {
            "model_dtype": precision.teacher,
            "param_offload": training.parameter_offload
            if job.memory.teacher_parameter_offload is None
            else job.memory.teacher_parameter_offload,
            "use_torch_compile": False,
            "mixed_precision": mixed,
        }
    }
    rollout = micro | {
        "name": "vllm",
        "dtype": "bfloat16" if precision.forward == "bf16" else "float32",
        "tensor_model_parallel_size": 1,
        "gpu_memory_utilization": resources.rollout_memory_fraction,
        "max_model_len": job.max_context_tokens,
        "max_num_seqs": resources.max_sequences,
        "max_num_batched_tokens": resources.max_batch_tokens,
        "calculate_log_probs": True,
        "enforce_eager": True,
        "temperature": training.temperature,
        "top_p": 1.0,
        "top_k": -1,
        "n": training.responses_per_prompt,
        "seed": training.seed,
        "agent": {"num_workers": 1},
        "val_kwargs": {
            "n": evaluation.samples_per_question,
            "do_sample": True,
            "temperature": evaluation.temperature,
            "top_p": evaluation.top_p,
            "top_k": evaluation.top_k,
            "max_tokens": evaluation.response_tokens,
            "seed": evaluation.seed,
        },
    }
    datasets = evidence["dataset"]["splits"]
    data = {
        "train_files": datasets["train"]["paths"],
        "val_files": datasets["validation"]["paths"],
        "train_batch_size": training.batch_size,
        "val_batch_size": None,
        "dataloader_num_workers": 0,
        "max_prompt_length": training.prompt_tokens,
        "max_response_length": training.response_tokens,
        "val_max_prompt_length": evaluation.prompt_tokens,
        "val_max_samples": evaluation.question_limit or -1,
        "truncation": "error",
        "filter_overlong_prompts": False,
        "seed": training.seed,
        "validation_shuffle": False,
    }
    mirror = {
        "schema_version": 2,
        "teacher_model_path": job.models.teacher,
        "teacher_id": evidence["models"]["teacher"]["identity"],
        "tokenizer_id": evidence["vocabulary_id"],
        "vocab_size": evidence["vocab_size"],
        "objective": job.objective.to_dict(),
        "memory": asdict(job.memory),
    }
    distillation = {
        "enabled": True,
        "n_gpus_per_node": 0,
        "nnodes": 0,
        "mirror": mirror,
        "distillation_loss": {
            "loss_mode": "mirror",
            "topk": job.objective.estimator.topj,
            "use_task_rewards": False,
            "use_policy_gradient": False,
            "loss_max_clamp": None,
            "log_prob_min_clamp": None,
        },
    }
    trainer = {
        "total_training_steps": training.outer_steps,
        "total_epochs": 1,
        "n_gpus_per_node": len(resources.gpu_ids),
        "nnodes": 1,
        "val_before_train": evaluation.before_training,
        "test_freq": evaluation.every,
        "save_freq": job.checkpoints.every if job.checkpoints.every else -1,
        "resume_mode": "disable",
        "default_local_dir": str(run_dir / "checkpoints"),
        "validation_data_dir": str(run_dir / "validation"),
        "rollout_data_dir": None,
        "log_val_generations": 0,
        "max_actor_ckpt_to_keep": job.checkpoints.max_to_keep,
        "project_name": job.logging.wandb.project,
        "experiment_name": environment.get("WANDB_NAME", job.name),
        "logger": ["file"] + (["wandb"] if job.logging.wandb.enabled else []),
        "mirror_protocol": evidence["protocol"],
        "mirror_expected_questions": evidence["evaluation_questions"],
    }
    return {
        "data": data,
        "actor_rollout_ref": {"model": model, "actor": actor, "ref": reference, "rollout": rollout},
        "distillation": distillation,
        "critic": {"enable": False},
        "algorithm": {
            "adv_estimator": "reinforce_plus_plus",
            "use_kl_in_reward": False,
            "rollout_correction": None,
        },
        "reward": {
            "num_workers": 1,
            "custom_reward_function": {
                "path": str(source_root / "mirroropd/integrations/verl/reward.py"),
                "name": "compute_score",
            },
        },
        "trainer": trainer,
        "ray_kwargs": {
            "ray_init": {
                "address": "local",
                "num_gpus": len(resources.gpu_ids),
                "num_cpus": resources.cpus,
                "include_dashboard": False,
                "object_store_memory": resources.object_store_bytes,
                "runtime_env": {"env_vars": environment},
            }
        },
    }


def compile_backend(job, evidence, run_dir, source_root, environment):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    source_root, run_dir = Path(source_root), Path(run_dir)
    with initialize_config_dir(
        config_dir=str((source_root / "verl/trainer/config").resolve()), version_base=None
    ):
        base = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(base, False)
    config = OmegaConf.merge(base, backend_values(job, evidence, run_dir, source_root, environment))
    OmegaConf.resolve(config)
    destination = run_dir / "backend.yaml"
    OmegaConf.save(config, destination, resolve=True)
    return destination
