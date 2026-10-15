"""Compile against the pinned YAML and execute its actual optimizer/scheduler."""

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_native_contract import native_function

from mirroropd.integrations.verl.compiler import compile_backend
from mirroropd.runtime.config import Job
from mirroropd.runtime.runner import public_environment


@pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW"])
def test_pinned_native_config_optimizer_and_outer_scheduler(tmp_path, optimizer_name):
    backend = Path(os.environ.get("MIRROR_TEST_BACKEND", "upstream/TA-OPD-v2")).resolve()
    source = tmp_path / "source"
    source.mkdir()
    (source / "verl").symlink_to(backend / "verl/verl", target_is_directory=True)
    job = Job.from_dict(
        {
            "name": "native-config",
            "models": {"student": "student", "teacher": "teacher"},
            "data": {"train": ["train"], "validation": ["val"]},
            "training": {
                "outer_steps": 4,
                "inner_steps": 3,
                "learning_rate": 0.01,
                "optim": {
                    "optimizer": optimizer_name,
                    "betas": [0.8, 0.95],
                    "eps": 1e-7,
                    "lr_scheduler_type": "cosine",
                    "min_lr_ratio": 0.1,
                },
            },
            "checkpoints": {"every": 2, "max_to_keep": 3, "save_contents": ["model", "hf_model"]},
        }
    )
    evidence = {
        "dataset": {"splits": {"train": {"paths": ["train"]}, "validation": {"paths": ["val"]}}},
        "models": {"teacher": {"identity": "teacher"}},
        "vocabulary_id": "tokens",
        "vocab_size": 32,
        "protocol": {},
        "evaluation_questions": 4,
    }
    env = public_environment(job, tmp_path, tmp_path / "scratch")
    config = OmegaConf.load(compile_backend(job, evidence, tmp_path, source, env))
    optim = config.actor_rollout_ref.actor.optim
    assert config.trainer.max_actor_ckpt_to_keep == 3
    assert list(config.actor_rollout_ref.actor.checkpoint.save_contents) == ["model", "hf_model"]
    assert config.trainer.resume_mode == "disable"
    assert config.trainer.experiment_name == env["WANDB_NAME"]
    assert config.actor_rollout_ref.actor.ppo_epochs == 3
    build = native_function(
        backend / "verl/verl/workers/config/optimizer.py", "build_optimizer", {}
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = build([parameter], optim)
    assert type(optimizer).__name__ == optimizer_name
    assert optimizer.defaults["betas"] == pytest.approx([0.8, 0.95])
    assert optimizer.defaults["eps"] == 1e-7
    # The trainer overwrites total_training_steps with the outer budget. The
    # worker advances only at the final inner iteration of each outer.
    optim.total_training_steps = job.training.outer_steps
    build_scheduler = native_function(
        backend / "verl/verl/workers/engine/fsdp/transformer_impl.py", "_build_lr_scheduler", {}
    )
    scheduler = build_scheduler(SimpleNamespace(rank=1, optimizer_config=optim), optimizer)
    rates = []
    for outer in range(4):
        rate = optimizer.param_groups[0]["lr"]
        rates.append(rate)
        for _inner in range(3):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            assert optimizer.param_groups[0]["lr"] == rate
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            0.01 * (0.1 + 0.9 * (1 + math.cos(math.pi * (outer + 1) / 4)) / 2)
        )
    assert rates[0] == pytest.approx(0.01)
    native = (backend / "verl/verl/workers/engine_workers.py").read_text()
    assert "update_lr_scheduler=batch_idx == total_num_iterations - 1" in native
