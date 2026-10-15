"""Validate the patched native verl outer/inner path on physical GPUs 0--3."""

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from training.gpu_exclusive import gpu_processes
from training.oneshot_suite import snapshot_sources
from training.power_implementation_suite import acquire_lease


def command(root, method, estimator, output):
    settings = [
        f"actor_rollout_ref.model.path={root.parent / 'models/DeepSeek-R1-Distill-Qwen-1.5B'}",
        f"distillation.mirror_teacher_model_path={root.parent / 'models/JustRL-DeepSeek-1.5B'}",
        "actor_rollout_ref.model.lora_rank=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.custom_chat_template=null",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
        "+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32}",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.15",
        "actor_rollout_ref.rollout.max_model_len=4096",
        "actor_rollout_ref.rollout.max_num_batched_tokens=4096",
        "actor_rollout_ref.rollout.max_num_seqs=2",
        "distillation.distillation_loss.mirror_kappa=0.5",
        "distillation.distillation_loss.mirror_eta=0.3",
        f"distillation.distillation_loss.mirror_center_mode={estimator}",
        "distillation.distillation_loss.mirror_action_mode="
        + ("full_vocab" if estimator == "full_vocab" else "sampled"),
        "data.train_batch_size=4",
        "data.max_prompt_length=2048",
        "data.max_response_length=32",
        "data.seed=42",
        "trainer.n_gpus_per_node=4",
        "trainer.total_training_steps=2",
        f"trainer.default_local_dir={output / 'checkpoints'}",
        f"trainer.experiment_name={output.name}",
        "+ray_kwargs.ray_init.address=local",
        "+ray_kwargs.ray_init.num_gpus=4",
        "ray_kwargs.ray_init.num_cpus=16",
        "+ray_kwargs.ray_init.include_dashboard=false",
        "ray_kwargs.ray_init.object_store_memory=4294967296",
        '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"',
    ]
    dataset = str(root.parent / "MD-OPD/data/math/DAPO-oneshot/t22.parquet")
    return [
        "bash",
        str(root / "training/run_upstream_mirror_training.sh"),
        "mirror_" + method,
        dataset,
        dataset,
        *settings,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bpo", "mdpo"), default="bpo")
    parser.add_argument("--estimator", choices=("topj_tail", "full_vocab"), default="full_vocab")
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--dry-config", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = root / "runs/power-implementation-20261015"
    output = suite / f"verl-{args.method}-{args.estimator}"
    output.mkdir(parents=True, exist_ok=True)
    if not args.dry_config and (output / "console.log").exists():
        history = output / "attempts"
        history.mkdir(exist_ok=True)
        archived = history / f"attempt-{len(list(history.iterdir())) + 1:03d}"
        archived.mkdir()
        for name in (
            "command.json",
            "console.log",
            "exit.json",
            "resolved-config.yaml",
            "config.stderr",
            "inner.jsonl",
        ):
            if (output / name).exists():
                shutil.move(output / name, archived / name)
    argv = command(root, args.method, args.estimator, output)
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": "" if args.dry_config else "0,1,2,3",
        "MIRROR_UPSTREAM_ROOT": str(args.backend.resolve()),
        "PYTHON_BIN": str(root.parent / "envs/opd/bin/python"),
        "MIRROR_INNER_METRICS_PATH": str(output / "inner.jsonl"),
        "MIRROR_RAY_TMPDIR": str(Path.home() / ".cache/power-verl-ray"),
        "MIRROR_TMPDIR": str(Path.home() / ".cache/power-verl-tmp"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "WANDB_MODE": "disabled",
        "VLLM_USE_V1": "1",
    }
    (output / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    if args.dry_config:
        with (
            (output / "resolved-config.yaml").open("w") as stream,
            (output / "config.stderr").open("w") as errors,
        ):
            result = subprocess.run(  # noqa: S603 -- fixed local entry and validated enum arguments
                [*argv, "--cfg", "job", "--resolve"],
                env=environment,
                stdout=stream,
                stderr=errors,
                check=False,
            )
        raise SystemExit(result.returncode)
    observed = gpu_processes()
    if any(gpu["pids"] or gpu["memory_mib"] > 16 for gpu in observed.values()):
        raise RuntimeError("Native validation requires idle physical GPUs 0--3")
    lease = acquire_lease(root, suite)
    try:
        snapshot_sources(root, output / "source")
        environment["MIRROROPD_SRC_ROOT"] = str(output / "source/src")
        patch = subprocess.check_output(
            ["/usr/bin/git", "diff", "--binary", "62669afb5101bfdf57dc5ddf81e78f000c5ce2fc"],
            cwd=args.backend,
        )
        (output / "backend.patch").write_bytes(patch)
        with (output / "console.log").open("w") as stream:
            result = subprocess.run(  # noqa: S603 -- fixed local entry and validated enum arguments
                argv, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        (output / "exit.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
        if result.returncode:
            raise RuntimeError(f"Native verl validation failed; inspect {output}")
        rows = [json.loads(line) for line in (output / "inner.jsonl").read_text().splitlines()]
        if len(rows) != 4 or sorted({row["outer_call"] for row in rows}) != [0, 1]:
            raise RuntimeError(
                "Native validation did not produce two outer windows and four updates"
            )
        for row in rows:
            if not all(
                math.isfinite(value)
                for value in row["metrics"].values()
                if isinstance(value, (float, int))
            ):
                raise RuntimeError("Native inner diagnostics are nonfinite")
        (output / "completion.jsonl").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "phase": "verl",
                    "method": args.method,
                    "estimator": args.estimator,
                    "outer_windows": 2,
                    "optimizer_updates": len(rows),
                    "response_budget": 32,
                    "scope": "Short fresh-rollout integration check, not mathematical task evaluation",
                }
            )
            + "\n"
        )
    finally:
        lease.release()


if __name__ == "__main__":
    main()
