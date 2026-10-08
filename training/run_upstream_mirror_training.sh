#!/usr/bin/env bash
# Launch patched upstream TA-OPD/verl with caller-provided datasets.
# Defaults are the validated 12 GB configuration; extend with Hydra overrides.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mirror_src=${MIRROROPD_SRC_ROOT:-"$repo_root/src"}
if [[ ! -f "$mirror_src/mirroropd/training/losses.py" ]]; then
  echo "MirrorOPD production package missing: $mirror_src; deploy the repository or set MIRROROPD_SRC_ROOT" >&2
  exit 2
fi

task_root=${MIRROR_UPSTREAM_ROOT:-"$HOME/src/mirroropd-taopd"}
mode=${1:-mirror_bpo}
if (($#)); then shift; fi
case "$mode" in
mirror_pmd_mean | mirror_bpo | mirror_mdpo | mirror_opd_onpolicy | mirror_opd_is | mirror_opd_is_no_baseline) ;;
*)
  echo "Expected a registered MirrorOPD loss mode" >&2
  exit 2
  ;;
esac
task_python="$task_root/.venv/bin/python"
mkdir -p "$task_root/data" "$task_root/logs" "$task_root/scratch"
export MIRROR_INNER_METRICS_PATH="${MIRROR_INNER_METRICS_PATH:-$task_root/logs/$mode-inner-$(date +%Y%m%d-%H%M%S).jsonl}"
export TMPDIR="$task_root/scratch"
# Ray creates long session/socket names; keep the disk-backed prefix short.
export RAY_TMPDIR="${MIRROR_RAY_TMPDIR:-$HOME/.cache/mopd-ray}"
mkdir -p "$RAY_TMPDIR"
export PYTHONPATH="$mirror_src:$task_root/verl${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-$repo_root/runs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/.cache/wandb}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

if (($# < 2)); then
  echo "Usage: $0 MODE TRAIN_PARQUET VAL_PARQUET [Hydra overrides...]" >&2
  exit 2
fi
train_file=$1
val_file=$2
shift 2
for dataset_file in "$train_file" "$val_file"; do
  if [[ ! -f "$dataset_file" ]]; then
    echo "Missing dataset: $dataset_file" >&2
    exit 2
  fi
done
# Resolve inputs before changing to the upstream project directory.
train_file="$(cd -- "$(dirname -- "$train_file")" && pwd)/$(basename -- "$train_file")"
val_file="$(cd -- "$(dirname -- "$val_file")" && pwd)/$(basename -- "$val_file")"

# Reuse upstream GSM8K scoring for monitoring without the fork's DeepScaler dependency.
cat >"$task_root/data/gsm8k_reward.py" <<'PY'
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return gsm8k_score(solution_str, ground_truth)
PY

cd "$task_root/verl"
"$task_python" -m verl.trainer.main_ppo \
  data.train_files="$train_file" \
  data.val_files="$val_file" \
  data.train_batch_size=2 data.max_prompt_length=64 data.max_response_length=64 \
  data.dataloader_num_workers=0 \
  reward.custom_reward_function.path="$task_root/data/gsm8k_reward.py" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers=1 \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.use_fused_kernels=false \
  actor_rollout_ref.model.lora_rank=8 actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=2 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model]' \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=256 \
  actor_rollout_ref.rollout.max_model_len=128 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.rollout.top_p=1 \
  actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.enforce_eager=true \
  distillation.enabled=true \
  distillation.n_gpus_per_node=0 distillation.nnodes=0 \
  distillation.mirror_teacher_model_path=Qwen/Qwen3-1.7B \
  distillation.distillation_loss.loss_mode="$mode" \
  distillation.distillation_loss.topk=16 \
  distillation.distillation_loss.mirror_eta=0.1 \
  distillation.distillation_loss.use_task_rewards=false \
  distillation.distillation_loss.use_policy_gradient=false \
  distillation.distillation_loss.loss_max_clamp=null \
  distillation.distillation_loss.log_prob_min_clamp=null \
  algorithm.adv_estimator=reinforce_plus_plus \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.total_training_steps=2 trainer.total_epochs=1 \
  trainer.val_before_train=false trainer.test_freq=-1 \
  trainer.save_freq=-1 trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.default_local_dir="$task_root/checkpoints/$mode" \
  trainer.project_name=mirror-opd trainer.experiment_name="$mode" \
  'trainer.logger=[console]' \
  +ray_kwargs.ray_init.object_store_memory=536870912 \
  ray_kwargs.ray_init.num_cpus=4 \
  +ray_kwargs.ray_init.runtime_env.env_vars.MIRROR_INNER_METRICS_PATH="$MIRROR_INNER_METRICS_PATH" \
  "$@"
