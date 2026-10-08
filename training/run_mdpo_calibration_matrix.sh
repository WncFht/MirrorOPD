#!/usr/bin/env bash
# Run the MDPO setting calibration matrix on the disk-backed archbox workspace.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

run_one() {
  local name=$1
  local group=$2
  shift 2
  echo "[$(date -Is)] start $name"
  WANDB_MODE=online WANDB_RUN_GROUP="$group" \
    bash "$script_dir/run_pythia_mdpo.sh" "$name" \
    trainer.logger='[console,file,wandb]' \
    trainer.total_training_steps=1 trainer.total_epochs=1 \
    trainer.save_freq=-1 trainer.test_freq=-1 \
    actor_rollout_ref.actor.checkpoint.save_contents='[model]' \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    "$@"
  echo "[$(date -Is)] done $name"
}

# Stage A: memory/throughput. The whole rollout batch stays B; micro-batch only
# changes gradient accumulation and therefore must not change the objective.
if [[ ${MIRROR_CALIBRATION_STAGE:-all} == all || ${MIRROR_CALIBRATION_STAGE} == resource ]]; then
  for spec in \
    "b16-m1 16 1" "b16-m2 16 2" "b16-m4 16 4" \
    "b32-m1 32 1" "b32-m2 32 2" "b32-m4 32 4" \
    "b64-m1 64 1" "b64-m2 64 2"; do
    read -r label batch micro <<<"$spec"
    run_one "mdpo-resource-$label" "mdpo-calibration/resource" \
      data.train_batch_size="$batch" data.val_batch_size="$batch" \
      actor_rollout_ref.actor.ppo_mini_batch_size="$batch" \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$micro" \
      actor_rollout_ref.actor.ppo_epochs=4 \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      distillation.distillation_loss.mirror_eta=1.0
  done
fi

# Stage B: fixed one-rollout inner dynamics. Use B=32 after the resource probe.
if [[ ${MIRROR_CALIBRATION_STAGE:-all} == all || ${MIRROR_CALIBRATION_STAGE} == inner ]]; then
  for spec in \
    "j1-lr1e-7 1 1e-7" "j4-lr1e-7 4 1e-7" "j16-lr1e-7 16 1e-7" \
    "j1-lr3e-7 1 3e-7" "j4-lr3e-7 4 3e-7" "j16-lr3e-7 16 3e-7" \
    "j1-lr1e-6 1 1e-6" "j4-lr1e-6 4 1e-6" "j16-lr1e-6 16 1e-6"; do
    read -r label epochs lr <<<"$spec"
    run_one "mdpo-inner-$label" "mdpo-calibration/inner" \
      data.train_batch_size=32 data.val_batch_size=32 \
      actor_rollout_ref.actor.ppo_mini_batch_size=32 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.actor.ppo_epochs="$epochs" \
      actor_rollout_ref.actor.optim.lr="$lr" \
      distillation.distillation_loss.mirror_eta=1.0
  done
fi

# Stage C: a small eta check at the chosen J/lr, still one outer rollout.
if [[ ${MIRROR_CALIBRATION_STAGE:-all} == all || ${MIRROR_CALIBRATION_STAGE} == eta ]]; then
  for eta in 0.1 0.3 1.0; do
    safe_eta=${eta//./p}
    run_one "mdpo-eta-$safe_eta" "mdpo-calibration/eta" \
      data.train_batch_size=32 data.val_batch_size=32 \
      actor_rollout_ref.actor.ppo_mini_batch_size=32 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.actor.ppo_epochs=4 \
      actor_rollout_ref.actor.optim.lr=3e-7 \
      distillation.distillation_loss.mirror_eta="$eta"
  done
fi

echo "calibration matrix complete: $repo_root/runs"
