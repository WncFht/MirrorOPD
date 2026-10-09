#!/usr/bin/env bash
# Native-template on-policy OPD with the KL baseline and one update per rollout.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
export MIRROR_TEXT_DATA="${MIRROR_TEXT_DATA:-$repo_root/data/gsm8k-full}"
export MIRROR_TEST_FILE="${MIRROR_TEST_FILE:-$MIRROR_TEXT_DATA/test.parquet}"
export MIRROR_SETTING_FILE="$script_dir/configs/gsm8k-opd-onpolicy.json"
export MIRROR_REWARD_SCRIPT="$script_dir/gsm8k_reward.py"
export MIRROR_STUDENT_MODEL="${MIRROR_STUDENT_MODEL:-HuggingFaceTB/SmolLM2-360M-Instruct}"
export MIRROR_TEACHER_MODEL="${MIRROR_TEACHER_MODEL:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
export MIRROR_METHOD=mirror_opd_onpolicy
export MIRROR_AUDIT_NATIVE_PAIR=1
export MIRROR_INFERENCE_EXPLICIT_OFFLOAD="${MIRROR_INFERENCE_EXPLICIT_OFFLOAD:-1}"
export MIRROR_ENABLE_WANDB="${MIRROR_ENABLE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gsm8k-baseline-opd}"
exec bash "$script_dir/run_configured_training.sh" "$@"
