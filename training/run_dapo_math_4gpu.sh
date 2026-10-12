#!/usr/bin/env bash
# Shared native DeepSeek math setting; physical GPUs 0--3 only.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd -- "$repo_root/.." && pwd)
requested_backend=${MIRROR_UPSTREAM_ROOT:-}
source "$workspace_root/.bashrc"
# The workspace shell sets a default backend. Honor the caller's isolated checkout.
if [[ -n "$requested_backend" ]]; then export MIRROR_UPSTREAM_ROOT="$requested_backend"; fi
fht-mirror
"$PYTHON_BIN" "$repo_root/training/check_math_evaluation_backend.py" \
  "${MIRROR_UPSTREAM_ROOT:-$HOME/src/mirroropd-taopd}"
"$PYTHON_BIN" "$repo_root/training/prepare_math_backend.py"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MIRROR_MONITOR_GPU_IDS=0,1,2,3
export RAY_ADDRESS=local
export MIRROR_RAY_TMPDIR="$repo_root/runs/math-4gpu-runtime/ray"
export MIRROR_TMPDIR="$repo_root/runs/math-4gpu-runtime/tmp"
mkdir -p "$MIRROR_RAY_TMPDIR" "$MIRROR_TMPDIR"
# Ray invokes the engine from a background thread. vLLM 0.10 otherwise
# auto-falls back to V0, whereas the deployed server uses V1 AsyncLLM.
export VLLM_USE_V1=1
export MIRROR_SETTING_FILE="$repo_root/training/configs/dapo-deepseek-4gpu.json"
export MIRROR_STUDENT_MODEL="$workspace_root/models/DeepSeek-R1-Distill-Qwen-1.5B"
export MIRROR_TEACHER_MODEL="$workspace_root/models/JustRL-DeepSeek-1.5B"
export MIRROR_REWARD_SCRIPT="$repo_root/training/math_reward.py"
export MIRROR_AUDIT_NATIVE_PAIR=1
export MIRROR_INFERENCE_EXPLICIT_OFFLOAD=1
export MIRROR_ENABLE_WANDB=1
export WANDB_MODE=online
export WANDB_RUN_GROUP=dapo-deepseek-4gpu-o50
unset WANDB_PROJECT
method=${1:?Usage: run_dapo_math_4gpu.sh METHOD RUN_NAME [overrides...]}
shift
export MIRROR_METHOD="$method"
export MIRROR_TEXT_DATA="$repo_root/runs/math-4gpu-input"
export MIRROR_TEST_FILE="$workspace_root/MD-OPD/data/math/MATH-500/test.parquet"
mkdir -p "$MIRROR_TEXT_DATA"
ln -sfn "$workspace_root/MD-OPD/data/math/DAPO/train.parquet" "$MIRROR_TEXT_DATA/train.parquet"
if [[ ! -f "$MIRROR_TEXT_DATA/manifest.json" ]]; then
  "$PYTHON_BIN" - "$MIRROR_TEXT_DATA/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"source": "MD-OPD math recipe", "model_paths": {}}) + "\n")
PY
fi
run_name=${1:?Missing run name}
shift
epochs=16
if [[ "$method" == mirror_opd_onpolicy ]]; then epochs=1; fi
exec bash "$repo_root/training/run_configured_training.sh" "$run_name" \
  "data.val_files=[$MIRROR_TEST_FILE,$workspace_root/MD-OPD/data/math/AMC23/test.parquet,$workspace_root/MD-OPD/data/math/AIME25/test.parquet]" \
  "actor_rollout_ref.actor.ppo_epochs=$epochs" "$@"
