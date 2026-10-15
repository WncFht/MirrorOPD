#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${MIRROR_MODEL_ROOT:?Set MIRROR_MODEL_ROOT to the shared model directory}"
: "${MIRROR_DATA_ROOT:?Set MIRROR_DATA_ROOT to the shared dataset directory}"
: "${MIRROR_RUN_DIR:?Set MIRROR_RUN_DIR to a new shared output directory}"

MIRROR_PYTHON="${MIRROR_PYTHON:-$REPO_ROOT/.venv/bin/python}"
MIRROR_BACKEND="${MIRROR_BACKEND:-$REPO_ROOT/upstream/TA-OPD-v2}"
MIRROR_RECIPE="${MIRROR_RECIPE:-math-four-gpu}"
MIRROR_ALGORITHM="${MIRROR_ALGORITHM:-on_policy}"
MIRROR_WANDB_MODE="${MIRROR_WANDB_MODE:-offline}"
MIRROR_GPU_IDS="${MIRROR_GPU_IDS:-[0,1,2,3]}"

# Check the environment inside the allocated container, not on the submission host.
"$MIRROR_PYTHON" -c 'import torch, vllm, flash_attn; assert torch.__version__.split(".")[:2] == ["2", "7"], "MirrorOPD requires PyTorch 2.7; prepare its environment inside this image"; print("torch", torch.__version__, "vllm", vllm.__version__, flush=True)'

# The launcher stays in foreground so hope observes failures and cancellation.
exec "$MIRROR_PYTHON" -u -m mirroropd run \
  --recipe "$MIRROR_RECIPE" --models deepseek-justrl --dataset dapo-math \
  --var "model_root=$MIRROR_MODEL_ROOT" --var "data_root=$MIRROR_DATA_ROOT" \
  --algorithm "$MIRROR_ALGORITHM" --wandb "$MIRROR_WANDB_MODE" \
  --set "resources.gpu_ids=$MIRROR_GPU_IDS" \
  --backend "$MIRROR_BACKEND" --output "$MIRROR_RUN_DIR" "$@"
