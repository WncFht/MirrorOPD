#!/usr/bin/env bash
# Evaluate the teacher on matched prompts, then train a fresh student for 1000 updates.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_python="$HOME/src/mirroropd-taopd/.venv/bin/python"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
teacher_output="$repo_root/runs/gsm8k-teacher-native-2048"
mkdir -p "$teacher_output"
"$task_python" "$repo_root/experiments/scoring/gsm8k_native_test.py" \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --test "$repo_root/data/gsm8k-full/test.parquet" \
  --output "$teacher_output" >"$teacher_output/console.log" 2>&1
exec bash "$repo_root/training/run_gsm8k_opd_onpolicy.sh" \
  gsm8k-smollm2-opd-onpolicy-o1000-j1 \
  trainer.total_training_steps=1000 trainer.total_epochs=9
