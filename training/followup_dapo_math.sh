#!/usr/bin/env bash
# Serialize math baselines after the existing 1000-outer GPU job.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_python="$HOME/src/mirroropd-taopd/.venv/bin/python"
parent_unit=mirroropd-gsm8k-opd-o1000.service
while systemctl --user is-active --quiet "$parent_unit"; do
  sleep 30
done
parent_run="$repo_root/runs/gsm8k-smollm2-opd-onpolicy-o1000-j1"
if [[ ! -f "$parent_run/exit-code.txt" ]] || [[ $(<"$parent_run/exit-code.txt") != 0 ]]; then
  echo "Parent training did not finish successfully; refusing concurrent or failed follow-up" >&2
  exit 1
fi
export PYTHONPATH="$repo_root/src:$repo_root/upstream/One-Shot-OPD/verl:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
for pair in student:360M teacher:1.7B; do
  label=${pair%%:*}
  size=${pair#*:}
  output="$repo_root/runs/dapo-context-baselines/$label"
  mkdir -p "$output"
  "$task_python" "$repo_root/experiments/scoring/math_context_probe.py" \
    --model "HuggingFaceTB/SmolLM2-$size-Instruct" \
    --data "$repo_root/data/dapo-suite/dapo-math-17k/train.parquet" \
    --context 8192 --output "$output" >"$output/console.log" 2>&1
  mkdir -p "$output/sampled"
  "$task_python" "$repo_root/experiments/scoring/math_context_probe.py" \
    --model "HuggingFaceTB/SmolLM2-$size-Instruct" \
    --data "$repo_root/data/dapo-suite/dapo-math-17k/train.parquet" \
    --context 8192 --temperature 1 --limit 1024 \
    --output "$output/sampled" >"$output/sampled/console.log" 2>&1
done
# Context selection is based on completed full-train probes, not test scores.
"$task_python" "$repo_root/training/select_dapo_context.py" --repo "$repo_root"
response_budget=$("$task_python" -c 'import json,sys; v=json.load(open(sys.argv[1]))["max_response_length"]; print(v or 0)' \
  "$repo_root/runs/dapo-context-baselines/context-selection.json")
budget_args=()
if [[ "$response_budget" != 0 ]]; then
  budget_args=(--response-budget "$response_budget")
else
  echo "No fixed budget passes; tests use native context and training remains blocked" >&2
fi
for pair in student:360M teacher:1.7B; do
  label=${pair%%:*}
  size=${pair#*:}
  output="$repo_root/runs/dapo-context-baselines/$label/tests"
  mkdir -p "$output"
  "$task_python" "$repo_root/experiments/scoring/math_context_probe.py" \
    --model "HuggingFaceTB/SmolLM2-$size-Instruct" \
    --data "$repo_root/data/dapo-suite/math-500-easy/test.parquet" \
    "$repo_root/data/gsm8k-full/test.parquet" \
    --context 8192 "${budget_args[@]}" \
    --output "$output" >"$output/console.log" 2>&1
done
# Bring only the official verifier into the existing backend after its GPU job
# has finished. No trainer, optimizer, or live environment is replaced.
if [[ "$response_budget" == 0 ]]; then
  exit 1
fi
verifier_source="$repo_root/upstream/One-Shot-OPD/verl/verl/utils/reward_score"
backend="$HOME/src/mirroropd-taopd"
verifier_target="$backend/verl/verl/utils/reward_score"
backup="$repo_root/runs/dapo-context-baselines/backend-verifier-before"
mkdir -p "$backup"
for file in _timeout.py math_verify.py ttrl_mathverify.py; do
  if [[ -e "$verifier_target/$file" ]] && ! cmp -s "$verifier_source/$file" "$verifier_target/$file"; then
    cp "$verifier_target/$file" "$backup/$file"
    sha256sum "$backup/$file" >>"$backup/sha256.txt"
  fi
done
if [[ -d "$verifier_target/ttrl_math" ]]; then
  echo "Existing ttrl_math directory requires comparison before integration" >&2
  exit 1
fi
cp "$verifier_source/_timeout.py" "$verifier_source/math_verify.py" \
  "$verifier_source/ttrl_mathverify.py" "$verifier_target/"
cp -r "$verifier_source/ttrl_math" "$verifier_target/"
export PYTHONPATH="$repo_root/src:$backend/verl:$repo_root"
"$task_python" "$repo_root/tests/test_math_reward_protocol.py"
export MIRROR_SETTING_FILE="$repo_root/training/configs/gsm8k-opd-onpolicy.json"
export MIRROR_TEXT_DATA="$repo_root/data/dapo-suite/dapo-math-17k-clean"
export MIRROR_TEST_FILE="$repo_root/data/dapo-suite/simple-tests/test.parquet"
export MIRROR_REWARD_SCRIPT="$repo_root/training/math_reward.py"
export MIRROR_STUDENT_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct
export MIRROR_TEACHER_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
export MIRROR_METHOD=mirror_opd_onpolicy
export MIRROR_AUDIT_NATIVE_PAIR=1
export MIRROR_ENABLE_WANDB=1
export MIRROR_INFERENCE_EXPLICIT_OFFLOAD=1
export WANDB_MODE=online
export WANDB_RUN_GROUP=dapo-math-context-calibrated
prompt_budget=$("$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_prompt_length"])' \
  "$repo_root/runs/dapo-context-baselines/context-selection.json")
context=$((prompt_budget + response_budget))
common=("data.max_prompt_length=$prompt_budget" "data.max_response_length=$response_budget"
  "actor_rollout_ref.rollout.max_model_len=$context" "trainer.project_name=mirror-opd-dapo")
bash "$repo_root/training/run_configured_training.sh" dapo-smollm2-opd-memory-canary \
  "${common[@]}" trainer.total_training_steps=2 trainer.total_epochs=1 \
  trainer.val_before_train=false trainer.test_freq=-1
bash "$repo_root/training/run_configured_training.sh" dapo-smollm2-opd-onpolicy-o1000-j1 \
  "${common[@]}" trainer.total_training_steps=1000 trainer.total_epochs=4
