#!/usr/bin/env bash
# Configured experiments through the shared TA-OPD/verl entry point.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
task_root=${MIRROR_UPSTREAM_ROOT:-"$HOME/src/mirroropd-taopd"}
data_dir=${MIRROR_TEXT_DATA:-"$repo_root/data/pythia-wikitext"}
test_file=${MIRROR_TEST_FILE:-"$data_dir/validation.parquet"}
run_name=${1:-pythia-mdpo-pilot}
if (($#)); then shift; fi
run_dir="$repo_root/runs/$run_name"
setting_file=${MIRROR_SETTING_FILE:-"$script_dir/configs/pythia-mdpo.json"}
reward_script=${MIRROR_REWARD_SCRIPT:-"$script_dir/natural_text_reward.py"}
if [[ -e "$run_dir" ]]; then
  echo "Run already exists: $run_dir; choose an unused run name" >&2
  exit 2
fi
mkdir -p "$run_dir"
method=${MIRROR_METHOD:-mirror_mdpo}
case "$method" in
mirror_pmd_mean | mirror_bpo | mirror_mdpo | mirror_opd_onpolicy | mirror_opd_is | mirror_opd_is_no_baseline) ;;
*)
  echo "Unsupported MIRROR_METHOD=$method" >&2
  exit 2
  ;;
esac
task_python=${PYTHON_BIN:-"$task_root/.venv/bin/python"}
export MIRROR_INNER_METRICS_PATH="$run_dir/inner.jsonl"
export VERL_FILE_LOGGER_PATH="$run_dir/outer.jsonl"
export MIRROR_WANDB_RUN_PATH="$run_dir/wandb-run.json"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-budget-calibration}"
if [[ "${MIRROR_ENABLE_WANDB:-0}" == "1" ]]; then
  logger_override='trainer.logger=[console,file,wandb]'
else
  logger_override='trainer.logger=[console,file]'
fi

# Only path placeholders are replaced; Jinja braces remain literal text.
# A failed producer inside process substitution does not fail mapfile. Write
# arguments first so malformed configuration aborts before the base run starts.
"$task_python" - "$setting_file" \
  "$data_dir/manifest.json" "$run_dir" "$run_name" "$reward_script" \
  >"$run_dir/setting-args.nul" <<'PY'
import json
import os
import sys
from pathlib import Path

setting = json.loads(Path(sys.argv[1]).read_text())
data = json.loads(Path(sys.argv[2]).read_text())
values = dict(student_model=os.environ.get("MIRROR_STUDENT_MODEL") or data["model_paths"]["student"],
              teacher_model=os.environ.get("MIRROR_TEACHER_MODEL") or data["model_paths"]["teacher"], run_dir=sys.argv[3],
              run_name=sys.argv[4], reward_script=sys.argv[5])
for argument in setting["hydra_overrides"]:
    for key, value in values.items():
        argument = argument.replace("{" + key + "}", value)
    sys.stdout.write(argument + "\0")
PY
mapfile -d '' -t setting_args <"$run_dir/setting-args.nul"
setting_args+=("$logger_override")
cp "$setting_file" "$run_dir/setting.json"
cp "$data_dir/manifest.json" "$run_dir/data-manifest.json"
printf '%s\n' "${setting_args[@]}" "$@" >"$run_dir/overrides.txt"
printf '%s\n' "$method" >"$run_dir/method.txt"
git -C "$repo_root" rev-parse HEAD >"$run_dir/source-commit.txt" 2>/dev/null || true
"$task_python" "$script_dir/record_training_provenance.py" --repo "$repo_root" \
  --upstream "$task_root" --output "$run_dir/source-provenance.json"
bash "$script_dir/run_upstream_mirror_training.sh" "$method" \
  "$data_dir/train.parquet" "$test_file" \
  "${setting_args[@]}" "$@" --cfg job --resolve >"$run_dir/resolved-config.yaml" 2>"$run_dir/config.stderr"
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$task_python" - "$run_dir/resolved-config.yaml" "$run_dir/evaluation-protocol.json" <<'PYCODE'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf
from mirroropd.training.evaluation import evaluation_protocol

config = OmegaConf.load(sys.argv[1])
Path(sys.argv[2]).write_text(json.dumps(evaluation_protocol(config), indent=2) + "\n")
PYCODE
if [[ "${MIRROR_AUDIT_NATIVE_PAIR:-0}" == "1" ]]; then
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$task_python" \
    "$script_dir/audit_gsm8k_setting.py" --config "$run_dir/resolved-config.yaml" \
    --output "$run_dir/model-data-audit.json"
fi
nvidia-smi ${MIRROR_MONITOR_GPU_IDS:+--id="$MIRROR_MONITOR_GPU_IDS"} --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv -lms 1000 >"$run_dir/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
set +e
bash "$script_dir/run_upstream_mirror_training.sh" "$method" \
  "$data_dir/train.parquet" "$test_file" \
  "${setting_args[@]}" "$@" \
  >"$run_dir/console.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$run_dir/exit-code.txt"
# Successful runs upload inner rows at every outer boundary from the driver.
# Keep the publisher as a recovery path for interrupted runs only; replaying a
# completed run would duplicate every W&B history row.
if [[ $status != 0 && -f "$run_dir/wandb-run.json" ]]; then
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" "$task_python" \
    "$script_dir/publish_wandb_diagnostics.py" --run "$run_dir" --include-outer \
    >"$run_dir/wandb-publish.log" 2>&1 || echo "W&B diagnostic publish failed; local JSONL retained" >&2
fi
echo "$method exit=$status; records: $run_dir"
exit "$status"
