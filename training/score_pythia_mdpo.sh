#!/usr/bin/env bash
# Evaluate identical held-out prefixes before training and at every saved checkpoint.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
task_root=${MIRROR_UPSTREAM_ROOT:-"$HOME/src/mirroropd-taopd"}
data_dir=${MIRROR_TEXT_DATA:-"$repo_root/data/pythia-wikitext"}
run_dir="$repo_root/runs/${1:-pythia-mdpo-pilot}"
task_python="$task_root/.venv/bin/python"
[[ $(cat "$run_dir/exit-code.txt") == 0 ]]
if [[ ! -d "$run_dir/checkpoints" ]] || ! find "$run_dir/checkpoints" \
  -name model_world_size_1_rank_0.pt -print -quit | grep -q .; then
  echo "No saved model to evaluate. Checkpoint-free runs retain training diagnostics only." >&2
  echo "For independent final evaluation, explicitly enable model-only saving with trainer.save_freq>0." >&2
  exit 2
fi
mapfile -t models < <(
  "$task_python" - "$data_dir/manifest.json" <<'PY'
import json
import sys
from pathlib import Path
paths = json.loads(Path(sys.argv[1]).read_text())["model_paths"]
print(paths["student"])
print(paths["teacher"])
PY
)
steps=(0)
while IFS= read -r checkpoint_dir; do
  step=${checkpoint_dir##*_}
  [[ $step =~ ^[0-9]+$ ]] && steps+=("$step")
done < <(find "$run_dir/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort -V)

for step in "${steps[@]}"; do
  output="$run_dir/evaluation/step-$step"
  mkdir -p "$output"
  checkpoint_args=()
  if ((step > 0)); then
    checkpoint_args=(--checkpoint "$run_dir/checkpoints/global_step_$step/actor")
  fi
  "$task_python" "$repo_root/experiments/scoring/pythia_checkpoint_diagnostics.py" \
    --dataset "$data_dir/validation-prefixes.json" \
    --student-model "${models[0]}" --teacher-model "${models[1]}" \
    --output "$output" --prompt-prefixes all --dtype float32 \
    "${checkpoint_args[@]}" >"$output/console.log" 2>&1
  echo "Scored checkpoint $step"
done
