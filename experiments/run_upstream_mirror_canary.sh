#!/usr/bin/env bash
# Prepare four execution-check prompts, then use the shared training launcher.
set -euo pipefail

task_root=${MIRROR_UPSTREAM_ROOT:-"$HOME/src/mirroropd-taopd"}
mode=${1:-mirror_bpo}
if (($#)); then shift; fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$task_root/data"
export TMPDIR="$task_root/scratch"
mkdir -p "$TMPDIR"
task_python="$task_root/.venv/bin/python"

# This dataset checks execution, not GSM8K accuracy or convergence.
"$task_python" - "$task_root/data/canary.parquet" <<'PY'
import sys
import pandas as pd

questions = [
    ("What is 2 plus 3?", "5"),
    ("What is 7 minus 4?", "3"),
    ("What is 3 times 4?", "12"),
    ("What is 10 divided by 2?", "5"),
]
pd.DataFrame([
    {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": question + " Put the answer after ####."}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"index": index, "split": "canary"},
    }
    for index, (question, answer) in enumerate(questions)
]).to_parquet(sys.argv[1], index=False)
PY

exec bash "$script_dir/run_upstream_mirror_training.sh" "$mode" \
  "$task_root/data/canary.parquet" "$task_root/data/canary.parquet" "$@"
