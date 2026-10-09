#!/usr/bin/env bash
# Queue the authorized OPD baseline after its canary passes.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
first="$repo_root/runs/gsm8k-smollm2-opd-canary"
while systemctl --user is-active --quiet mirroropd-gsm8k-opd-canary.service; do
  sleep 30
done
if [[ ! -f "$first/exit-code.txt" ]] || [[ $(cat "$first/exit-code.txt") != 0 ]]; then
  echo "OPD canary failed; retain logs and investigate before long training" >&2
  exit 1
fi
"$HOME/src/mirroropd-taopd/.venv/bin/python" - "$first" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "inner.jsonl").read_text().splitlines()]
assert len(rows) == 2
assert [r["outer_call"] for r in rows] == [0, 1]
for row in rows:
    assert row["epochs"] == 1 and row["inner_step"] == 0
    metrics = row["metrics"]
    assert all(math.isfinite(v) for v in metrics.values())
    assert abs(metrics["mirror/student_center/mean"]) < 1e-6
    assert abs(metrics["mirror/importance_ratio/mean"] - 1) < 1e-6
    assert metrics["mirror/teacher_center/mean"] > 0
outer = [json.loads(line) for line in (root / "outer.jsonl").read_text().splitlines()]
assert [r["step"] for r in outer] == [0, 1, 2]
assert sum("evaluation/generated_examples" in r["data"] for r in outer) == 2
print("PASS OPD canary: baseline, fresh policy, two updates/syncs and tests")
PY
exec bash "$repo_root/training/run_gsm8k_opd_onpolicy.sh" gsm8k-smollm2-opd-onpolicy-o100-j1
