#!/usr/bin/env bash
# Continue the fixed Pythia eta scan sequentially on one GPU.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for label in inf 0p3 0p03; do
  case "$label" in
  inf) eta=1e309 ;;
  0p3) eta=0.3 ;;
  0p03) eta=0.03 ;;
  esac
  run_name="pythia-mdpo-eta-$label"
  bash "$script_dir/run_pythia_mdpo.sh" "$run_name" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    "distillation.distillation_loss.mirror_eta=$eta"
  bash "$script_dir/score_pythia_mdpo.sh" "$run_name"
done
