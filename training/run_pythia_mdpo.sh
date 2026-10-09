#!/usr/bin/env bash
# Preserve the historical Pythia pilot entry point.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/run_configured_training.sh" "$@"
