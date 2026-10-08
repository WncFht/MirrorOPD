#!/usr/bin/env bash
set -euo pipefail
if (($# == 0)); then
  while IFS= read -r -d '' file; do
    head -1 "$file" | grep -q zsh || shellcheck -S warning "$file"
  done < <(git ls-files -z '*.sh' '.githooks/*')
else
  for file in "$@"; do
    head -1 "$file" | grep -q zsh || shellcheck -S warning "$file"
  done
fi
