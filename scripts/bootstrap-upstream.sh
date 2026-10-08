#!/usr/bin/env bash
# Reconstruct audited upstream checkouts without overwriting local work.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p upstream
while IFS=$'\t' read -r name url revision patch module_paths; do
  checkout="upstream/$name"
  if [[ ! -d "$checkout/.git" ]]; then
    if [[ -e "$checkout" ]]; then
      echo "Refusing to replace existing non-Git path: $checkout" >&2
      exit 1
    fi
    git clone "$url" "$checkout"
    git -C "$checkout" checkout --detach "$revision"
  fi
  actual_url=$(git -C "$checkout" remote get-url origin)
  if [[ "$actual_url" != "$url" ]]; then
    echo "Unexpected origin for $checkout: $actual_url" >&2
    exit 1
  fi
  if [[ "$module_paths" != "-" ]]; then
    IFS=',' read -r -a modules <<<"$module_paths"
    git -C "$checkout" submodule update --init -- "${modules[@]}"
  fi
  if [[ "$patch" != "-" ]]; then
    if git -C "$checkout" apply --reverse --check "$repo_root/$patch" 2>/dev/null; then
      echo "$name: patch already present"
    elif git -C "$checkout" apply --check "$repo_root/$patch"; then
      git -C "$checkout" apply "$repo_root/$patch"
      echo "$name: applied $patch"
    else
      echo "Patch conflicts with existing $checkout; local work preserved" >&2
      exit 1
    fi
  fi
done < <(
  python3 - <<'PY'
import json
from pathlib import Path

for item in json.loads(Path("upstream-manifest.json").read_text())["checkouts"]:
    modules = ",".join(item.get("submodules", [])) or "-"
    print("\t".join([item["name"], item["url"], item["base_commit"], item["patch"] or "-", modules]))
PY
)
