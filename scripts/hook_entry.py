"""Run MirrorOPD's owned hook checks after formatting the real index."""

import argparse
import subprocess
import sys
from pathlib import Path

from install_hooks import format_staged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pre-commit", "pre-push"])
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    if args.stage == "pre-commit":
        format_staged(root)
        changed = (
            subprocess.check_output(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM", "-z"], cwd=root
            )
            .decode()
            .split("\0")
        )
        for name in filter(None, changed):
            size = int(subprocess.check_output(["git", "cat-file", "-s", ":" + name], cwd=root))
            if size > 20 * 1024**2:
                raise SystemExit(f"Artifact exceeds 20 MiB: {name}")
        checker = root / ".crossnote/scripts/check-pseudocode.cjs"
        if checker.is_file():
            subprocess.run(["node", str(checker), "--staged"], cwd=root, check=True)
    command = [sys.executable, "-m", "pre_commit", "run", "--hook-stage", args.stage]
    if args.stage == "pre-push":
        command.append("--all-files")
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
