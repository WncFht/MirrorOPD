"""配置本仓已有的 hooks 与技能入口，不覆盖外来 hooksPath。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def setup(root: Path) -> None:
    root = root.resolve()
    old = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if old.returncode not in {0, 1}:
        raise ValueError("无法读取 core.hooksPath")
    if old.returncode == 0:
        hooks = Path(old.stdout.strip()).expanduser()
        if not hooks.is_absolute():
            hooks = root / hooks
        if hooks.resolve() != root / ".githooks":
            raise ValueError("已有外部 core.hooksPath，拒绝覆盖；请合并本仓 hooks")
    for name in ("pre-commit", "pre-push"):
        hook = root / ".githooks" / name
        if not hook.is_file() or not hook.stat().st_mode & 0o111:
            raise ValueError("缺少可执行 hook：" + str(hook))
    subprocess.run(["bash", "scripts/agent-links.sh"], cwd=root, check=True)
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "--local", "mirroropd.maintenancePython", sys.executable],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "blame.ignoreRevsFile", ".git-blame-ignore-revs"],
        cwd=root,
        check=True,
    )
    print("已配置 .githooks/pre-commit、pre-push 和仓库技能入口")


if __name__ == "__main__":
    try:
        setup(Path(__file__).resolve().parents[1])
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
