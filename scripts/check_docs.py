"""验证维护包的技能结构与 Markdown 本地文件链接，不访问外网。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from markdown_it import MarkdownIt

SKILLS = {
    "maintain-docs",
    "maintain-prose",
    "maintain-decision-notes",
}
IGNORED = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".claude",
    ".crossnote",
    "upstream",
    "runs",
    "transfers",
    "imported",
    "build",
    "dist",
}
# 理论来源尚待 review/merge；第三方材料与冻结历史不参与当前文档检查。
IGNORED_PATHS = {
    "assets/.crossnote",
    ".venv-maintenance",
    ".tools",
    "data",
    "results",
    "paper-search-logs",
    "paper-search-reports",
    "theory",
    "docs/book",
    "docs/book-old",
    "docs/Notes",
    "docs/paper",
    "docs/papers",
    ".agents/notes/archived",
    "src/mirroropd/scoring/vendor",
}


def markdown_paths(root: Path):
    for folder, directories, names in os.walk(root, followlinks=False):
        parent = Path(folder)
        directories[:] = sorted(
            name
            for name in directories
            if name not in IGNORED
            and (parent / name).relative_to(root).as_posix() not in IGNORED_PATHS
            and not (parent / name).is_symlink()
        )
        for name in sorted(names):
            if name.endswith(".md"):
                yield parent / name


def deferred_documents(root: Path) -> dict[str, str]:
    """仅豁免接入时原样保留的旧文档；新增或修改内容自动进入检查。"""
    path = root / "scripts/maintenance-deferred-docs.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"version", "files"} or data["version"] != 1:
        raise ValueError("历史文档基线格式错误")
    if not isinstance(data["files"], dict):
        raise ValueError("历史文档基线 files 必须是对象")
    for name, digest in data["files"].items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2]
            not in {("docs", "operations"), ("docs", "experiments"), ("docs", "earlier")}
            or relative.suffix != ".md"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("非法历史文档基线项：" + str(name))
    return data["files"]


def check(root: Path) -> list[str]:
    errors = []
    try:
        deferred = deferred_documents(root)
    except (ValueError, TypeError, OSError) as exc:
        return [str(exc)]
    adoption = root / "scripts/maintenance-source.json"
    if adoption.is_file():
        try:
            adopted = json.loads(adoption.read_text())["files"]
            if not isinstance(adopted, list):
                raise ValueError("维护来源 files 必须是数组")
            for item in adopted:
                if item.get("status") == "not-copied":
                    continue
                name = item["target"]
                target = root / name
                if (
                    Path(name).is_absolute()
                    or not target.resolve().is_relative_to(root.resolve())
                    or not target.is_file()
                ):
                    errors.append(f"维护来源目标缺失或越界：{name}")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f"维护来源清单无效：{exc}")
    skills = root / ".agents/skills"
    found = {p.parent.name for p in skills.glob("*/SKILL.md")}
    if found != SKILLS:
        errors.append(f"技能集合不符：缺少 {sorted(SKILLS - found)}；多出 {sorted(found - SKILLS)}")
    for name in sorted(found):
        path = skills / name / "SKILL.md"
        parts = path.read_text(encoding="utf-8").split("---", 2)
        try:
            metadata = yaml.safe_load(parts[1]) if len(parts) == 3 and not parts[0] else None
            if not isinstance(metadata, dict) or metadata.get("name") != name:
                raise ValueError("frontmatter name 与目录不符")
            if (
                not isinstance(metadata.get("description"), str)
                or not metadata["description"].strip()
            ):
                raise ValueError("description 为空")
            ui = yaml.safe_load((path.parent / "agents/openai.yaml").read_text(encoding="utf-8"))
            if not ui["interface"]["display_name"] or not ui["interface"]["short_description"]:
                raise ValueError("界面元数据为空")
        except (ValueError, TypeError, KeyError, OSError, yaml.YAMLError) as exc:
            errors.append(f"{path.relative_to(root)}：{exc}")
    parser = MarkdownIt()
    files = 0
    links = 0
    unchanged = 0
    for path in markdown_paths(root):
        relative = path.relative_to(root)
        if not path.resolve().is_relative_to(root):
            errors.append(f"{relative}：文档软链越出仓库")
            continue
        if deferred.get(relative.as_posix()) == hashlib.sha256(path.read_bytes()).hexdigest():
            unchanged += 1
            continue
        files += 1
        tokens = parser.parse(path.read_text(encoding="utf-8"))
        for token in tokens:
            for child in token.children or []:
                target = (
                    child.attrGet("href")
                    if child.type == "link_open"
                    else child.attrGet("src")
                    if child.type == "image"
                    else None
                )
                if not target:
                    continue
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path:
                    continue
                links += 1
                destination = (path.parent / unquote(url.path)).resolve()
                if not destination.is_relative_to(root) or not destination.exists():
                    errors.append(f"{relative}：本地链接不存在或越出仓库：{target}")
    if not files:
        errors.append("没有扫描到 Markdown 文件")
    print(
        f"Markdown {files} 份，本地文件链接 {links} 条，技能 {len(found)} 个；历史文档延后 {unchanged} 份"
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    args = parser.parse_args()
    errors = check(args.root.resolve())
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
