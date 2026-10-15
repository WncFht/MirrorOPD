"""Exercise MirrorOPD's migration boundaries and its actual hook setup."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# The sibling pipeline tests also import these standalone script entry points.
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from check_docs import SKILLS, check  # noqa: E402


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in SKILLS:
        skill = tmp_path / ".agents/skills" / name
        (skill / "agents").mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: 测试\n---\n\n# 技能\n")
        (skill / "agents/openai.yaml").write_text(
            "interface:\n  display_name: 测试\n  short_description: 测试说明\n"
        )
    (tmp_path / "AGENTS.md").write_text("# AGENTS.md\n")
    (tmp_path / "scripts").mkdir()
    return tmp_path


def baseline(root, path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "scripts/maintenance-deferred-docs.json").write_text(
        json.dumps({"version": 1, "files": {path.relative_to(root).as_posix(): digest}})
    )


def test_only_unchanged_legacy_documents_are_deferred(repository):
    page = repository / "docs/operations/old.md"
    page.parent.mkdir(parents=True)
    page.write_text("# 历史\n\n[旧入口](gone.py)\n")
    baseline(repository, page)
    assert check(repository) == []
    page.write_text(page.read_text() + "\n修改后的页面需要 review。\n")
    assert any("gone.py" in error for error in check(repository))


def test_baseline_cannot_exempt_agent_rules(repository):
    page = repository / "AGENTS.md"
    baseline(repository, page)
    assert any("非法历史文档" in error for error in check(repository))


def test_new_documents_in_legacy_directories_are_checked(repository):
    page = repository / "docs/operations/new.md"
    page.parent.mkdir(parents=True)
    page.write_text("# 新文档\n\n![缺失图片](missing.png)\n")
    assert any("missing.png" in error for error in check(repository))


def test_theory_and_vendor_are_excluded_but_new_hidden_rules_are_checked(repository):
    for name in (
        "theory/core/source.md",
        "upstream/backend/source.md",
        "results/raw/history.md",
        "transfers/code/history.md",
        "imported/snapshot/history.md",
        "data/raw/source.md",
        ".agents/notes/archived/process/history.md",
    ):
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 保留来源\n\n[历史](gone.md)\n")
    assert check(repository) == []
    (repository / ".agents/current.md").write_text("# 当前\n\n[断链](missing.md)\n")
    assert any("missing.md" in error for error in check(repository))


def test_current_links_to_evidence_still_require_a_target(repository):
    page = repository / "AGENTS.md"
    page.write_text("# 当前\n\n[数据](results/missing.json)\n")
    assert any("missing.json" in error for error in check(repository))


def test_scheduler_preserves_its_python_environment(repository, monkeypatch):
    commands = repository / "commands"
    commands.mkdir()
    fake = commands / "python3"
    fake.write_text("#!/bin/sh\nexit 19\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(commands))
    config = repository / "checks.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "checks": [
                    {
                        "id": "python",
                        "command": ["python3", "-c", "pass"],
                        "needs": [],
                        "modes": ["test"],
                        "timeout": 10,
                        "allow_failure": False,
                    }
                ],
            }
        )
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_checks.py"), "--config", str(config), "--mode", "test"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_skill_metadata_and_escaping_symlink_fail(repository):
    metadata = repository / ".agents/skills/maintain-docs/agents/openai.yaml"
    metadata.unlink()
    assert check(repository)
    (repository / "outside.md").symlink_to(repository.parent / "outside.md")
    assert any("软链越出" in error for error in check(repository))


def install_fixture(root):
    for name in ("setup.py", "agent-links.sh"):
        shutil.copy2(SCRIPTS / name, root / "scripts" / name)
    hooks = root / ".githooks"
    hooks.mkdir()
    for name in ("pre-commit", "pre-push"):
        path = hooks / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)


def call_setup(root):
    return subprocess.run(
        [sys.executable, str(root / "scripts/setup.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_setup_is_idempotent_and_preserves_owned_hooks(repository):
    install_fixture(repository)
    expected = (repository / ".githooks/pre-commit").read_bytes()
    assert call_setup(repository).returncode == 0
    assert call_setup(repository).returncode == 0
    assert (repository / ".githooks/pre-commit").read_bytes() == expected
    assert (repository / ".claude/skills").resolve() == repository / ".agents/skills"
    configured = subprocess.check_output(
        ["git", "config", "--get", "core.hooksPath"], cwd=repository, text=True
    ).strip()
    assert configured == ".githooks"


def test_setup_rejects_foreign_hooks_without_side_effects(repository):
    install_fixture(repository)
    subprocess.run(["git", "config", "core.hooksPath", "foreign-hooks"], cwd=repository, check=True)
    result = call_setup(repository)
    assert result.returncode != 0
    assert "拒绝覆盖" in result.stderr
    assert not (repository / ".claude").exists()


def test_setup_preserves_foreign_skill_entries(repository):
    install_fixture(repository)
    external = repository.parent / "foreign-skills"
    external.mkdir()
    (repository / ".claude").mkdir()
    (repository / ".claude/skills").symlink_to(external, target_is_directory=True)
    assert call_setup(repository).returncode != 0
    assert (repository / ".claude/skills").resolve() == external
