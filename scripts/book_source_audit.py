#!/usr/bin/env python3
"""Verify source assembly and preserve the pre-reconstruction archive bytewise."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs/book"
ENTRY = BOOK / "2026-10-08-固定测度Token级MirrorOPD.tex"
OUT = ROOT / "results/book-reconstruction/source-audit.json"


def main():
    assembled = []

    def visit(path):
        assert path not in assembled, f"Duplicate source inclusion: {path}"
        assembled.append(path)
        text = path.read_text()
        assert r"\[" not in text, path
        assert r"\]" not in text, path
        assert len(text.splitlines()) < 1000, path
        for include in re.findall(r"\\input\{([^}]+)\}", text):
            visit(BOOK / (include + ".tex"))

    visit(ENTRY)
    all_text = "\n".join(p.read_text() for p in assembled)
    labels = re.findall(r"\\label\{([^}]+)\}", all_text)
    assert len(labels) == len(set(labels)), "Duplicate labels"
    refs = re.findall(r"\\(?:eqref|ref|cref|Cref)\{([^}]+)\}", all_text)
    missing = sorted(set(refs) - set(labels))
    assert not missing, missing
    orphan = sorted(str(p.relative_to(BOOK)) for p in BOOK.rglob("*.tex") if p not in assembled)
    assert not orphan, orphan
    images = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", all_text)
    assert all((BOOK / p).exists() for p in images), images
    git = shutil.which("git")
    tracked = (
        subprocess.check_output(  # noqa: S603 -- fixed read-only repository queries
            [git, "ls-tree", "-rz", "--name-only", "3a89c77", "docs/book"], cwd=ROOT, text=True
        )
        .strip("\0")
        .split("\0")
    )
    archive = []
    for original in tracked:
        baseline = subprocess.check_output(  # noqa: S603 -- validated path from ls-tree
            [git, "show", "3a89c77:" + original], cwd=ROOT
        )
        archived = ROOT / original.replace("docs/book/", "docs/book-old/", 1)
        assert baseline == archived.read_bytes(), archived
        archive.append(
            {
                "path": str(archived.relative_to(ROOT)),
                "sha256": hashlib.sha256(baseline).hexdigest(),
            }
        )
    result = {
        "entry": str(ENTRY.relative_to(ROOT)),
        "assembled_files": len(assembled),
        "body_lines": sum(len(p.read_text().splitlines()) for p in assembled),
        "labels": len(labels),
        "references": len(refs),
        "images": images,
        "missing_references": missing,
        "orphan_sources": orphan,
        "archive_baseline": "3a89c77",
        "archive": archive,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "archive"}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
