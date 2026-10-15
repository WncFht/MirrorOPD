#!/usr/bin/env python3
"""Audit the core book's literal input graph, references, and proof structure.

This scanner supports the manuscript's literal input/label syntax only; it is
not a TeX parser. XeLaTeX remains the authoritative build check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "theory/core"
ENTRY = "2026-10-15-MirrorOPD核心推导.tex"


def audit(book: Path, entry: str) -> dict:
    """Reject empty, missing, duplicate, or incomplete manuscript sources."""
    assembled: dict[Path, str] = {}

    def visit(path: Path) -> None:
        path = path.resolve()
        if not path.is_relative_to(book.resolve()):
            raise ValueError(f"Source outside book: {path}")
        if path in assembled:
            raise ValueError(f"Duplicate source inclusion: {path}")
        source = path.read_text()
        if not source.strip() or len(source.splitlines()) > 1000:
            raise ValueError(f"Empty or oversized source: {path}")
        # The maintained sources do not use verbatim environments or commented
        # inputs. Strip ordinary comments; escaped percent signs remain text.
        source = re.sub(r"(?<!\\)%[^\n]*", "", source)
        assembled[path] = source
        for include in re.findall(r"\\input\{([^}]+)\}", source):
            visit(book / (include + ".tex"))

    visit(book / entry)
    text = "\n".join(assembled.values())
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate labels")
    refs = re.findall(r"\\(?:eqref|ref\*?|cref|Cref|appref)\{([^}]+)\}", text)
    # appref's macro body uses #1; concrete calls are checked above.
    missing = sorted({r for r in refs if "#" not in r} - set(labels))
    if missing:
        raise ValueError(f"Missing references: {missing}")
    statements = 0
    for path, source in assembled.items():
        if "appendices" not in path.parts:
            continue
        for match in re.finditer(r"\\end\{(theorem|proposition|lemma|corollary)\}", source):
            if not source[match.end() :].lstrip().startswith(r"\begin{proof}"):
                raise ValueError(f"Statement without following proof: {path}")
            statements += 1
    if statements == 0:
        raise ValueError("No appendix statements were assembled")
    return {
        "entry": entry,
        "assembled_files": len(assembled),
        "labels": len(labels),
        "appendix_statements_with_proofs": statements,
        "missing_references": missing,
    }


def main() -> None:
    print(json.dumps(audit(BOOK, ENTRY), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
