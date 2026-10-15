"""The book audit checks its documented literal TeX subset and fails closed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from book_source_audit import audit


@pytest.fixture
def book(tmp_path):
    (tmp_path / "appendices").mkdir()
    (tmp_path / "entry.tex").write_text(r"\input{appendices/a}")
    (tmp_path / "appendices/a.tex").write_text(
        r"\begin{proposition}\label{prop:a}Claim.\end{proposition}"
        r"\begin{proof}Proof.\end{proof}\ref{prop:a}"
    )
    return tmp_path


def test_literal_inputs_and_proof_sequence(book):
    result = audit(book, "entry.tex")
    assert result["assembled_files"] == 2
    assert result["appendix_statements_with_proofs"] == 1
    assert result["missing_references"] == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (r"\ref{prop:missing}", "Missing references"),
        (r"\label{prop:a}", "Duplicate labels"),
        (r"\begin{theorem}Claim.\end{theorem}", "without following proof"),
        (r"\input{appendices/a}", "Duplicate source"),
    ],
)
def test_invalid_sources_fail(book, extra, message):
    path = book / "appendices/a.tex"
    path.write_text(path.read_text() + extra)
    with pytest.raises(ValueError, match=message):
        audit(book, "entry.tex")


def test_empty_assembly_is_rejected(book):
    (book / "entry.tex").write_text("No appendix inputs.")
    with pytest.raises(ValueError, match="No appendix statements"):
        audit(book, "entry.tex")


def test_comment_reference_is_not_a_real_reference(book):
    path = book / "appendices/a.tex"
    path.write_text(path.read_text() + "\n% \\ref{missing}\n")
    assert audit(book, "entry.tex")["missing_references"] == []
