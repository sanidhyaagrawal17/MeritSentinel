"""Tests for PDF and text resume parser, provenance, fallback, and ligature normalization."""

from pathlib import Path
import pytest
from resumeproof.extractor.pdf_parser import (
    parse_resume,
    normalize_text,
    LIGATURE_MAP,
)

DEMO_DIR = Path("data/demo")

def test_normalize_text_ligatures():
    raw = "The \ufb01rst \ufb02ag and e\ufb03cient code \ufb00 is here."
    clean = normalize_text(raw)
    assert clean == "The first flag and efficient code ff is here."

def test_normalize_text_dashes_and_quotes():
    raw = "Raft—Consensus – Protocol “Leader” ‘Node’ • Item 1 ▪ Item 2"
    clean = normalize_text(raw)
    assert " - " in clean
    assert '"Leader"' in clean
    assert "'Node'" in clean
    assert "- Item 1" in clean
    assert "- Item 2" in clean

def test_parse_markdown_resume():
    alex_md = DEMO_DIR / "alex_rivera" / "resume.md"
    result = parse_resume(alex_md)
    assert result.extractor_used == "plaintext"
    assert result.total_pages == 1
    assert len(result.lines) > 10
    assert result.lines[0].text.startswith("# Alex Rivera")
    assert any("Raft-KV" in line.text for line in result.lines)

def test_parse_pdf_pdfplumber():
    alex_pdf = DEMO_DIR / "alex_rivera" / "resume.pdf"
    result = parse_resume(alex_pdf)
    assert result.extractor_used == "pdfplumber"
    assert result.total_pages >= 1
    assert len(result.lines) >= 15
    # Verify line numbering is strictly contiguous
    line_nums = [l.line_number for l in result.lines]
    assert line_nums == list(range(1, len(result.lines) + 1))
    assert any("Alex Rivera" in l.text for l in result.lines)

def test_parse_pdf_pypdf_fallback():
    sam_pdf = DEMO_DIR / "sam_okafor" / "resume.pdf"
    result = parse_resume(sam_pdf, force_fallback=True)
    assert result.extractor_used == "pypdf"
    assert result.total_pages >= 1
    assert len(result.lines) >= 15
    line_nums = [l.line_number for l in result.lines]
    assert line_nums == list(range(1, len(result.lines) + 1))
    assert any("Sam Okafor" in l.text for l in result.lines)

def test_parse_all_three_demo_pdfs():
    for persona in ["alex_rivera", "priya_nair", "sam_okafor"]:
        pdf_path = DEMO_DIR / persona / "resume.pdf"
        assert pdf_path.exists()
        res = parse_resume(pdf_path)
        assert len(res.lines) >= 15
        assert res.lines[0].page_number == 1

def test_nonexistent_file_raises_error():
    with pytest.raises(FileNotFoundError):
        parse_resume("data/demo/nonexistent/resume.pdf")

def test_unsupported_format_raises_error(tmp_path):
    bad_file = tmp_path / "resume.docx"
    bad_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported resume file format"):
        parse_resume(bad_file)
