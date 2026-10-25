"""Tests for Markdown and ATS-safe PDF export."""

import pytest
from pathlib import Path
from resumeproof.exporter.export import export_markdown, export_pdf, clean_reportlab_text
from resumeproof.verifier.strict_verifier import VerificationReport, VerificationBlockedError


def test_clean_reportlab_text():
    """Verifies that LaTeX syntax and XML special characters are safely escaped for ReportLab."""
    raw = "Formula with $x < y$ and \\frac{1}{2} & company"
    cleaned = clean_reportlab_text(raw)
    assert "\\frac" not in cleaned
    assert "&amp;" in cleaned
    assert "$" not in cleaned


def test_export_markdown(tmp_path: Path):
    """Verifies markdown export creates a clean file."""
    md_content = "# Sam Okafor\nEmail: sam@example.com\n\n## PROJECTS\n- Built RESTful API"
    out_file = tmp_path / "tailored_resume.md"
    res = export_markdown(md_content, out_file)
    assert res.exists()
    assert res.read_text(encoding="utf-8").strip() == md_content.strip()


def test_export_pdf_success_on_verified_report(tmp_path: Path):
    """Verifies that export_pdf builds a valid single-page PDF when verification passes."""
    md_content = """# Sam Okafor
Chicago, IL | sam@example.com | (555) 432-1098

## EDUCATION
**Illinois Tech** - June 2024
B.Tech in Computer Science and Engineering (CGPA: 7.3 / 10.0)

## VERIFIED PROJECTS
### Task-Tracker API
- Built RESTful API with SQLite connection pooling sustaining 350 concurrent read req/sec.

### Mini-DNS-Resolver
- Implemented recursive UDP DNS resolver resolving queries in under 45ms for cached records.

## VERIFIED TECHNICAL SKILLS
- FastAPI, Python, SQLite, Sockets, TCP/IP
"""
    report = VerificationReport(
        passed=True,
        total_sentences=6,
        verified_count=6,
        blocked_count=0,
        sentence_results=[],
    )

    pdf_out = tmp_path / "resume.pdf"
    res_path = export_pdf(md_content, report, pdf_out)
    assert res_path.exists()
    assert res_path.stat().st_size > 1000  # valid PDF generated


def test_export_pdf_blocked_on_unverified_report(tmp_path: Path):
    """CRITICAL INVARIANT: Zero force-export allowed. export_pdf must raise VerificationBlockedError."""
    md_content = "# Candidate\n- Claiming unverified skills"
    failing_report = VerificationReport(
        passed=False,
        total_sentences=2,
        verified_count=1,
        blocked_count=1,
        sentence_results=[],
        blocking_summary=["Unbacked tool: Kubernetes"],
    )

    pdf_out = tmp_path / "blocked_resume.pdf"
    with pytest.raises(VerificationBlockedError, match="Cannot export unverified resume draft"):
        export_pdf(md_content, failing_report, pdf_out)

    assert not pdf_out.exists()


def test_export_markdown_blocked_on_unverified_report(tmp_path: Path):
    """CRITICAL INVARIANT: Zero force-export allowed. export_markdown must raise VerificationBlockedError when unverified."""
    md_content = "# Candidate\n- Claiming unverified skills"
    failing_report = VerificationReport(
        passed=False,
        total_sentences=2,
        verified_count=1,
        blocked_count=1,
        sentence_results=[],
        blocking_summary=["Unbacked tool: Kubernetes"],
    )

    md_out = tmp_path / "blocked_resume.md"
    with pytest.raises(VerificationBlockedError, match="Cannot export unverified resume markdown"):
        export_markdown(md_content, md_out, verifier_report=failing_report)

    assert not md_out.exists()
