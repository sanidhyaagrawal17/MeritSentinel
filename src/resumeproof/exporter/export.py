"""Export engine for Markdown and single-page ATS-safe PDF."""

from pathlib import Path
import re
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
)

from resumeproof.verifier.strict_verifier import (
    VerificationReport,
    VerificationBlockedError,
)


def export_markdown(
    markdown_content: str,
    output_path: Path,
    verifier_report: Optional[VerificationReport] = None,
) -> Path:
    """Exports tailored resume to a clean Markdown file.

    If verifier_report is provided and failed, export is unconditionally blocked.
    """
    if verifier_report is not None and not verifier_report.passed:
        raise VerificationBlockedError(
            f"Cannot export unverified resume markdown: {verifier_report.blocked_count} sentence(s) failed strict verification.\n"
            f"Zero force-export allowed. Summary: {'; '.join(verifier_report.blocking_summary)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_content.strip() + "\n", encoding="utf-8")
    return output_path


def clean_reportlab_text(text: str) -> str:
    """Sanitizes text for ReportLab Paragraph XML parsing, eliminating raw LaTeX and unclosed tags."""
    # Strip any raw LaTeX syntax like $...$, \frac, \text
    cleaned = re.sub(r"\$([^$]+)\$", r"\1", text)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    # Escape XML entities (avoid double-escaping existing &amp;, &lt;, &gt;, &quot;, &apos;, &bull;)
    cleaned = re.sub(r"&(?!(?:amp|lt|gt|quot|apos|bull);)", "&amp;", cleaned)
    # Convert markdown bold **text** to <b>text</b>
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", cleaned)
    # Convert markdown italic *text* to <i>\1</i>
    cleaned = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", cleaned)
    # Clean any unmatched angle brackets
    cleaned = re.sub(r"<(?!(?:b|/b|i|/i|font|/font|u|/u)\b)", "&lt;", cleaned)
    return cleaned.strip()


def _build_story(markdown_content: str, scale: float = 1.0) -> list:
    """Builds ReportLab flowable story scaled by a font and spacing multiplier."""
    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        "ResumeName",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=max(11.0, 15.0 * scale),
        leading=max(13.0, 17.0 * scale),
        textColor=colors.HexColor("#111827"),
        spaceAfter=max(1.0, 3.0 * scale),
    )

    contact_style = ParagraphStyle(
        "ResumeContact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=max(7.0, 8.5 * scale),
        leading=max(8.5, 11.0 * scale),
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=max(2.0, 5.0 * scale),
    )

    heading_style = ParagraphStyle(
        "ResumeHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=max(8.0, 10.0 * scale),
        leading=max(10.0, 12.0 * scale),
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=max(2.0, 4.0 * scale),
        spaceAfter=max(1.0, 2.0 * scale),
    )

    subheading_style = ParagraphStyle(
        "ResumeSubheading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=max(7.5, 9.0 * scale),
        leading=max(9.0, 11.5 * scale),
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=max(1.5, 3.0 * scale),
        spaceAfter=max(0.5, 1.0 * scale),
    )

    bullet_style = ParagraphStyle(
        "ResumeBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=max(7.0, 8.5 * scale),
        leading=max(8.5, 11.0 * scale),
        textColor=colors.HexColor("#374151"),
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=max(0.5, 1.5 * scale),
    )

    body_style = ParagraphStyle(
        "ResumeBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=max(7.0, 8.5 * scale),
        leading=max(8.5, 11.0 * scale),
        textColor=colors.HexColor("#374151"),
        spaceAfter=max(0.5, 1.5 * scale),
    )

    story = []
    lines = markdown_content.strip().splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# "):
            name_text = clean_reportlab_text(stripped[2:])
            story.append(Paragraph(name_text, name_style))
        elif stripped.startswith("## "):
            sec_text = clean_reportlab_text(stripped[3:]).upper()
            story.append(Paragraph(sec_text, heading_style))
            story.append(HRFlowable(
                width="100%",
                thickness=0.75,
                color=colors.HexColor("#D1D5DB"),
                spaceBefore=1,
                spaceAfter=max(1.0, 2.5 * scale),
            ))
        elif stripped.startswith("### "):
            sub_text = clean_reportlab_text(stripped[4:])
            story.append(Paragraph(sub_text, subheading_style))
        elif stripped.startswith(("-", "*", "•")):
            clean_b = re.sub(r"^[-*•]\s*", "", stripped)
            bullet_xml = f"&bull; {clean_reportlab_text(clean_b)}"
            story.append(Paragraph(bullet_xml, bullet_style))
        else:
            if "|" in stripped or "@" in stripped:
                c_text = clean_reportlab_text(stripped)
                story.append(Paragraph(c_text, contact_style))
            else:
                body_xml = clean_reportlab_text(stripped)
                story.append(Paragraph(body_xml, body_style))

    return story


def export_pdf(
    markdown_content: str,
    verifier_report: VerificationReport,
    output_path: Path,
) -> Path:
    """Compiles tailored resume into a single-page, ATS-safe PDF.

    STRICT INVARIANT:
    If verifier_report.passed is False, export is unconditionally blocked with
    VerificationBlockedError. Zero force-export allowed.
    """
    if not verifier_report.passed:
        raise VerificationBlockedError(
            f"Cannot export unverified resume draft: {verifier_report.blocked_count} sentence(s) failed strict verification.\n"
            f"Zero force-export allowed. Summary: {'; '.join(verifier_report.blocking_summary)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Iterative downscaling to guarantee 1-page fit
    for scale in [1.0, 0.90, 0.82, 0.75]:
        margin = max(24.0, 36.0 * scale)
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
        )
        story = _build_story(markdown_content, scale=scale)
        doc.build(story)
        if doc.page <= 1:
            break

    return output_path
