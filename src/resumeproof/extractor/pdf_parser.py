"""PDF and Text resume parser with pdfplumber and pypdf fallback, ligature/dash cleanup, and line numbering."""

from dataclasses import dataclass
from pathlib import Path
import re
import logging
from typing import Optional

logger = logging.getLogger("resumeproof.extractor.pdf_parser")

class ResumePageLimitExceededError(ValueError):
    """Raised when an uploaded document exceeds the maximum allowed resume page limit."""
    def __init__(self, page_count: int, max_pages: int = 5):
        self.page_count = page_count
        self.max_pages = max_pages
        super().__init__(
            f"Document exceeds maximum resume page limit: uploaded file has {page_count} pages, "
            f"but a resume cannot exceed {max_pages} pages. Please upload a standard 1–3 page resume."
        )

class ResumeLengthExceededError(ValueError):
    """Raised when an uploaded plaintext or markdown document exceeds reasonable resume length."""
    def __init__(self, line_count: int, max_lines: int = 500):
        self.line_count = line_count
        self.max_lines = max_lines
        super().__init__(
            f"Document exceeds maximum resume line limit: uploaded text has {line_count} lines, "
            f"exceeding the {max_lines} line threshold for a resume. Please upload a concise resume."
        )

@dataclass
class ExtractedLine:
    line_number: int
    page_number: int
    text: str
    raw_text: str

@dataclass
class ParseResult:
    source_path: str
    extractor_used: str  # "pdfplumber" | "pypdf" | "plaintext"
    total_pages: int
    lines: list[ExtractedLine]

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

# Ligatures and unicode normalizations
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "—": " - ",
    "–": " - ",
    "―": " - ",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "•": "-",
    "·": "-",
    "▪": "-",
    "►": "-",
    "\xa0": " ",  # non-breaking space
}

def normalize_text(text: str) -> str:
    """Normalize ligatures, unicode quotes, dashes, and bullet points."""
    for bad, good in LIGATURE_MAP.items():
        text = text.replace(bad, good)
    # Strip font-encoded bullet glyph artifacts like (cid:127) -> -
    text = re.sub(r"\(cid:\d+\)", "-", text)
    # Collapse multiple whitespace within line, but preserve single spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def unwrap_lines(lines: list[ExtractedLine]) -> list[ExtractedLine]:
    """Joins visually wrapped lines into logical semantic sentences."""
    def is_header_text(text: str) -> bool:
        clean = text.lstrip("#*•-► ").strip().upper().rstrip(':')
        if len(clean.split()) > 5 or len(clean) > 45:
            return False
        clean = re.sub(r"\s+\d{4}$", "", clean).strip()
        for kw in ["EDUCATION", "PROJECT", "SKILL", "EXPERIENCE", "EMPLOYMENT", "COMPETITION", "COURSEWORK", "ACHIEVEMENT", "CERTIFICATION", "SUMMARY", "PUBLICATIONS"]:
            if kw in clean:
                return True
        return False

    def should_join(prev_str: str, curr_str: str) -> bool:
        if not prev_str:
            return False
        prev_s = prev_str.strip()
        curr_s = curr_str.strip()
        if is_header_text(curr_s):
            return False
        # Do not join if current line starts with bullet
        if curr_s.startswith(("-", "*", "•", "►", "1.", "2.", "3.", "4.")):
            return False
        # Do not join bullet lines onto preceding metadata lines
        if curr_s.startswith(("-", "*", "•")):
            return False
        # A structural header ending with colon or bullet should not join
        if prev_s.endswith((".", "!", "?", ":")):
            return False
        # If previous line had pipes and date range, but current line is a bullet, never join
        if "|" in prev_s and (re.search(r"\b20\d\d\b", prev_s) or "Expected" in prev_s):
            # If current line is just a continuation year e.g. "2027", do join
            if re.match(r"^\d{4}$", curr_s):
                return True
            # Otherwise do not join onto complete pipe metadata headers
            if "|" in curr_s or len(curr_s.split()) > 3:
                return False
        if curr_s[0].islower() or curr_s.startswith(("[", "/", "&", "≤", "<", ">", "=")):
            return True
        if prev_s.endswith((",", ";", "-", "–", "—", "/", "\\", "tool-", "non-")):
            return True
        last_word = prev_s.split()[-1].lower() if prev_s.split() else ""
        if last_word in ("and", "or", "with", "for", "in", "to", "per", "across", "that", "using", "of", "by", "via", "on", "expected", "ibm"):
            return True
        if re.match(r"^\d{4}$", curr_s):
            return True
        return False

    joined: list[ExtractedLine] = []
    for l in lines:
        if joined and should_join(joined[-1].text, l.text):
            prev = joined[-1]
            if prev.text.endswith("-"):
                new_text = prev.text[:-1] + l.text.strip()
            else:
                new_text = prev.text + " " + l.text.strip()
            joined[-1] = ExtractedLine(
                line_number=prev.line_number,
                page_number=prev.page_number,
                text=new_text,
                raw_text=prev.raw_text + "\n" + l.raw_text
            )
        else:
            joined.append(l)

    return [
        ExtractedLine(line_number=i+1, page_number=l.page_number, text=l.text, raw_text=l.raw_text)
        for i, l in enumerate(joined)
    ]

def parse_pdf_with_pdfplumber(pdf_path: Path, max_pages: int = 5) -> list[ExtractedLine]:
    import pdfplumber

    lines: list[ExtractedLine] = []
    line_counter = 1

    # Split inline bullets e.g. "Header | Title | 2026 (cid:127) Bullet text"
    inline_bullet_pat = re.compile(r"(?<=[a-zA-Z0-9\.\)])\s*(?:\(cid:\d+\)|[•►▪])\s*(?=[A-Za-z0-9])")

    with pdfplumber.open(str(pdf_path)) as pdf:
        num_pages = len(pdf.pages)
        if num_pages > max_pages:
            raise ResumePageLimitExceededError(page_count=num_pages, max_pages=max_pages)
        if num_pages == 0:
            raise ValueError("The uploaded PDF contains 0 pages or could not be read.")

        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True) or ""
            for raw_line in text.splitlines():
                # Check for inline bullet splits
                parts = inline_bullet_pat.split(raw_line)
                for idx, part in enumerate(parts):
                    if idx > 0 and not part.strip().startswith(("-", "*", "•", "►")):
                        part = "- " + part.strip()
                    norm = normalize_text(part)
                    if norm:
                        lines.append(ExtractedLine(
                            line_number=line_counter,
                            page_number=page_idx,
                            text=norm,
                            raw_text=part
                        ))
                        line_counter += 1

    return unwrap_lines(lines)

def parse_pdf_with_pypdf(pdf_path: Path, max_pages: int = 5) -> list[ExtractedLine]:
    from pypdf import PdfReader

    lines: list[ExtractedLine] = []
    line_counter = 1

    inline_bullet_pat = re.compile(r"(?<=[a-zA-Z0-9\.\)])\s*(?:\(cid:\d+\)|[•►▪])\s*(?=[A-Za-z0-9])")

    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    if num_pages > max_pages:
        raise ResumePageLimitExceededError(page_count=num_pages, max_pages=max_pages)
    if num_pages == 0:
        raise ValueError("The uploaded PDF contains 0 pages or could not be read.")

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            parts = inline_bullet_pat.split(raw_line)
            for idx, part in enumerate(parts):
                if idx > 0 and not part.strip().startswith(("-", "*", "•", "►")):
                    part = "- " + part.strip()
                norm = normalize_text(part)
                if norm:
                    lines.append(ExtractedLine(
                        line_number=line_counter,
                        page_number=page_idx,
                        text=norm,
                        raw_text=part
                    ))
                    line_counter += 1

    return unwrap_lines(lines)

def parse_resume(
    file_path: str | Path,
    force_fallback: bool = False,
    max_pages: int = 5,
    max_lines: int = 500,
) -> ParseResult:
    """Parses a resume file (PDF or Markdown/Plaintext) with provenance, line numbers, and strict page limits."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    # Plaintext / Markdown
    if path.suffix.lower() in [".md", ".txt"]:
        content = path.read_text(encoding="utf-8")
        raw_lines = content.splitlines()
        if len(raw_lines) > max_lines:
            raise ResumeLengthExceededError(line_count=len(raw_lines), max_lines=max_lines)

        lines = []
        for idx, raw_line in enumerate(raw_lines, start=1):
            norm = normalize_text(raw_line)
            if norm:
                lines.append(ExtractedLine(
                    line_number=idx,
                    page_number=1,
                    text=norm,
                    raw_text=raw_line
                ))
        return ParseResult(
            source_path=str(path),
            extractor_used="plaintext",
            total_pages=1,
            lines=lines
        )

    # PDF Parsing
    if path.suffix.lower() == ".pdf":
        if not force_fallback:
            try:
                extracted_lines = parse_pdf_with_pdfplumber(path, max_pages=max_pages)
                return ParseResult(
                    source_path=str(path),
                    extractor_used="pdfplumber",
                    total_pages=max([l.page_number for l in extracted_lines], default=1),
                    lines=extracted_lines
                )
            except ResumePageLimitExceededError:
                raise
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed for {path} ({e}); falling back to pypdf.")

        # Fallback to pypdf
        extracted_lines = parse_pdf_with_pypdf(path, max_pages=max_pages)
        return ParseResult(
            source_path=str(path),
            extractor_used="pypdf",
            total_pages=max([l.page_number for l in extracted_lines], default=1),
            lines=extracted_lines
        )

    raise ValueError(f"Unsupported resume file format: {path.suffix}")
