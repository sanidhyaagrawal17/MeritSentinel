"""PDF and Text resume parser with pdfplumber and pypdf fallback, ligature/dash cleanup, and line numbering."""

from dataclasses import dataclass
from pathlib import Path
import re
import logging
from typing import Optional

logger = logging.getLogger("resumeproof.extractor.pdf_parser")

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
    # Collapse multiple whitespace within line, but preserve single spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def parse_pdf_with_pdfplumber(pdf_path: Path) -> list[ExtractedLine]:
    import pdfplumber

    lines: list[ExtractedLine] = []
    line_counter = 1

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True) or ""
            for raw_line in text.splitlines():
                norm = normalize_text(raw_line)
                if norm:
                    lines.append(ExtractedLine(
                        line_number=line_counter,
                        page_number=page_idx,
                        text=norm,
                        raw_text=raw_line
                    ))
                    line_counter += 1

    return lines

def parse_pdf_with_pypdf(pdf_path: Path) -> list[ExtractedLine]:
    from pypdf import PdfReader

    lines: list[ExtractedLine] = []
    line_counter = 1

    reader = PdfReader(str(pdf_path))
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            norm = normalize_text(raw_line)
            if norm:
                lines.append(ExtractedLine(
                    line_number=line_counter,
                    page_number=page_idx,
                    text=norm,
                    raw_text=raw_line
                ))
                line_counter += 1

    return lines

def parse_resume(file_path: str | Path, force_fallback: bool = False) -> ParseResult:
    """Parses a resume file (PDF or Markdown/Plaintext) with provenance and line numbers."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    # Plaintext / Markdown
    if path.suffix.lower() in [".md", ".txt"]:
        content = path.read_text(encoding="utf-8")
        lines = []
        for idx, raw_line in enumerate(content.splitlines(), start=1):
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
                extracted_lines = parse_pdf_with_pdfplumber(path)
                return ParseResult(
                    source_path=str(path),
                    extractor_used="pdfplumber",
                    total_pages=max([l.page_number for l in extracted_lines], default=1),
                    lines=extracted_lines
                )
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed for {path} ({e}); falling back to pypdf.")

        # Fallback to pypdf
        extracted_lines = parse_pdf_with_pypdf(path)
        return ParseResult(
            source_path=str(path),
            extractor_used="pypdf",
            total_pages=max([l.page_number for l in extracted_lines], default=1),
            lines=extracted_lines
        )

    raise ValueError(f"Unsupported resume file format: {path.suffix}")
