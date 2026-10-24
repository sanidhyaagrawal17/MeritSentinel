"""Rubric builder grounded strictly in project documentation (README and design notes).

Invariant Rules:
1. Every rubric item MUST be grounded in a verified project document line.
2. No rubric item may exist without a positive 1-based source_line and exact source_quote.
3. Scoring criteria define anchors 0 through 4 without referencing answer quality tier labels.
"""

from pathlib import Path
from typing import Optional
import re
import uuid

from resumeproof.schemas.rubric import RubricItem

def _derive_topic(text: str, current_heading: str) -> str:
    """Derives a short descriptive topic slug from heading and text."""
    clean_heading = re.sub(r"[^a-zA-Z0-9]+", "_", current_heading.lower()).strip("_")
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())[:3]
    text_snippet = "_".join(words)
    if clean_heading and clean_heading not in ("readme", "features", "architecture", "verified_benchmarks", "verified_results", "performance"):
        return f"{clean_heading}_{text_snippet}"[:40]
    return text_snippet[:40] or "core_implementation"

def _build_criteria_for_statement(quote: str, project_name: str) -> dict[int, str]:
    """Generates 0-4 scoring anchors directly grounded in the source statement."""
    # Anchors focus on concrete technical evidence without using answer tier labels
    return {
        0: f"Irrelevant, refuses to answer, or makes unsubstantiated claims contradicting verified facts (e.g. claims unverified technologies or scale for {project_name}).",
        1: f"Surface-level or generic answer. Fails to provide concrete details matching '{quote}'.",
        2: f"Partial technical answer. Mentions some relevant aspects but omits key parameters or constraints from '{quote}'.",
        3: f"Competent answer matching verified ground truth: '{quote}'. Demonstrates accurate understanding of project parameters.",
        4: f"Expert answer demonstrating deep architectural mastery: accurately articulates '{quote}', explains underlying trade-offs, constraints, and why this design was chosen.",
    }

def build_rubrics_from_file(
    file_path: Path,
    project_id: str,
    persona_id: Optional[str] = None
) -> list[RubricItem]:
    """Extracts grounded rubric items from each factual line of a markdown project document."""
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    rubric_items: list[RubricItem] = []
    current_heading = "Overview"

    # Normalize relative path using forward slashes for cross-platform consistency
    try:
        rel_path = file_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel_path = file_path.as_posix()

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            current_heading = stripped.lstrip("#").strip()
            continue

        # Extract statement, removing bullet point prefixes
        statement = re.sub(r"^[-*+]\s+", "", stripped)
        if len(statement) < 10:
            continue

        topic = _derive_topic(statement, current_heading)
        criteria = _build_criteria_for_statement(statement, project_id)

        # Invariant: Must have verified source_file, 1-based source_line, and non-empty source_quote
        item = RubricItem(
            id=str(uuid.uuid4()),
            project_id=project_id,
            persona_id=persona_id,
            topic=topic,
            source_file=rel_path,
            source_line=line_idx,
            source_quote=stripped,
            criteria=criteria,
        )
        rubric_items.append(item)

    return rubric_items

def build_rubrics_for_project(
    project_dir: Path,
    persona_id: Optional[str] = None
) -> list[RubricItem]:
    """Builds rubrics for a specific project directory by scanning README.md and design_notes.md."""
    if not project_dir.is_dir():
        return []

    project_id = project_dir.name
    rubrics: list[RubricItem] = []

    for doc_name in ["README.md", "design_notes.md"]:
        doc_path = project_dir / doc_name
        if doc_path.is_file():
            rubrics.extend(build_rubrics_from_file(doc_path, project_id=project_id, persona_id=persona_id))

    return rubrics

def build_rubrics_for_persona(
    persona_id: str,
    base_dir: Path = Path("data/demo")
) -> list[RubricItem]:
    """Builds all grounded rubrics for all projects of a persona."""
    persona_dir = base_dir / persona_id / "projects"
    if not persona_dir.exists():
        return []

    all_rubrics: list[RubricItem] = []
    for proj in sorted(persona_dir.iterdir()):
        if proj.is_dir():
            all_rubrics.extend(build_rubrics_for_project(proj, persona_id=persona_id))

    return all_rubrics
