"""Rubric builder grounded strictly in project documentation (README and design notes).

Invariant Rules:
1. Every rubric item MUST be grounded in a verified project document line.
2. No rubric item may exist without a positive 1-based source_line and exact source_quote.
3. Scoring criteria define anchors 0 through 4 without referencing answer quality tier labels.
"""

import json
from pathlib import Path
from typing import Optional
import re
import uuid

from resumeproof.schemas.rubric import RubricItem, AtomicPoint

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
    return {
        0: f"Irrelevant, refuses to answer, or makes unsubstantiated claims contradicting verified facts (e.g. claims unverified technologies or scale for {project_name}).",
        1: f"Surface-level or generic answer. Fails to provide concrete details matching '{quote}'.",
        2: f"Partial technical answer. Mentions some relevant aspects but omits key parameters or mechanisms from '{quote}'.",
        3: f"Competent answer matching verified ground truth: '{quote}'. Demonstrates accurate understanding of project parameters.",
        4: f"Complete technical answer articulating verified atomic parameters matching '{quote}' with exact metrics, configuration, and implementation mechanisms.",
    }

def _find_supporting_atomic_points(
    statement: str,
    project_dir: Path,
) -> list[AtomicPoint]:
    """Finds supporting atomic points from design_notes.md and metrics.json for a statement."""
    supporting: list[AtomicPoint] = []
    
    STOPWORDS = {
        "and", "with", "for", "the", "a", "an", "in", "on", "at", "to", "of", "by",
        "from", "as", "is", "using", "built", "implemented", "sustained", "how",
        "what", "did", "you", "your", "can", "could", "would", "explain", "describe",
        "detail", "details", "approach", "handle", "handling", "were", "was", "are",
        "that", "this", "these", "those", "when", "which", "where", "why", "who", "their",
        "note", "tested", "never", "did", "not"
    }
    stmt_tokens = {
        t.lower() for t in re.findall(r"[a-zA-Z0-9_\-\.]+", statement)
        if len(t) >= 2 and t.lower() not in STOPWORDS
    }

    # 1. Supporting lines from design_notes.md
    design_notes_file = project_dir / "design_notes.md"
    if design_notes_file.is_file():
        try:
            rel_path = design_notes_file.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            rel_path = design_notes_file.as_posix()

        dn_lines = design_notes_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(dn_lines, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            line_tokens = {
                t.lower() for t in re.findall(r"[a-zA-Z0-9_\-\.]+", line_str)
                if len(t) >= 2 and t.lower() not in STOPWORDS
            }
            if stmt_tokens.intersection(line_tokens) or len(dn_lines) == 1:
                clean_text = re.sub(r"^[-*+]\s+", "", line_str)
                supporting.append(AtomicPoint(
                    id=str(uuid.uuid4()),
                    text=clean_text,
                    source_file=rel_path,
                    source_line=idx,
                    source_quote=line_str,
                ))

    # 2. Supporting metrics from metrics.json
    metrics_file = project_dir / "metrics.json"
    if metrics_file.is_file():
        try:
            rel_m_path = metrics_file.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            rel_m_path = metrics_file.as_posix()

        m_content = metrics_file.read_text(encoding="utf-8").strip()
        if m_content:
            try:
                m_data = json.loads(m_content)
                matches_metric = False
                for k, v in m_data.items():
                    k_str = str(k).lower()
                    v_str = str(v).lower()
                    if any(t in k_str or t in v_str or (t.isdigit() and t in v_str) for t in stmt_tokens):
                        matches_metric = True
                        break
                    for num in re.findall(r"\d+(?:\.\d+)?", statement):
                        if num in str(v) or num in k_str:
                            matches_metric = True
                            break
                if matches_metric or len(m_data) <= 2:
                    metric_summary = ", ".join(f"{k}={v}" for k, v in m_data.items())
                    supporting.append(AtomicPoint(
                        id=str(uuid.uuid4()),
                        text=f"Verified metrics: {metric_summary}",
                        source_file=rel_m_path,
                        source_line=1,
                        source_quote=m_content,
                    ))
            except Exception:
                pass

    return supporting

def build_rubrics_from_file(
    file_path: Path,
    project_id: str,
    persona_id: Optional[str] = None,
    project_dir: Optional[Path] = None,
) -> list[RubricItem]:
    """Extracts grounded rubric items from each factual line of a markdown project document."""
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    rubric_items: list[RubricItem] = []
    current_heading = "Overview"

    try:
        rel_path = file_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel_path = file_path.as_posix()

    effective_proj_dir = project_dir or file_path.parent

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            current_heading = stripped.lstrip("#").strip()
            continue

        statement = re.sub(r"^[-*+]\s+", "", stripped)
        if len(statement) < 10:
            continue

        topic = _derive_topic(statement, current_heading)
        criteria = _build_criteria_for_statement(statement, project_id)

        # Primary atomic point from the bullet line itself
        primary_point = AtomicPoint(
            id=str(uuid.uuid4()),
            text=statement,
            source_file=rel_path,
            source_line=line_idx,
            source_quote=stripped,
        )

        atomic_points = [primary_point]
        if effective_proj_dir and effective_proj_dir.is_dir() and file_path.name == "README.md":
            atomic_points.extend(_find_supporting_atomic_points(statement, effective_proj_dir))

        item = RubricItem(
            id=str(uuid.uuid4()),
            project_id=project_id,
            persona_id=persona_id,
            topic=topic,
            source_file=rel_path,
            source_line=line_idx,
            source_quote=stripped,
            criteria=criteria,
            atomic_points=atomic_points,
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
            rubrics.extend(build_rubrics_from_file(
                doc_path,
                project_id=project_id,
                persona_id=persona_id,
                project_dir=project_dir,
            ))

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
