"""Tests for grounded rubric builder and rubric store."""

from pathlib import Path
import pytest
from resumeproof.core.db import Database
from resumeproof.schemas.rubric import RubricItem
from resumeproof.schemas.claims import Claim
from resumeproof.rubric.builder import (
    build_rubrics_for_project,
    build_rubrics_for_persona,
)
from resumeproof.rubric.store import RubricStore

DEMO_DIR = Path("data/demo")

def test_rubrics_grounded_in_sam_okafor_projects():
    """Verifies that all rubric items for sam_okafor are grounded in verified document lines."""
    rubrics = build_rubrics_for_persona("sam_okafor", base_dir=DEMO_DIR)
    assert len(rubrics) >= 5

    for item in rubrics:
        # Invariant 1: Mandatory positive 1-based source line
        assert item.source_line is not None and item.source_line > 0
        # Invariant 2: Non-empty source quote
        assert len(item.source_quote.strip()) > 0
        # Invariant 3: Source file exists
        source_path = Path(item.source_file)
        assert source_path.exists(), f"Source file {source_path} does not exist"

        # Invariant 4: Verify the quoted line actually exists at that line number in the source file
        lines = source_path.read_text(encoding="utf-8").splitlines()
        assert 1 <= item.source_line <= len(lines)
        actual_line = lines[item.source_line - 1].strip()
        assert item.source_quote in actual_line or actual_line in item.source_quote

        # Invariant 5: Criteria define anchors 0 through 4
        assert set(item.criteria.keys()) == {0, 1, 2, 3, 4}

def test_rubrics_grounded_in_alex_and_priya_projects():
    """Verifies grounding across senior personas (alex_rivera and priya_nair)."""
    for persona in ["alex_rivera", "priya_nair"]:
        rubrics = build_rubrics_for_persona(persona, base_dir=DEMO_DIR)
        assert len(rubrics) >= 6
        for item in rubrics:
            assert item.source_line > 0
            assert len(item.source_quote) > 0
            assert Path(item.source_file).exists()

def test_rubric_item_validation_enforces_source_line_and_quote():
    """Verifies that invalid rubric items without source_line or source_quote are rejected."""
    criteria = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four"}

    # Must reject non-positive source_line
    with pytest.raises(ValueError, match="source_line must be a positive 1-based line number"):
        RubricItem(
            project_id="test_proj",
            topic="concurrency",
            source_file="README.md",
            source_line=0,
            source_quote="Some quote",
            criteria=criteria,
        )

    # Must reject empty source_quote
    with pytest.raises(ValueError, match="source_quote cannot be empty"):
        RubricItem(
            project_id="test_proj",
            topic="concurrency",
            source_file="README.md",
            source_line=5,
            source_quote="   ",
            criteria=criteria,
        )

def test_rubric_store_sqlite_crud_and_best_match():
    """Verifies SQLite persistence and best rubric matching for claims."""
    db = Database(":memory:")
    store = RubricStore(db)

    rubrics = build_rubrics_for_persona("sam_okafor", base_dir=DEMO_DIR)
    store.save_rubrics(rubrics)

    # Retrieval by project
    api_rubrics = store.get_rubrics_for_project("task-tracker-api")
    assert len(api_rubrics) >= 2

    # Retrieval by persona
    all_sam = store.get_rubrics_for_persona("sam_okafor")
    assert len(all_sam) == len(rubrics)

    # Match for a specific claim
    claim = Claim(
        id="c1",
        user_id="u1",
        persona_id="sam_okafor",
        project_id="task-tracker-api",
        text="Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling",
        tools=["SQLite", "SQLAlchemy"],
        metric_values=["350 req/sec"],
    )
    best = store.find_best_rubric_for_claim(claim, persona_id="sam_okafor")
    assert best is not None
    assert "350" in best.source_quote or "wal" in best.source_quote.lower()
