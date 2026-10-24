"""SQLite persistence for grounded rubrics."""

import json
from datetime import datetime
from typing import Optional
from resumeproof.core.db import Database
from resumeproof.schemas.rubric import RubricItem
from resumeproof.schemas.claims import Claim

class RubricStore:
    """Store for managing grounded rubric items in SQLite."""

    def __init__(self, db: Database):
        self.db = db

    def save_rubric(self, item: RubricItem) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rubrics (
                    id, project_id, persona_id, topic, source_file,
                    source_line, source_quote, criteria, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.project_id,
                    item.persona_id,
                    item.topic,
                    item.source_file,
                    item.source_line,
                    item.source_quote,
                    json.dumps(item.criteria),
                    item.created_at.isoformat(),
                )
            )

    def save_rubrics(self, items: list[RubricItem]) -> None:
        for item in items:
            self.save_rubric(item)

    def _hydrate(self, r) -> RubricItem:
        return RubricItem(
            id=r["id"],
            project_id=r["project_id"],
            persona_id=r["persona_id"],
            topic=r["topic"],
            source_file=r["source_file"],
            source_line=r["source_line"],
            source_quote=r["source_quote"],
            criteria={int(k): str(v) for k, v in json.loads(r["criteria"]).items()},
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    def get_rubric_by_id(self, rubric_id: str) -> Optional[RubricItem]:
        with self.db.transaction() as conn:
            r = conn.execute("SELECT * FROM rubrics WHERE id = ?", (rubric_id,)).fetchone()
            return self._hydrate(r) if r else None

    def get_rubrics_for_project(self, project_id: str) -> list[RubricItem]:
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT * FROM rubrics WHERE project_id = ? ORDER BY source_line ASC", (project_id,)).fetchall()
            return [self._hydrate(r) for r in rows]

    def get_rubrics_for_persona(self, persona_id: str) -> list[RubricItem]:
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT * FROM rubrics WHERE persona_id = ? ORDER BY project_id, source_line ASC", (persona_id,)).fetchall()
            return [self._hydrate(r) for r in rows]

    def find_best_rubric_for_claim(self, claim: Claim, persona_id: Optional[str] = None) -> Optional[RubricItem]:
        """Finds the most relevant grounded rubric item for a given claim based on project and text overlap."""
        rubrics = []
        if claim.project_id:
            rubrics = self.get_rubrics_for_project(claim.project_id)
        if not rubrics and persona_id:
            rubrics = self.get_rubrics_for_persona(persona_id)

        if not rubrics:
            return None

        # Score rubrics by keyword overlap with claim text, tools, and metrics
        claim_terms = set(claim.text.lower().split())
        for tool in claim.tools:
            claim_terms.add(tool.lower())
        for metric in claim.metric_values:
            claim_terms.add(metric.lower())

        best_rubric = rubrics[0]
        max_overlap = -1

        for r in rubrics:
            quote_terms = set(r.source_quote.lower().split())
            overlap = len(claim_terms.intersection(quote_terms))
            if overlap > max_overlap:
                max_overlap = overlap
                best_rubric = r

        return best_rubric
