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
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.db.transaction() as conn:
            columns = [c[1] for c in conn.execute("PRAGMA table_info(rubrics)").fetchall()]
            if columns and "atomic_points" not in columns:
                conn.execute("ALTER TABLE rubrics ADD COLUMN atomic_points TEXT DEFAULT '[]'")

    def save_rubric(self, item: RubricItem) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rubrics (
                    id, project_id, persona_id, topic, source_file,
                    source_line, source_quote, criteria, created_at, atomic_points
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps([p.model_dump() for p in item.atomic_points]),
                )
            )

    def save_rubrics(self, items: list[RubricItem]) -> None:
        for item in items:
            self.save_rubric(item)

    def _hydrate(self, r) -> RubricItem:
        from resumeproof.schemas.rubric import AtomicPoint
        raw_pts = r["atomic_points"] if "atomic_points" in r.keys() and r["atomic_points"] else "[]"
        try:
            atomic_points = [AtomicPoint.model_validate(p) for p in json.loads(raw_pts)]
        except Exception:
            atomic_points = []

        return RubricItem(
            id=r["id"],
            project_id=r["project_id"],
            persona_id=r["persona_id"],
            topic=r["topic"],
            source_file=r["source_file"],
            source_line=r["source_line"],
            source_quote=r["source_quote"],
            criteria={int(k): str(v) for k, v in json.loads(r["criteria"]).items()},
            atomic_points=atomic_points,
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
        """Finds the most relevant grounded rubric item for a given claim based on project and text overlap.

        Strict Isolation Invariant:
        1. If claim has a project_id, ONLY search rubrics belonging to that project. NEVER leak another project's rubrics!
        2. If claim has no project_id, search persona rubrics ONLY if keyword overlap > 0.
        3. Never match against answer text; matching is strictly grounded in the question's source claim.
        """
        import re

        rubrics = []
        if claim.project_id:
            rubrics = self.get_rubrics_for_project(claim.project_id)
        elif persona_id:
            rubrics = self.get_rubrics_for_persona(persona_id)

        if not rubrics:
            return None

        STOPWORDS = {"and", "with", "for", "the", "a", "an", "in", "on", "at", "to", "of", "by", "from", "as", "is", "using", "built", "implemented", "sustained"}
        claim_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", claim.text.lower()) if t not in STOPWORDS}
        for tool in claim.tools:
            claim_terms.update(t for t in re.findall(r"[a-zA-Z0-9]+", tool.lower()) if t not in STOPWORDS)
        for metric in claim.metric_values:
            claim_terms.update(t for t in re.findall(r"[a-zA-Z0-9]+", metric.lower()) if t not in STOPWORDS)

        best_rubric = None
        max_overlap = 0

        for r in rubrics:
            quote_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", r.source_quote.lower()) if t not in STOPWORDS}
            overlap = len(claim_terms.intersection(quote_terms))
            if overlap > max_overlap:
                max_overlap = overlap
                best_rubric = r

        # If claim belongs to a project, but no specific sentence overlap was found,
        # fallback to the primary overview rubric of that SAME project.
        if best_rubric is None and claim.project_id and rubrics:
            return rubrics[0]

        return best_rubric
