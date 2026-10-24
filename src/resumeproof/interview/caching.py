"""In-memory and SQLite caching for evaluation grades."""

from datetime import datetime, timezone
import hashlib
from typing import Optional
from resumeproof.core.db import Database
from resumeproof.schemas.interview import GradeResult

class GradingCache:
    """Two-tier (in-memory + SQLite) cache for grading evaluations."""

    def __init__(self, db: Database):
        self.db = db
        self._memory_cache: dict[str, GradeResult] = {}

    @staticmethod
    def compute_cache_key(claim_id: str, question_text: str, answer_text: str, model: str) -> str:
        q_norm = question_text.strip().lower()
        a_norm = answer_text.strip().lower()
        raw = f"{claim_id}::{q_norm}::{a_norm}::{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, claim_id: str, question_text: str, answer_text: str, model: str) -> Optional[GradeResult]:
        key = self.compute_cache_key(claim_id, question_text, answer_text, model)

        # Tier 1: Check in-memory cache
        if key in self._memory_cache:
            res = self._memory_cache[key]
            return GradeResult(
                raw_score=res.raw_score,
                evidence_quote=res.evidence_quote,
                rubric_reference=res.rubric_reference,
                feedback=res.feedback,
                cached=True,
                latency_ms=0.0,
            )

        # Tier 2: Check SQLite database cache
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM grading_cache WHERE cache_key = ?",
                (key,)
            ).fetchone()
            if row:
                res = GradeResult(
                    raw_score=row["raw_score"],
                    evidence_quote=row["evidence_quote"],
                    rubric_reference=row["rubric_reference"],
                    feedback=row["feedback"],
                    cached=True,
                    latency_ms=0.0,
                )
                self._memory_cache[key] = res
                return res

        return None

    def set(
        self,
        claim_id: str,
        question_text: str,
        answer_text: str,
        model: str,
        grade: GradeResult,
    ) -> None:
        key = self.compute_cache_key(claim_id, question_text, answer_text, model)
        self._memory_cache[key] = grade

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO grading_cache (
                    cache_key, raw_score, evidence_quote, rubric_reference, feedback, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    grade.raw_score,
                    grade.evidence_quote,
                    grade.rubric_reference,
                    grade.feedback,
                    model,
                    datetime.now(timezone.utc).isoformat(),
                )
            )
