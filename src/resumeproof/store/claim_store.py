"""SQLite Claim and Profile Fact Store with audit tracking and mastery states."""

import json
from datetime import datetime, timezone
from typing import Optional
from resumeproof.core.db import Database
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus, ClaimAudit
from resumeproof.schemas.mastery import AnswerRecord, MasteryState

class ClaimStore:
    def __init__(self, db: Database):
        self.db = db

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "ClaimStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _hydrate_claim(self, r, conn) -> Claim:
        audit_row = conn.execute("SELECT * FROM claim_audits WHERE claim_id = ?", (r["id"],)).fetchone()
        audit = None
        if audit_row:
            audit = ClaimAudit(
                extractor_text=audit_row["extractor_text"],
                extractor_tools=json.loads(audit_row["extractor_tools"]),
                extractor_metrics=json.loads(audit_row["extractor_metrics"]),
                user_correction_text=audit_row["user_correction_text"],
                user_correction_tools=json.loads(audit_row["user_correction_tools"]) if audit_row["user_correction_tools"] else None,
                user_correction_metrics=json.loads(audit_row["user_correction_metrics"]) if audit_row["user_correction_metrics"] else None,
                action_taken=ClaimStatus(audit_row["action_taken"]),
                reviewed_at=datetime.fromisoformat(audit_row["reviewed_at"]),
                notes=audit_row["notes"]
            )

        return Claim(
            id=r["id"],
            user_id=r["user_id"],
            persona_id=r["persona_id"],
            project_id=r["project_id"],
            claim_type=ClaimType(r["claim_type"]),
            text=r["text"],
            tools=json.loads(r["tools"]),
            metric_values=json.loads(r["metric_values"]),
            source_line=r["source_line"],
            source_page=r["source_page"],
            raw_sentence=r["raw_sentence"],
            status=ClaimStatus(r["status"]),
            audit=audit,
            created_at=datetime.fromisoformat(r["created_at"])
        )

    # ---------------- Profile Facts (Category 1: Not interview-gated) ----------------

    def save_profile_fact(self, fact: ProfileFact) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO profile_facts (
                    id, user_id, persona_id, fact_type, value, source_line,
                    source_page, is_confirmed, user_correction, confirmed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.id,
                    fact.user_id,
                    fact.persona_id,
                    fact.fact_type.value,
                    fact.value,
                    fact.source_line,
                    fact.source_page,
                    1 if fact.is_confirmed else 0,
                    fact.user_correction,
                    fact.confirmed_at.isoformat() if fact.confirmed_at else None,
                    fact.created_at.isoformat()
                )
            )

    def confirm_profile_fact(self, fact_id: str, user_correction: Optional[str] = None) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE profile_facts
                SET is_confirmed = 1,
                    user_correction = ?,
                    confirmed_at = ?
                WHERE id = ?
                """,
                (user_correction, datetime.now(timezone.utc).isoformat(), fact_id)
            )

    def get_profile_facts(self, user_id: str, only_confirmed: bool = False) -> list[ProfileFact]:
        query = "SELECT * FROM profile_facts WHERE user_id = ?"
        if only_confirmed:
            query += " AND is_confirmed = 1"

        with self.db.transaction() as conn:
            rows = conn.execute(query, (user_id,)).fetchall()

        facts: list[ProfileFact] = []
        for r in rows:
            facts.append(ProfileFact(
                id=r["id"],
                user_id=r["user_id"],
                persona_id=r["persona_id"],
                fact_type=ProfileFactType(r["fact_type"]),
                value=r["value"],
                source_line=r["source_line"],
                source_page=r["source_page"],
                is_confirmed=bool(r["is_confirmed"]),
                user_correction=r["user_correction"],
                confirmed_at=datetime.fromisoformat(r["confirmed_at"]) if r["confirmed_at"] else None,
                created_at=datetime.fromisoformat(r["created_at"])
            ))
        return facts

    # ---------------- Mastery Claims (Category 2: Interview-gated) ----------------

    def save_claim(self, claim: Claim) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO claims (
                    id, user_id, persona_id, project_id, claim_type, text,
                    tools, metric_values, source_line, source_page, raw_sentence, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.id,
                    claim.user_id,
                    claim.persona_id,
                    claim.project_id,
                    claim.claim_type.value,
                    claim.text,
                    json.dumps(claim.tools),
                    json.dumps(claim.metric_values),
                    claim.source_line,
                    claim.source_page,
                    claim.raw_sentence,
                    claim.status.value,
                    claim.created_at.isoformat()
                )
            )

            # Save initial or updated audit record
            if claim.audit:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO claim_audits (
                        id, claim_id, extractor_text, extractor_tools, extractor_metrics,
                        user_correction_text, user_correction_tools, user_correction_metrics,
                        action_taken, reviewed_at, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"audit_{claim.id}",
                        claim.id,
                        claim.audit.extractor_text,
                        json.dumps(claim.audit.extractor_tools),
                        json.dumps(claim.audit.extractor_metrics),
                        claim.audit.user_correction_text,
                        json.dumps(claim.audit.user_correction_tools) if claim.audit.user_correction_tools is not None else None,
                        json.dumps(claim.audit.user_correction_metrics) if claim.audit.user_correction_metrics is not None else None,
                        claim.audit.action_taken.value,
                        (claim.audit.reviewed_at or datetime.now(timezone.utc)).isoformat(),
                        claim.audit.notes
                    )
                )

            # Initialize mastery state if not present
            row = conn.execute("SELECT claim_id FROM mastery_states WHERE claim_id = ?", (claim.id,)).fetchone()
            if not row:
                conn.execute(
                    """
                    INSERT INTO mastery_states (
                        claim_id, aggregate_score, effective_score, confidence,
                        n_effective, total_answers, is_eligible, last_updated
                    ) VALUES (?, 2.0, 2.0, 0.0, 0.0, 0, 0, ?)
                    """,
                    (claim.id, datetime.now(timezone.utc).isoformat())
                )

    def update_claim_review(
        self,
        claim_id: str,
        action: ClaimStatus,
        user_correction_text: Optional[str] = None,
        user_correction_tools: Optional[list[str]] = None,
        user_correction_metrics: Optional[list[str]] = None,
        notes: Optional[str] = None
    ) -> None:
        """Mandatory review step: records user confirmation, edits, or rejection with full audit log."""
        with self.db.transaction() as conn:
            claim_row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
            if not claim_row:
                raise ValueError(f"Claim with id {claim_id} not found.")

            new_text = user_correction_text if user_correction_text is not None else claim_row["text"]
            new_tools = json.dumps(user_correction_tools) if user_correction_tools is not None else claim_row["tools"]
            new_metrics = json.dumps(user_correction_metrics) if user_correction_metrics is not None else claim_row["metric_values"]

            conn.execute(
                """
                UPDATE claims
                SET status = ?,
                    text = ?,
                    tools = ?,
                    metric_values = ?
                WHERE id = ?
                """,
                (action.value, new_text, new_tools, new_metrics, claim_id)
            )

            # Update or create audit log
            conn.execute(
                """
                INSERT OR REPLACE INTO claim_audits (
                    id, claim_id, extractor_text, extractor_tools, extractor_metrics,
                    user_correction_text, user_correction_tools, user_correction_metrics,
                    action_taken, reviewed_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"audit_{claim_id}",
                    claim_id,
                    claim_row["text"],
                    claim_row["tools"],
                    claim_row["metric_values"],
                    user_correction_text,
                    json.dumps(user_correction_tools) if user_correction_tools is not None else None,
                    json.dumps(user_correction_metrics) if user_correction_metrics is not None else None,
                    action.value,
                    datetime.now(timezone.utc).isoformat(),
                    notes
                )
            )

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        with self.db.transaction() as conn:
            r = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
            return self._hydrate_claim(r, conn) if r else None

    def get_claims(
        self,
        user_id: str,
        statuses: Optional[list[ClaimStatus]] = None,
        persona_id: Optional[str] = None
    ) -> list[Claim]:
        query = "SELECT * FROM claims WHERE user_id = ?"
        params: list = [user_id]
        if persona_id:
            query += " AND persona_id = ?"
            params.append(persona_id)

        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend([s.value for s in statuses])

        with self.db.transaction() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._hydrate_claim(r, conn) for r in rows]

    def clear_claims(self, user_id: str, persona_id: Optional[str] = None) -> None:
        """Clears claims, audits, and mastery states for a given user and persona."""
        with self.db.transaction() as conn:
            params: list = [user_id]
            persona_filter = ""
            if persona_id:
                persona_filter = " AND persona_id = ?"
                params.append(persona_id)

            claim_ids = [
                r[0] for r in conn.execute(
                    f"SELECT id FROM claims WHERE user_id = ?{persona_filter}",
                    params
                ).fetchall()
            ]

            for cid in claim_ids:
                conn.execute("DELETE FROM claim_audits WHERE claim_id = ?", (cid,))
                conn.execute("DELETE FROM mastery_states WHERE claim_id = ?", (cid,))
                conn.execute("DELETE FROM answer_records WHERE claim_id = ?", (cid,))

            conn.execute(f"DELETE FROM claims WHERE user_id = ?{persona_filter}", params)
            conn.execute(f"DELETE FROM profile_facts WHERE user_id = ?{persona_filter}", params)

    def get_audit_trail(self, claim_id: str) -> Optional[ClaimAudit]:
        with self.db.transaction() as conn:
            r = conn.execute("SELECT * FROM claim_audits WHERE claim_id = ?", (claim_id,)).fetchone()
            if not r:
                return None
            return ClaimAudit(
                extractor_text=r["extractor_text"],
                extractor_tools=json.loads(r["extractor_tools"]),
                extractor_metrics=json.loads(r["extractor_metrics"]),
                user_correction_text=r["user_correction_text"],
                user_correction_tools=json.loads(r["user_correction_tools"]) if r["user_correction_tools"] else None,
                user_correction_metrics=json.loads(r["user_correction_metrics"]) if r["user_correction_metrics"] else None,
                action_taken=ClaimStatus(r["action_taken"]),
                reviewed_at=datetime.fromisoformat(r["reviewed_at"]),
                notes=r["notes"]
            )

    # ---------------- Mastery State & Answer Logging ----------------

    def save_answer_and_mastery(self, answer: AnswerRecord, mastery: MasteryState) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO answer_records (
                    id, claim_id, question_id, raw_score, evidence_quote, feedback, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    answer.id,
                    answer.claim_id,
                    answer.question_id,
                    answer.raw_score,
                    answer.evidence_quote,
                    answer.feedback,
                    answer.timestamp.isoformat()
                )
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO mastery_states (
                    claim_id, aggregate_score, effective_score, confidence,
                    n_effective, total_answers, is_eligible, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mastery.claim_id,
                    mastery.aggregate_score,
                    mastery.effective_score,
                    mastery.confidence,
                    mastery.n_effective,
                    mastery.total_answers,
                    1 if mastery.is_eligible else 0,
                    mastery.last_updated.isoformat()
                )
            )

    def get_mastery_state(self, claim_id: str) -> Optional[MasteryState]:
        with self.db.transaction() as conn:
            r = conn.execute("SELECT * FROM mastery_states WHERE claim_id = ?", (claim_id,)).fetchone()
            if not r:
                return None
            return MasteryState(
                claim_id=r["claim_id"],
                aggregate_score=r["aggregate_score"],
                effective_score=r["effective_score"],
                confidence=r["confidence"],
                n_effective=r["n_effective"],
                total_answers=r["total_answers"],
                is_eligible=bool(r["is_eligible"]),
                last_updated=datetime.fromisoformat(r["last_updated"])
            )

    def get_answer_records(self, claim_id: str) -> list[AnswerRecord]:
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM answer_records WHERE claim_id = ? ORDER BY timestamp ASC",
                (claim_id,)
            ).fetchall()
            return [
                AnswerRecord(
                    id=r["id"],
                    claim_id=r["claim_id"],
                    question_id=r["question_id"],
                    raw_score=r["raw_score"],
                    evidence_quote=r["evidence_quote"],
                    feedback=r["feedback"],
                    timestamp=datetime.fromisoformat(r["timestamp"])
                )
                for r in rows
            ]

    def get_eligible_claims(
        self,
        user_id: str,
        effective_threshold: float | None = None,
        confidence_threshold: float | None = None,
        tolerance: float = 1e-4,
    ) -> list[Claim]:
        """Returns claims meeting mastery eligibility thresholds with small tolerance comparison."""
        with self.db.transaction() as conn:
            if effective_threshold is not None or confidence_threshold is not None:
                from resumeproof.core.config import settings
                eff_t = (effective_threshold if effective_threshold is not None else settings.effective_score_threshold) - tolerance
                conf_t = (confidence_threshold if confidence_threshold is not None else settings.confidence_threshold) - tolerance
                rows = conn.execute(
                    """
                    SELECT c.* FROM claims c
                    JOIN mastery_states m ON c.id = m.claim_id
                    WHERE c.user_id = ? AND m.effective_score >= ? AND m.confidence >= ?
                    """,
                    (user_id, eff_t, conf_t)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.* FROM claims c
                    JOIN mastery_states m ON c.id = m.claim_id
                    WHERE c.user_id = ? AND m.is_eligible = 1
                    """,
                    (user_id,)
                ).fetchall()
            return [self._hydrate_claim(r, conn) for r in rows]
