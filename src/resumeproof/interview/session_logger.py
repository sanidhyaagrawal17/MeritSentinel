"""Structured session logger and immutable audit trail for MeritSentinel interviews."""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Optional
import uuid

from resumeproof.core.db import Database
from resumeproof.schemas.interview import (
    InterviewSession,
    InterviewQuestion,
    InterviewTurn,
    GradeResult,
)

logger = logging.getLogger("resumeproof.session_audit")

class SessionLogger:
    """Records structured lifecycle, security, and grading events for interview sessions."""

    def __init__(self, db: Database):
        self.db = db

    def log_event(
        self,
        session_id: str,
        user_id: str,
        event_type: str,
        payload: dict[str, Any],
        severity: str = "INFO",
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """Appends an immutable audit event to session_audit_logs."""
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        payload_json = json.dumps(payload, default=str)

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO session_audit_logs (
                    id, session_id, user_id, event_type, event_payload,
                    severity, client_ip, user_agent, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    user_id,
                    event_type,
                    payload_json,
                    severity,
                    client_ip,
                    user_agent,
                    now.isoformat(),
                )
            )

        logger.info(
            f"[SESSION_AUDIT] {event_type} | session={session_id} user={user_id} "
            f"severity={severity} payload_keys={list(payload.keys())}"
        )
        return event_id

    # ---------------- High-Level Event Loggers ----------------

    def log_session_start(
        self,
        session: InterviewSession,
        claims_count: int,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        return self.log_event(
            session_id=session.id,
            user_id=session.user_id,
            event_type="SESSION_START",
            payload={
                "persona_id": session.persona_id,
                "budget": session.budget,
                "claims_in_scope": claims_count,
                "started_at": session.started_at.isoformat(),
            },
            severity="INFO",
            client_ip=client_ip,
            user_agent=user_agent,
        )

    def log_question_generated(
        self,
        session_id: str,
        user_id: str,
        question: InterviewQuestion,
        turn_index: int,
        reused_pending: bool = False,
    ) -> str:
        return self.log_event(
            session_id=session_id,
            user_id=user_id,
            event_type="QUESTION_SERVED" if reused_pending else "QUESTION_GENERATED",
            payload={
                "question_id": question.id,
                "claim_id": question.claim_id,
                "project_id": question.project_id,
                "priority": question.priority.value,
                "is_followup": question.is_followup,
                "followup_reason": question.followup_reason,
                "question_text": question.question_text,
                "turn_index": turn_index,
                "reused_pending": reused_pending,
                "latency_ms": question.latency_ms,
            },
            severity="INFO",
        )

    def log_answer_submitted(
        self,
        session_id: str,
        user_id: str,
        turn_id: str,
        question_id: str,
        claim_id: str,
        answer_text: str,
        turn_index: int,
        is_adversarial: bool = False,
        adversarial_reasons: Optional[list[str]] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        severity = "SECURITY_ALERT" if is_adversarial else "INFO"
        return self.log_event(
            session_id=session_id,
            user_id=user_id,
            event_type="SECURITY_INJECTION_DETECTED" if is_adversarial else "ANSWER_SUBMITTED",
            payload={
                "turn_id": turn_id,
                "turn_index": turn_index,
                "question_id": question_id,
                "claim_id": claim_id,
                "answer_length": len(answer_text),
                "answer_snippet": answer_text[:120],
                "is_adversarial": is_adversarial,
                "adversarial_flags": adversarial_reasons or [],
            },
            severity=severity,
            client_ip=client_ip,
            user_agent=user_agent,
        )

    def log_grade_applied(
        self,
        session_id: str,
        user_id: str,
        turn_id: str,
        claim_id: str,
        grade: GradeResult,
    ) -> str:
        return self.log_event(
            session_id=session_id,
            user_id=user_id,
            event_type="GRADE_RECORDED",
            payload={
                "turn_id": turn_id,
                "claim_id": claim_id,
                "raw_score": grade.raw_score,
                "evidence_quote": grade.evidence_quote,
                "rubric_reference": grade.rubric_reference,
                "feedback": grade.feedback,
                "matched_rubric_ids": grade.matched_rubric_ids,
                "missed_rubric_ids": grade.missed_rubric_ids,
                "cached": grade.cached,
                "latency_ms": grade.latency_ms,
            },
            severity="INFO",
        )

    def log_question_rephrased(
        self,
        session_id: str,
        user_id: str,
        original_question: str,
        rephrased_question: str,
        claim_id: str,
        turn_index: int,
        latency_ms: float,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        return self.log_event(
            session_id=session_id,
            user_id=user_id,
            event_type="QUESTION_REPHRASED",
            payload={
                "claim_id": claim_id,
                "original_question": original_question,
                "rephrased_question": rephrased_question,
                "turn_index": turn_index,
                "latency_ms": latency_ms,
            },
            severity="INFO",
            client_ip=client_ip,
            user_agent=user_agent,
        )

    def log_security_event(
        self,
        session_id: str,
        user_id: str,
        event_type: str,
        details: dict[str, Any],
        severity: str = "INFO",
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        return self.log_event(
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            payload=details,
            severity=severity,
            client_ip=client_ip,
            user_agent=user_agent,
        )

    def log_mastery_updated(
        self,
        session_id: str,
        user_id: str,
        claim_id: str,
        aggregate_score: float,
        effective_score: float,
        confidence: float,
        is_eligible: bool,
    ) -> str:
        return self.log_event(
            session_id=session_id,
            user_id=user_id,
            event_type="MASTERY_SYNCHRONIZED",
            payload={
                "claim_id": claim_id,
                "aggregate_score": aggregate_score,
                "effective_score": effective_score,
                "confidence": confidence,
                "is_eligible": is_eligible,
            },
            severity="INFO",
        )

    def log_session_completed(
        self,
        session: InterviewSession,
        avg_score: float,
        duration_seconds: float,
    ) -> str:
        return self.log_event(
            session_id=session.id,
            user_id=session.user_id,
            event_type="SESSION_COMPLETED",
            payload={
                "total_turns": len(session.turns),
                "budget": session.budget,
                "average_score": round(avg_score, 2),
                "duration_seconds": round(duration_seconds, 2),
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            },
            severity="INFO",
        )

    # ---------------- Retrieval & Audit Queries ----------------

    def get_session_logs(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieves chronological audit events for a session."""
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, user_id, event_type, event_payload,
                       severity, client_ip, user_agent, timestamp
                FROM session_audit_logs
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,)
            ).fetchall()

            events = []
            for r in rows:
                events.append({
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "user_id": r["user_id"],
                    "event_type": r["event_type"],
                    "payload": json.loads(r["event_payload"]),
                    "severity": r["severity"],
                    "client_ip": r["client_ip"],
                    "user_agent": r["user_agent"],
                    "timestamp": r["timestamp"],
                })
            return events

    def export_session_logs_json(self, session_id: str) -> str:
        """Exports audit logs as pretty-formatted JSON with metadata envelope."""
        events = self.get_session_logs(session_id)
        envelope = {
            "session_id": session_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(events),
            "events": events,
        }
        return json.dumps(envelope, indent=2, default=str)
