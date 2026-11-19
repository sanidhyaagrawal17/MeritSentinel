"""SQLite storage and synchronization for interview sessions and turns."""

from datetime import datetime, timezone
from typing import Optional
from resumeproof.core.db import Database
from resumeproof.schemas.interview import (
    InterviewSession,
    InterviewTurn,
    InterviewQuestion,
    CandidateAnswer,
    GradeResult,
    QuestionPriority,
)
from resumeproof.schemas.mastery import AnswerRecord
from resumeproof.store.claim_store import ClaimStore
from resumeproof.mastery.calculator import compute_mastery_state

class SessionStore:
    """Manages persistence for interview sessions, turns, and mastery sync."""

    def __init__(self, db: Database, claim_store: ClaimStore):
        self.db = db
        self.claim_store = claim_store

    def create_session(self, session: InterviewSession) -> None:
        cq_json = session.current_question.model_dump_json() if session.current_question else None
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO interview_sessions (
                    id, user_id, persona_id, budget, total_questions, status,
                    current_question_json, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.user_id,
                    session.persona_id,
                    session.budget,
                    session.total_questions,
                    session.status,
                    cq_json,
                    session.started_at.isoformat(),
                    session.completed_at.isoformat() if session.completed_at else None,
                )
            )

    def update_session(self, session: InterviewSession) -> None:
        cq_json = session.current_question.model_dump_json() if session.current_question else None
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE interview_sessions
                SET total_questions = ?,
                    status = ?,
                    completed_at = ?,
                    current_question_json = ?
                WHERE id = ?
                """,
                (
                    session.total_questions,
                    session.status,
                    session.completed_at.isoformat() if session.completed_at else None,
                    cq_json,
                    session.id,
                )
            )

    def record_turn(self, turn: InterviewTurn) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_turns (
                    id, session_id, turn_index, claim_id, question_id, question_text,
                    answer_text, raw_score, evidence_quote, rubric_reference, feedback,
                    is_followup, turn_latency_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.id,
                    turn.session_id,
                    turn.turn_index,
                    turn.claim_id,
                    turn.question.id,
                    turn.question.question_text,
                    turn.answer.answer_text,
                    turn.grade.raw_score if turn.grade else None,
                    turn.grade.evidence_quote if turn.grade else None,
                    turn.grade.rubric_reference if turn.grade else None,
                    turn.grade.feedback if turn.grade else None,
                    1 if turn.question.is_followup else 0,
                    turn.turn_latency_ms,
                    turn.created_at.isoformat(),
                )
            )

    def update_turn_grade_and_mastery(
        self,
        turn_id: str,
        session_id: str,
        claim_id: str,
        question_id: str,
        grade: GradeResult,
    ) -> None:
        """Applies grade to session turn and synchronizes mastery state in ClaimStore."""
        now = datetime.now(timezone.utc)

        with self.db._write_lock:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    UPDATE session_turns
                    SET raw_score = ?,
                        evidence_quote = ?,
                        rubric_reference = ?,
                        feedback = ?
                    WHERE id = ?
                    """,
                    (
                        grade.raw_score,
                        grade.evidence_quote,
                        grade.rubric_reference,
                        grade.feedback,
                        turn_id,
                    )
                )

            # Synchronize mastery in ClaimStore
            existing_answers = self.claim_store.get_answer_records(claim_id)
            prev_state = self.claim_store.get_mastery_state(claim_id)

            answer_record = AnswerRecord(
                id=turn_id,
                claim_id=claim_id,
                question_id=question_id,
                raw_score=grade.raw_score,
                evidence_quote=grade.evidence_quote,
                feedback=grade.feedback,
                timestamp=now,
            )

            all_answers = existing_answers + [answer_record]
            updated_state = compute_mastery_state(
                claim_id=claim_id,
                answers=all_answers,
                previous_state=prev_state,
                evaluation_time=now,
            )

            self.claim_store.save_answer_and_mastery(answer_record, updated_state)

    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        with self.db.transaction() as conn:
            r = conn.execute("SELECT * FROM interview_sessions WHERE id = ?", (session_id,)).fetchone()
            if not r:
                return None
            turns = self.get_turns_for_session(session_id)
            current_q = None
            if "current_question_json" in r.keys() and r["current_question_json"]:
                try:
                    current_q = InterviewQuestion.model_validate_json(r["current_question_json"])
                except Exception:
                    current_q = None

            return InterviewSession(
                id=r["id"],
                user_id=r["user_id"],
                persona_id=r["persona_id"],
                budget=r["budget"],
                total_questions=max(r["total_questions"], len(turns)),
                status=r["status"],
                current_question=current_q,
                turns=turns,
                started_at=datetime.fromisoformat(r["started_at"]),
                completed_at=datetime.fromisoformat(r["completed_at"]) if r["completed_at"] else None,
            )

    def get_active_session_for_user(self, user_id: str) -> Optional[InterviewSession]:
        """Returns the most recent active in-progress interview session for a user."""
        with self.db.transaction() as conn:
            r = conn.execute(
                """
                SELECT id FROM interview_sessions
                WHERE user_id = ? AND status = 'active'
                ORDER BY started_at DESC LIMIT 1
                """,
                (user_id,)
            ).fetchone()
            if not r:
                return None
            return self.get_session(r["id"])

    # ---------------- Persistent User Sessions (Browser Authentication) ----------------

    def save_user_session(
        self,
        session_token: str,
        user_id: str,
        active_interview_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        ttl_hours: int = 168,
    ) -> None:
        """Persists a signed browser session in user_sessions table."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=ttl_hours)
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_sessions (
                    session_token, user_id, active_interview_id, client_ip, user_agent,
                    created_at, last_seen_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_token,
                    user_id,
                    active_interview_id,
                    client_ip,
                    user_agent,
                    now.isoformat(),
                    now.isoformat(),
                    expires.isoformat(),
                )
            )

    def get_user_session(self, session_token: str) -> Optional[dict]:
        """Retrieves and validates an active user browser session."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.transaction() as conn:
            r = conn.execute(
                "SELECT * FROM user_sessions WHERE session_token = ? AND expires_at > ?",
                (session_token, now_iso)
            ).fetchone()
            if not r:
                return None
            return {
                "session_token": r["session_token"],
                "user_id": r["user_id"],
                "active_interview_id": r["active_interview_id"],
                "client_ip": r["client_ip"],
                "user_agent": r["user_agent"],
                "created_at": r["created_at"],
                "last_seen_at": r["last_seen_at"],
                "expires_at": r["expires_at"],
            }

    def touch_user_session(self, session_token: str, active_interview_id: Optional[str] = None) -> None:
        """Updates last_seen timestamp and optionally active interview session pointer."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db.transaction() as conn:
            if active_interview_id:
                conn.execute(
                    "UPDATE user_sessions SET last_seen_at = ?, active_interview_id = ? WHERE session_token = ?",
                    (now_iso, active_interview_id, session_token)
                )
            else:
                conn.execute(
                    "UPDATE user_sessions SET last_seen_at = ? WHERE session_token = ?",
                    (now_iso, session_token)
                )

    def delete_user_session(self, session_token: str) -> None:
        """Removes a browser session upon logout or invalidation."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))

    def get_turns_for_session(self, session_id: str) -> list[InterviewTurn]:
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM session_turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,)
            ).fetchall()

            turns: list[InterviewTurn] = []
            for r in rows:
                q = InterviewQuestion(
                    id=r["question_id"],
                    claim_id=r["claim_id"],
                    priority=QuestionPriority.WEAK if r["is_followup"] else QuestionPriority.UNTESTED,
                    question_text=r["question_text"],
                    is_followup=bool(r["is_followup"]),
                )
                a = CandidateAnswer(
                    question_id=r["question_id"],
                    claim_id=r["claim_id"],
                    answer_text=r["answer_text"],
                    submitted_at=datetime.fromisoformat(r["timestamp"]),
                )
                grade = None
                if r["raw_score"] is not None:
                    grade = GradeResult(
                        raw_score=r["raw_score"],
                        evidence_quote=r["evidence_quote"] or "",
                        rubric_reference=r["rubric_reference"] or "",
                        feedback=r["feedback"] or "",
                    )
                turns.append(InterviewTurn(
                    id=r["id"],
                    session_id=r["session_id"],
                    turn_index=r["turn_index"],
                    claim_id=r["claim_id"],
                    question=q,
                    answer=a,
                    grade=grade,
                    turn_latency_ms=r["turn_latency_ms"],
                    created_at=datetime.fromisoformat(r["timestamp"]),
                ))
            return turns
