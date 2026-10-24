"""End-to-end interview loop engine coordinating planner, generator, background grader, and stores."""

from datetime import datetime, timezone
import time
from typing import Callable, Optional
import uuid

from resumeproof.schemas.claims import Claim, ClaimStatus
from resumeproof.schemas.interview import (
    InterviewSession,
    InterviewTurn,
    InterviewQuestion,
    CandidateAnswer,
    GradeResult,
    QuestionPriority,
)
from resumeproof.store.claim_store import ClaimStore
from resumeproof.rubric.store import RubricStore
from resumeproof.interview.planner import QuestionPlanner
from resumeproof.interview.generator import QuestionGenerator
from resumeproof.interview.background import BackgroundGrader
from resumeproof.interview.session_store import SessionStore

class InterviewEngine:
    """Coordinates mock interview execution, background grading, and mastery updates."""

    def __init__(
        self,
        claim_store: ClaimStore,
        rubric_store: RubricStore,
        session_store: SessionStore,
        generator: QuestionGenerator,
        background_grader: BackgroundGrader,
    ):
        self.claim_store = claim_store
        self.rubric_store = rubric_store
        self.session_store = session_store
        self.generator = generator
        self.background_grader = background_grader

        # Active in-memory session states
        self.active_sessions: dict[str, InterviewSession] = {}
        self.active_planners: dict[str, QuestionPlanner] = {}
        self.last_answers: dict[str, str] = {}
        self.last_scores: dict[str, Optional[int]] = {}
        self.last_questions: dict[str, str] = {}

    def start_session(
        self,
        persona_id: str,
        user_id: str = "default_user",
        budget: int = 5,
    ) -> InterviewSession:
        """Initializes a new interview session for confirmed claims."""
        # Retrieve confirmed/edited claims from store
        claims = self.claim_store.get_claims(
            user_id=user_id,
            persona_id=persona_id,
            statuses=[ClaimStatus.CONFIRMED, ClaimStatus.EDITED],
        )
        if not claims:
            # Fallback: get all claims for persona if not yet reviewed
            claims = self.claim_store.get_claims(user_id=user_id, persona_id=persona_id)

        mastery_states = {}
        for c in claims:
            m = self.claim_store.get_mastery_state(c.id)
            if m:
                mastery_states[c.id] = m

        planner = QuestionPlanner(claims=claims, mastery_states=mastery_states, budget=budget)

        session = InterviewSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            persona_id=persona_id,
            budget=budget,
            status="active",
        )

        self.session_store.create_session(session)
        self.active_sessions[session.id] = session
        self.active_planners[session.id] = planner

        return session

    def get_next_question(self, session_id: str) -> Optional[InterviewQuestion]:
        """Plans and generates the next technical interview question."""
        session = self.active_sessions.get(session_id)
        planner = self.active_planners.get(session_id)
        if not session or not planner:
            return None

        last_ans = self.last_answers.get(session_id)
        last_score = self.last_scores.get(session_id)

        plan = planner.plan_next_claim(last_answer=last_ans, last_score=last_score)
        if plan is None:
            return None

        claim, priority, is_followup, fu_reason = plan
        rubric = self.rubric_store.find_best_rubric_for_claim(claim, persona_id=session.persona_id)

        prev_q = self.last_questions.get(session_id)

        question = self.generator.generate_question(
            claim=claim,
            priority=priority,
            rubric=rubric,
            is_followup=is_followup,
            followup_reason=fu_reason,
            previous_question=prev_q,
            previous_answer=last_ans,
        )

        self.last_questions[session_id] = question.question_text
        return question

    def submit_answer(
        self,
        session_id: str,
        question: InterviewQuestion,
        answer_text: str,
        wait_for_grade: bool = True,
    ) -> InterviewTurn:
        """Processes candidate answer, queues background grading, and records turn."""
        t_start = time.perf_counter()
        session = self.active_sessions[session_id]
        turn_index = len(session.turns) + 1

        turn_id = str(uuid.uuid4())
        cand_answer = CandidateAnswer(
            question_id=question.id,
            claim_id=question.claim_id,
            answer_text=answer_text,
        )

        claim_obj = self.claim_store.get_claim(question.claim_id)
        if not claim_obj:
            claim_obj = Claim(
                id=question.claim_id,
                user_id=session.user_id,
                persona_id=session.persona_id,
                text="Claim under review",
            )

        rubric = self.rubric_store.find_best_rubric_for_claim(claim_obj, persona_id=session.persona_id)

        def _on_grade_complete(grade: GradeResult) -> None:
            self.session_store.update_turn_grade_and_mastery(
                turn_id=turn_id,
                session_id=session_id,
                claim_id=question.claim_id,
                question_id=question.id,
                grade=grade,
            )
            self.last_scores[session_id] = grade.raw_score
            # Synchronize planner's mastery state for this claim
            new_state = self.claim_store.get_mastery_state(question.claim_id)
            if new_state:
                planner = self.active_planners.get(session_id)
                if planner:
                    planner.update_mastery_state(question.claim_id, new_state)

        # Dispatch grading to background worker
        future = self.background_grader.grade_async(
            task_id=turn_id,
            claim=claim_obj,
            question_text=question.question_text,
            answer_text=answer_text,
            rubric=rubric,
            on_complete=_on_grade_complete,
        )

        grade: Optional[GradeResult] = None
        if wait_for_grade:
            grade = future.result(timeout=120.0)
            self.last_scores[session_id] = grade.raw_score

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        turn = InterviewTurn(
            id=turn_id,
            session_id=session_id,
            turn_index=turn_index,
            claim_id=question.claim_id,
            question=question,
            answer=cand_answer,
            grade=grade,
            turn_latency_ms=latency_ms,
        )

        session.turns.append(turn)
        session.total_questions = len(session.turns)
        self.session_store.record_turn(turn)
        self.session_store.update_session(session)

        self.last_answers[session_id] = answer_text

        return turn

    def finalize_session(self, session_id: str) -> InterviewSession:
        """Completes session, drains any pending background grading tasks, and records completion."""
        session = self.active_sessions.get(session_id)
        if not session:
            session = self.session_store.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

        # Drain all background grading jobs
        self.background_grader.drain_all(timeout=120.0)

        # Reload finalized session from store with all updated grades
        finalized = self.session_store.get_session(session_id)
        if finalized:
            finalized.status = "completed"
            finalized.completed_at = datetime.now(timezone.utc)
            self.session_store.update_session(finalized)
            self.active_sessions[session_id] = finalized
            return finalized

        return session
