"""End-to-end interview loop engine coordinating planner, generator, background grader, and stores."""

from datetime import datetime, timezone
import time
from typing import Callable, Optional
import uuid

from resumeproof.schemas.claims import Claim, ClaimStatus, ClaimType
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
from resumeproof.interview.session_logger import SessionLogger

class InterviewEngine:
    """Coordinates mock interview execution, background grading, mastery updates, and audit logging."""

    def __init__(
        self,
        claim_store: ClaimStore,
        rubric_store: RubricStore,
        session_store: SessionStore,
        generator: QuestionGenerator,
        background_grader: BackgroundGrader,
        session_logger: Optional[SessionLogger] = None,
    ):
        self.claim_store = claim_store
        self.rubric_store = rubric_store
        self.session_store = session_store
        self.generator = generator
        self.background_grader = background_grader
        self.session_logger = session_logger

        # Active in-memory session states
        self.active_sessions: dict[str, InterviewSession] = {}
        self.active_planners: dict[str, QuestionPlanner] = {}
        self.last_answers: dict[str, str] = {}
        self.last_scores: dict[str, Optional[int]] = {}
        self.last_questions: dict[str, str] = {}

    def _rehydrate_planner(self, session: InterviewSession) -> QuestionPlanner:
        """Reconstructs question planner state from SQLite claims and past recorded turns."""
        claims = self.claim_store.get_claims(
            user_id=session.user_id,
            persona_id=session.persona_id,
            statuses=[ClaimStatus.CONFIRMED, ClaimStatus.EDITED],
        )
        if not claims:
            claims = self.claim_store.get_claims(user_id=session.user_id, persona_id=session.persona_id)

        # Invariant: Prioritize substantive project metrics, experience claims, and achievements
        interview_claims = [c for c in claims if c.claim_type != ClaimType.SKILL]
        if interview_claims:
            claims = interview_claims

        mastery_states = {}
        for c in claims:
            m = self.claim_store.get_mastery_state(c.id)
            if m:
                mastery_states[c.id] = m

        planner = QuestionPlanner(claims=claims, mastery_states=mastery_states, budget=session.budget)
        for turn in session.turns:
            planner.questions_asked += 1
            planner.questions_per_claim[turn.claim_id] = planner.questions_per_claim.get(turn.claim_id, 0) + 1
            is_fu = getattr(turn.question, "is_followup", False)
            if is_fu:
                planner.followup_counts[turn.claim_id] = planner.followup_counts.get(turn.claim_id, 0) + 1
            planner.last_claim_id = turn.claim_id
            planner.last_was_followup = is_fu
        return planner

    def start_session(
        self,
        persona_id: str,
        user_id: str = "default_user",
        budget: int = 5,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
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

        # Invariant: Prioritize substantive project metrics, experience claims, and achievements
        interview_claims = [c for c in claims if c.claim_type != ClaimType.SKILL]
        if interview_claims:
            claims = interview_claims

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

        if self.session_logger:
            self.session_logger.log_session_start(
                session=session,
                claims_count=len(claims),
                client_ip=client_ip,
                user_agent=user_agent,
            )

        return session

    def get_next_question(self, session_id: str) -> Optional[InterviewQuestion]:
        """Plans and generates the next technical interview question or returns pending question."""
        session = self.active_sessions.get(session_id)
        if not session:
            session = self.session_store.get_session(session_id)
            if session:
                self.active_sessions[session_id] = session

        if not session:
            return None

        # If session already reached its budget, return None
        if len(session.turns) >= session.budget:
            return None

        # Invariant: If a question was already generated and is awaiting candidate response, return it directly!
        if session.current_question:
            if self.session_logger:
                self.session_logger.log_question_generated(
                    session_id=session_id,
                    user_id=session.user_id,
                    question=session.current_question,
                    turn_index=len(session.turns) + 1,
                    reused_pending=True,
                )
            return session.current_question

        # Ensure planner is loaded or rehydrated
        planner = self.active_planners.get(session_id)
        if not planner:
            planner = self._rehydrate_planner(session)
            self.active_planners[session_id] = planner

        last_ans = self.last_answers.get(session_id)
        last_score = self.last_scores.get(session_id)

        plan = planner.plan_next_claim(last_answer=last_ans, last_score=last_score)
        if plan is None:
            return None

        claim, priority, is_followup, fu_reason = plan
        rubric = self.rubric_store.find_best_rubric_for_claim(claim, persona_id=session.persona_id)

        prev_q = self.last_questions.get(session_id)

        # Collect past question texts in this session to prevent repetition
        past_questions = [t.question.question_text for t in session.turns]
        for t in session.turns:
            orig = getattr(t.question, "original_question_text", None)
            if orig and orig not in past_questions:
                past_questions.append(orig)

        t_q_start = time.perf_counter()
        question = self.generator.generate_question(
            claim=claim,
            priority=priority,
            rubric=rubric,
            is_followup=is_followup,
            followup_reason=fu_reason,
            previous_question=prev_q,
            previous_answer=last_ans,
            past_questions=past_questions,
        )
        question.latency_ms = (time.perf_counter() - t_q_start) * 1000.0

        # Persist as pending question in SQLite session state
        session.current_question = question
        self.session_store.update_session(session)
        self.last_questions[session_id] = question.question_text

        if self.session_logger:
            self.session_logger.log_question_generated(
                session_id=session_id,
                user_id=session.user_id,
                question=question,
                turn_index=len(session.turns) + 1,
                reused_pending=False,
            )

        return question

    def rephrase_current_question(
        self,
        session_id: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[InterviewQuestion]:
        """Rephrases the current pending interview question for greater clarity upon candidate request."""
        session = self.active_sessions.get(session_id)
        if not session:
            session = self.session_store.get_session(session_id)
            if session:
                self.active_sessions[session_id] = session

        if not session or not session.current_question:
            return None

        current_q = session.current_question
        claim = self.claim_store.get_claim(current_q.claim_id)
        if not claim:
            claim = Claim(
                id=current_q.claim_id,
                user_id=session.user_id,
                persona_id=session.persona_id,
                claim_type=ClaimType.PROJECT_METRIC,
                text=current_q.question_text,
                project_id=current_q.project_id,
            )

        rubric = None
        if current_q.rubric_id:
            rubric = self.rubric_store.get_rubric(current_q.rubric_id)
        if not rubric:
            rubric = self.rubric_store.find_best_rubric_for_claim(claim, persona_id=session.persona_id)

        rephrased_q = self.generator.rephrase_question(
            current_question=current_q,
            claim=claim,
            rubric=rubric,
        )

        # Update in-memory session and persistent SQLite store
        session.current_question = rephrased_q
        self.session_store.update_session(session)
        self.last_questions[session_id] = rephrased_q.question_text

        if self.session_logger:
            self.session_logger.log_question_rephrased(
                session_id=session_id,
                user_id=session.user_id,
                original_question=current_q.question_text,
                rephrased_question=rephrased_q.question_text,
                claim_id=current_q.claim_id,
                turn_index=len(session.turns) + 1,
                latency_ms=round(rephrased_q.latency_ms, 2),
                client_ip=client_ip,
                user_agent=user_agent,
            )

        return rephrased_q

    def submit_answer(
        self,
        session_id: str,
        question: InterviewQuestion,
        answer_text: str,
        wait_for_grade: bool = True,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        is_adversarial: bool = False,
        adversarial_reasons: Optional[list[str]] = None,
    ) -> InterviewTurn:
        """Processes candidate answer, clears pending question, queues background grading, and records turn."""
        t_start = time.perf_counter()
        session = self.active_sessions.get(session_id)
        if not session:
            session = self.session_store.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            self.active_sessions[session_id] = session

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
                project_id=question.project_id,
                text="Claim under review",
            )
        elif question.project_id and not claim_obj.project_id:
            claim_obj.project_id = question.project_id

        # Grader receives only the rubric items attached to the claim the question was generated from
        rubric = None
        if getattr(question, "rubric_id", None):
            rubric = self.rubric_store.get_rubric_by_id(question.rubric_id)
        if not rubric:
            rubric = self.rubric_store.find_best_rubric_for_claim(claim_obj, persona_id=session.persona_id)

        q_latency = getattr(question, "latency_ms", 0.0)

        turn = InterviewTurn(
            id=turn_id,
            session_id=session_id,
            turn_index=turn_index,
            claim_id=question.claim_id,
            question=question,
            answer=cand_answer,
            grade=None,
            question_latency_ms=q_latency,
            grading_latency_ms=0.0,
            turn_latency_ms=0.0,
        )

        session.turns.append(turn)
        session.total_questions = len(session.turns)
        # Clear pending current question once answered
        session.current_question = None

        self.session_store.record_turn(turn)
        self.session_store.update_session(session)

        # Log answer event
        if self.session_logger:
            self.session_logger.log_answer_submitted(
                session_id=session_id,
                user_id=session.user_id,
                turn_id=turn_id,
                question_id=question.id,
                claim_id=question.claim_id,
                answer_text=answer_text,
                turn_index=turn_index,
                is_adversarial=is_adversarial,
                adversarial_reasons=adversarial_reasons,
                client_ip=client_ip,
                user_agent=user_agent,
            )

        def _on_grade_complete(grade: GradeResult) -> None:
            turn.grade = grade
            turn.grading_latency_ms = grade.latency_ms
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

            if self.session_logger:
                self.session_logger.log_grade_applied(
                    session_id=session_id,
                    user_id=session.user_id,
                    turn_id=turn_id,
                    claim_id=question.claim_id,
                    grade=grade,
                )
                if new_state:
                    self.session_logger.log_mastery_updated(
                        session_id=session_id,
                        user_id=session.user_id,
                        claim_id=question.claim_id,
                        aggregate_score=new_state.aggregate_score,
                        effective_score=new_state.effective_score,
                        confidence=new_state.confidence,
                        is_eligible=new_state.is_eligible,
                    )

        if is_adversarial:
            flags_summary = "; ".join(adversarial_reasons) if adversarial_reasons else "Adversarial prompt injection / obfuscation detected."
            adv_grade = GradeResult(
                raw_score=0,
                evidence_quote="",
                rubric_reference="Security Policy: Adversarial Input Blocked",
                feedback=f"Answer flagged as adversarial and scored 0/4 without LLM evaluation. Flags: {flags_summary}",
                cached=False,
                latency_ms=0.0,
            )
            _on_grade_complete(adv_grade)
            turn.turn_latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.last_answers[session_id] = answer_text
            return turn

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
            turn.grade = grade
            turn.grading_latency_ms = grade.latency_ms
            self.last_scores[session_id] = grade.raw_score

        elapsed_turn_ms = (time.perf_counter() - t_start) * 1000.0
        turn.turn_latency_ms = elapsed_turn_ms

        self.last_answers[session_id] = answer_text

        return turn

    def finalize_session(self, session_id: str) -> InterviewSession:
        """Completes session, drains any pending background grading tasks, and records completion."""
        session = self.active_sessions.get(session_id)
        if not session:
            session = self.session_store.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            self.active_sessions[session_id] = session

        # Drain all background grading jobs
        self.background_grader.drain_all(timeout=120.0)

        # Reload finalized session from store with all updated grades
        finalized = self.session_store.get_session(session_id)
        if finalized:
            finalized.status = "completed"
            now = datetime.now(timezone.utc)
            finalized.completed_at = now
            finalized.current_question = None
            self.session_store.update_session(finalized)
            self.active_sessions[session_id] = finalized

            # Calculate session stats for audit log
            duration_s = (now - finalized.started_at).total_seconds()
            scores = [t.grade.raw_score for t in finalized.turns if t.grade is not None]
            avg_score = (sum(scores) / len(scores)) if scores else 0.0

            if self.session_logger:
                self.session_logger.log_session_completed(
                    session=finalized,
                    avg_score=avg_score,
                    duration_seconds=duration_s,
                )

            return finalized

        return session
