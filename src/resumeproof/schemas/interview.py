"""Schemas for interview sessions, questions, turns, and structured grades."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid
from pydantic import BaseModel, Field

class QuestionPriority(str, Enum):
    UNTESTED = "untested"
    WEAK = "weak"
    STALE = "stale"

class InterviewQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str
    project_id: Optional[str] = None
    rubric_id: Optional[str] = None
    priority: QuestionPriority
    question_text: str
    is_followup: bool = False
    followup_reason: Optional[str] = None
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CandidateAnswer(BaseModel):
    question_id: str
    claim_id: str
    answer_text: str
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PointEvaluation(BaseModel):
    """Evaluation result for an individual atomic rubric point."""
    point_id: str = Field(description="ID of the evaluated atomic point")
    matched: bool = Field(description="Whether the candidate answer provided factual evidence satisfying this point")
    evidence_quote: str = Field(default="", description="Verbatim quote from candidate answer proving this point")

class GradeResult(BaseModel):
    raw_score: int = Field(ge=0, le=4, description="Raw integer rating 0 to 4")
    evidence_quote: str = Field(description="Exact quote from candidate answer supporting the evaluation")
    rubric_reference: str = Field(description="Exact quote or citation from the project rubric")
    feedback: str = Field(description="Detailed technical assessment of strengths and omissions")
    matched_rubric_ids: list[str] = Field(default_factory=list, description="List of rubric item IDs satisfied by answer")
    missed_rubric_ids: list[str] = Field(default_factory=list, description="List of rubric item IDs missed or violated")
    point_evaluations: dict[str, PointEvaluation] = Field(default_factory=dict, description="Point-by-point atomic evaluations")
    cached: bool = False
    latency_ms: float = 0.0

class InterviewTurn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    turn_index: int
    claim_id: str
    question: InterviewQuestion
    answer: CandidateAnswer
    grade: Optional[GradeResult] = None
    question_latency_ms: float = 0.0
    grading_latency_ms: float = 0.0
    turn_latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InterviewSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    persona_id: Optional[str] = None
    budget: int = 5
    total_questions: int = 0
    status: str = "active"  # "active", "completed", "aborted"
    turns: list[InterviewTurn] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
