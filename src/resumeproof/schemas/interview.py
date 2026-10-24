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
    priority: QuestionPriority
    question_text: str
    is_followup: bool = False
    followup_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CandidateAnswer(BaseModel):
    question_id: str
    claim_id: str
    answer_text: str
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GradeResult(BaseModel):
    raw_score: int = Field(ge=0, le=4, description="Raw integer rating 0 to 4")
    evidence_quote: str = Field(description="Exact quote from candidate answer supporting the evaluation")
    rubric_reference: str = Field(description="Exact quote or citation from the project rubric")
    feedback: str = Field(description="Detailed technical assessment of strengths and omissions")
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
