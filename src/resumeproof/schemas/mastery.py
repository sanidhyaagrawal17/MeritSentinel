"""Schemas for mastery tracking, answer scores, and confidence math."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field

class AnswerRecord(BaseModel):
    """Raw answer score from a grading session. Uncapped and immutable."""
    id: Optional[str] = None
    claim_id: str
    question_id: Optional[str] = None
    raw_score: int = Field(ge=0, le=4, description="Ordinal score 0-4 awarded by grader")
    evidence_quote: Optional[str] = None
    feedback: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MasteryState(BaseModel):
    """Current mastery state for a claim, with aggregate jump cap applied."""
    claim_id: str
    aggregate_score: float = Field(default=2.0, description="Decayed weighted average score (0-4), capped per answer")
    effective_score: float = Field(default=2.0, description="Bayesian shrinkage score: prior + (agg - prior)*confidence")
    confidence: float = Field(default=0.0, description="n_eff / (n_eff + 2)")
    n_effective: float = Field(default=0.0, description="Sum of time-decay weights")
    total_answers: int = 0
    is_eligible: bool = False
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
