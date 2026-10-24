"""Rubric schemas strictly grounded in project documentation."""

from datetime import datetime, timezone
from typing import Optional
import uuid
from pydantic import BaseModel, Field, field_validator

class RubricItem(BaseModel):
    """Rubric criterion grounded strictly in a project README or design note line."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = Field(description="Project identifier, e.g. task-tracker-api")
    persona_id: Optional[str] = Field(default=None, description="Persona identifier")
    topic: str = Field(description="Architectural aspect or topic")
    source_file: str = Field(description="Relative path to project README or design notes")
    source_line: int = Field(description="Mandatory 1-based source line number")
    source_quote: str = Field(description="Exact quote from the source file at source_line")
    criteria: dict[int, str] = Field(
        description="Scoring criteria anchors mapping 0-4 to observable factual expectations"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source_line")
    @classmethod
    def validate_source_line(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"source_line must be a positive 1-based line number; got {v}")
        return v

    @field_validator("source_quote")
    @classmethod
    def validate_source_quote(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source_quote cannot be empty; rubric items must quote the ground-truth document")
        return v.strip()

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: dict[int, str]) -> dict[int, str]:
        for score in (0, 1, 2, 3, 4):
            if score not in v and str(score) not in v:
                raise ValueError(f"Rubric criteria must define anchors for scores 0 through 4 (missing {score})")
        # Normalize keys to int
        return {int(k): str(val) for k, val in v.items()}
