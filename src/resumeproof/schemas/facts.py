"""Pydantic schemas for Profile Facts (Category 1: Not interview-gated, pre-approved upon user confirmation)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ProfileFactType(str, Enum):
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    LOCATION = "location"
    EDUCATION_INSTITUTION = "education_institution"
    DEGREE = "degree"
    CGPA = "cgpa"
    GRADUATION_DATE = "graduation_date"
    EMPLOYER_NAME = "employer_name"
    EMPLOYMENT_DATES = "employment_dates"

class ProfileFact(BaseModel):
    id: Optional[str] = None
    user_id: str = "default_user"
    persona_id: Optional[str] = None
    fact_type: ProfileFactType
    value: str
    source_line: Optional[int] = None
    source_page: Optional[int] = 1
    is_confirmed: bool = False
    user_correction: Optional[str] = None
    confirmed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def allowed_tokens(self) -> set[str]:
        """Returns normalized tokens allowed by this profile fact."""
        val = self.user_correction or self.value
        # Split on whitespace and punctuation for basic tokenization
        tokens = set()
        cleaned = "".join(c if c.isalnum() or c in ("-", ".", "@", "/") else " " for c in val.lower())
        for part in cleaned.split():
            if len(part) > 1:
                tokens.add(part)
        return tokens
