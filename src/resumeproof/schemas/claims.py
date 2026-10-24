"""Pydantic schemas for Mastery Claims (Category 2: Interview-gated skills, metrics, tools, and claims)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field

class ClaimType(str, Enum):
    SKILL = "skill"
    TOOL = "tool"
    PROJECT_METRIC = "project_metric"
    EXPERIENCE_CLAIM = "experience_claim"
    ACHIEVEMENT = "achievement"

class ClaimStatus(str, Enum):
    EXTRACTED = "extracted"      # Extracted by parser/LLM, awaiting review
    CONFIRMED = "confirmed"      # User approved claim as-is
    EDITED = "edited"            # User modified the claim text/metrics/tools
    REJECTED = "rejected"        # User rejected claim (excluded from store/interviews)

class ClaimAudit(BaseModel):
    """Audit trail recording extractor output vs user correction."""
    extractor_text: str
    extractor_tools: list[str] = Field(default_factory=list)
    extractor_metrics: list[str] = Field(default_factory=list)
    user_correction_text: Optional[str] = None
    user_correction_tools: Optional[list[str]] = None
    user_correction_metrics: Optional[list[str]] = None
    action_taken: ClaimStatus = ClaimStatus.EXTRACTED
    reviewed_at: Optional[datetime] = None
    notes: Optional[str] = None

class Claim(BaseModel):
    id: Optional[str] = None
    user_id: str = "default_user"
    persona_id: Optional[str] = None
    project_id: Optional[str] = None
    claim_type: ClaimType
    text: str
    tools: list[str] = Field(default_factory=list)
    metric_values: list[str] = Field(default_factory=list)
    source_line: Optional[int] = None
    source_page: Optional[int] = 1
    raw_sentence: Optional[str] = None
    status: ClaimStatus = ClaimStatus.EXTRACTED
    audit: Optional[ClaimAudit] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def effective_text(self) -> str:
        """Returns the user-corrected text if edited, or original text."""
        if self.audit and self.audit.user_correction_text:
            return self.audit.user_correction_text
        return self.text

    def effective_tools(self) -> list[str]:
        """Returns user-corrected tools if edited, or original tools."""
        if self.audit and self.audit.user_correction_tools is not None:
            return self.audit.user_correction_tools
        return self.tools

    def effective_metrics(self) -> list[str]:
        """Returns user-corrected metrics if edited, or original metrics."""
        if self.audit and self.audit.user_correction_metrics is not None:
            return self.audit.user_correction_metrics
        return self.metric_values
