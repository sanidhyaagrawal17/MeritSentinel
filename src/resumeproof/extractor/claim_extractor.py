"""Structured claim and profile fact extractor using LLM structured outputs."""

import uuid
from typing import Optional
from pydantic import BaseModel, Field

from resumeproof.core.llm import LLMClient
from resumeproof.extractor.pdf_parser import ParseResult
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus, ClaimAudit

class RawProfileFactExtraction(BaseModel):
    fact_type: str = Field(description="One of: name, email, phone, location, education_institution, degree, cgpa, graduation_date, employer_name, employment_dates")
    value: str = Field(description="The exact extracted fact value")
    source_line: Optional[int] = Field(default=None, description="Line number where this fact appears")

class RawClaimExtraction(BaseModel):
    claim_type: str = Field(description="One of: skill, tool, project_metric, experience_claim, achievement")
    project_id: Optional[str] = Field(default=None, description="Identifier of the associated project, if applicable")
    text: str = Field(description="The extracted claim statement")
    tools: list[str] = Field(default_factory=list, description="Specific tools, libraries, or protocols mentioned")
    metric_values: list[str] = Field(default_factory=list, description="Numeric metrics or benchmarks mentioned (e.g., '45,000 req/sec', '1.13ms')")
    source_line: Optional[int] = Field(default=None, description="Line number where this claim appears")

class FullExtractionPayload(BaseModel):
    profile_facts: list[RawProfileFactExtraction] = Field(default_factory=list)
    claims: list[RawClaimExtraction] = Field(default_factory=list)

EXTRACTION_SYSTEM_PROMPT = """You are an adversarial resume auditor.
Extract all claims and facts from the provided numbered resume lines into two distinct categories:

CATEGORY 1: PROFILE FACTS (Not interview-gated)
- Name, contact details (email, phone, location), education institution, degree, CGPA, graduation date, employer names, and employment date ranges.

CATEGORY 2: MASTERY CLAIMS (Interview-gated)
- Specific technical skills, tools/libraries, performance metrics/numbers, project claims, and achievements.
- Extract any numbers, latency stats, throughput figures, and tools explicitly mentioned.

Always map each fact and claim to its exact line number in the source text.
"""

def extract_from_resume(
    parse_result: ParseResult,
    llm_client: LLMClient,
    persona_id: Optional[str] = None,
    user_id: str = "default_user",
    model: str = "qwen2.5:7b",
) -> tuple[list[ProfileFact], list[Claim]]:
    """Extracts Profile Facts and Mastery Claims from parsed resume lines."""
    numbered_text = "\n".join(f"[{line.line_number}] {line.text}" for line in parse_result.lines)

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Resume Text with Line Numbers:\n\n{numbered_text}"}
    ]

    raw_data = llm_client.structured_completion(
        messages=messages,
        schema=FullExtractionPayload,
        model=model,
        temperature=0.0
    )

    extracted_payload = FullExtractionPayload.model_validate(raw_data)

    facts: list[ProfileFact] = []
    for item in extracted_payload.profile_facts:
        try:
            ftype = ProfileFactType(item.fact_type.lower())
        except ValueError:
            ftype = ProfileFactType.NAME
        facts.append(ProfileFact(
            id=str(uuid.uuid4()),
            user_id=user_id,
            persona_id=persona_id,
            fact_type=ftype,
            value=item.value,
            source_line=item.source_line,
            source_page=1,
            is_confirmed=False
        ))

    claims: list[Claim] = []
    for item in extracted_payload.claims:
        try:
            ctype = ClaimType(item.claim_type.lower())
        except ValueError:
            ctype = ClaimType.PROJECT_METRIC

        claim_id = str(uuid.uuid4())
        audit = ClaimAudit(
            extractor_text=item.text,
            extractor_tools=item.tools,
            extractor_metrics=item.metric_values,
            action_taken=ClaimStatus.EXTRACTED
        )

        claims.append(Claim(
            id=claim_id,
            user_id=user_id,
            persona_id=persona_id,
            project_id=item.project_id,
            claim_type=ctype,
            text=item.text,
            tools=item.tools,
            metric_values=item.metric_values,
            source_line=item.source_line,
            source_page=1,
            status=ClaimStatus.EXTRACTED,
            audit=audit
        ))

    return facts, claims
