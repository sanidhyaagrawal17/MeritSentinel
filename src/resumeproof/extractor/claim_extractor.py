"""Structured claim and profile fact extractor using LLM structured outputs."""

import uuid
from typing import Literal, Optional
from pydantic import BaseModel, Field

from resumeproof.core.llm import LLMClient
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus, ClaimAudit
from resumeproof.extractor.pdf_parser import ParseResult

ProfileFactTypeLiteral = Literal[
    "name",
    "email",
    "phone",
    "location",
    "education_institution",
    "degree",
    "cgpa",
    "graduation_date",
    "employer_name",
    "employment_dates",
]

ClaimTypeLiteral = Literal[
    "skill",
    "tool",
    "project_metric",
    "experience_claim",
    "achievement",
]

class RawProfileFactExtraction(BaseModel):
    fact_type: ProfileFactTypeLiteral | str = Field(
        description="One of: name, email, phone, location, education_institution, degree, cgpa, graduation_date, employer_name, employment_dates"
    )
    value: str = Field(description="The exact extracted fact value")
    source_line: Optional[int] = Field(default=None, description="Line number where this fact appears")

class RawClaimExtraction(BaseModel):
    claim_type: ClaimTypeLiteral | str = Field(
        description="One of: skill, tool, project_metric, experience_claim, achievement"
    )
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
- Use exact lowercase enum keys: 'name', 'email', 'phone', 'location', 'education_institution', 'degree', 'cgpa', 'graduation_date', 'employer_name', 'employment_dates'.

CATEGORY 2: MASTERY CLAIMS (Interview-gated)
- Specific technical skills, tools/libraries, performance metrics/numbers, project claims, achievements, and all bullet points under projects.
- Extract any numbers, latency stats, throughput figures, and tools explicitly mentioned.
- Do NOT restrict yourself only to lines starting with 'Claim:'; extract all project achievements, metrics, and architecture statements.
- Use exact lowercase enum keys: 'skill', 'tool', 'project_metric', 'experience_claim', 'achievement'.

Always map each fact and claim to its exact line number in the source text.
"""

def normalize_fact_type(raw_type: str, value: str) -> ProfileFactType:
    """Robustly normalizes raw LLM fact type string to ProfileFactType enum."""
    clean = raw_type.lower().strip().replace(" ", "_").replace("-", "_")
    try:
        return ProfileFactType(clean)
    except ValueError:
        pass

    # Heuristic fallback based on string content and value patterns
    if "email" in clean or "@" in value:
        return ProfileFactType.EMAIL
    if "phone" in clean or (any(c.isdigit() for c in value) and any(p in value for p in ["(", ")", "-", "+"])):
        return ProfileFactType.PHONE
    if "location" in clean or "city" in clean or "address" in clean or ("," in value and any(st in value for st in ["IL", "CA", "NY", "TX", "WA", "CO"])):
        return ProfileFactType.LOCATION
    if "institution" in clean or "school" in clean or "university" in clean or "college" in clean or "tech" in value.lower():
        return ProfileFactType.EDUCATION_INSTITUTION
    if "degree" in clean or "b.tech" in value.lower() or "b.s." in value.lower() or "m.s." in value.lower():
        return ProfileFactType.DEGREE
    if "cgpa" in clean or "gpa" in clean or "/ 10" in value or "/ 4" in value:
        return ProfileFactType.CGPA
    if "date" in clean or "graduation" in clean or any(m in value.lower() for m in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
        return ProfileFactType.GRADUATION_DATE
    if "employer" in clean or "company" in clean:
        return ProfileFactType.EMPLOYER_NAME
    if "employment" in clean:
        return ProfileFactType.EMPLOYMENT_DATES

    return ProfileFactType.NAME

def normalize_claim_type(raw_type: str) -> ClaimType:
    """Robustly normalizes raw LLM claim type string to ClaimType enum."""
    clean = raw_type.lower().strip().replace(" ", "_").replace("-", "_")
    try:
        return ClaimType(clean)
    except ValueError:
        pass

    if "metric" in clean or "benchmark" in clean or "number" in clean:
        return ClaimType.PROJECT_METRIC
    if "skill" in clean:
        return ClaimType.SKILL
    if "tool" in clean:
        return ClaimType.TOOL
    if "experience" in clean or "project" in clean:
        return ClaimType.EXPERIENCE_CLAIM
    if "achieve" in clean:
        return ClaimType.ACHIEVEMENT

    return ClaimType.PROJECT_METRIC

def extract_from_resume(
    parse_result: ParseResult,
    llm_client: LLMClient,
    persona_id: Optional[str] = None,
    user_id: str = "default_user",
    model: str = "qwen2.5:3b",
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
        ftype = normalize_fact_type(str(item.fact_type), item.value)
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
        ctype = normalize_claim_type(str(item.claim_type))
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
