"""Grader using qwen2.5:7b, temperature 0, structured output, with strict zero-tier-label invariant."""

import time
from typing import Optional
from pydantic import BaseModel, Field

from resumeproof.core.config import settings
from resumeproof.core.llm import LLMClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem
from resumeproof.schemas.interview import GradeResult

class GradeStructuredPayload(BaseModel):
    raw_score: int = Field(ge=0, le=4, description="Raw integer score from 0 to 4 based strictly on factual evidence")
    evidence_quote: str = Field(description="Exact quote from the candidate's answer that supports this evaluation")
    rubric_reference: str = Field(description="Exact quote or parameter from the project rubric grounding this evaluation")
    feedback: str = Field(description="Objective technical critique explaining strengths and deficiencies")

GRADER_SYSTEM_PROMPT = """You are an objective, adversarial technical evaluator auditing candidate mock interview responses.
Evaluate the candidate's answer against the verified project ground-truth rubric and assign an integer score from 0 to 4:

Scoring Rubric:
- Score 0: Irrelevant, refuses to answer, or makes claims that contradict verified facts.
- Score 1: Surface-level or generic answer omitting concrete implementation details.
- Score 2: Partial technical answer; mentions concepts but omits parameters, constraints, or verification.
- Score 3: Competent answer accurately matching verified project facts and architecture.
- Score 4: Expert answer demonstrating deep architectural mastery, trade-offs, and design rationale.

Rules:
1. Ground your score strictly on verifiable technical evidence.
2. Quote exact text from the candidate's answer as evidence.
3. Quote the project ground-truth benchmark or parameter that was satisfied or violated.
4. Output valid JSON matching the specified schema.
"""

class AnswerGrader:
    """Evaluates candidate answers using qwen2.5:7b with structured output at temperature 0."""

    def __init__(self, llm_client: LLMClient, model: Optional[str] = None):
        self.llm_client = llm_client
        self.model = model or settings.grader_model

    def grade_answer(
        self,
        claim: Claim,
        question_text: str,
        answer_text: str,
        rubric: Optional[RubricItem] = None,
        timeout: float = 60.0,
    ) -> GradeResult:
        """Evaluates a candidate answer against claim and project rubric without exposing tier labels."""
        start_time = time.perf_counter()

        rubric_quote = rubric.source_quote if rubric else "No specific document quote available."
        rubric_criteria_text = ""
        if rubric and rubric.criteria:
            rubric_criteria_text = "\n".join(f"  Score {k}: {v}" for k, v in sorted(rubric.criteria.items()))

        user_content = f"""Resume Claim Under Audit:
{claim.text}

Project Verified Reference:
{rubric_quote}
{f'Reference File & Line: {rubric.source_file}:{rubric.source_line}' if rubric else ''}

Criteria Anchors:
{rubric_criteria_text if rubric_criteria_text else 'Standard 0-4 technical scoring.'}

Interview Question Asked:
{question_text}

Candidate Answer:
{answer_text}
"""

        messages = [
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw_grade = self.llm_client.structured_completion(
            messages=messages,
            schema=GradeStructuredPayload,
            model=self.model,
            temperature=0.0,
            timeout=timeout,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        validated = GradeStructuredPayload.model_validate(raw_grade)

        return GradeResult(
            raw_score=validated.raw_score,
            evidence_quote=validated.evidence_quote,
            rubric_reference=validated.rubric_reference or rubric_quote,
            feedback=validated.feedback,
            cached=False,
            latency_ms=latency_ms,
        )
