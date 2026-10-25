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
    matched_rubric_ids: list[str] = Field(default_factory=list, description="IDs of rubric items that were satisfied by the answer")
    missed_rubric_ids: list[str] = Field(default_factory=list, description="IDs of rubric items that were missed or unsatisfied by the answer")

GRADER_SYSTEM_PROMPT = """You are an objective, adversarial technical evaluator auditing candidate mock interview responses.
Evaluate the candidate's answer strictly against the supplied verified project rubric items.

CRITICAL EVALUATION RULES:
1. ONLY evaluate against the supplied rubric items. You may ONLY credit or penalize the candidate against the explicitly supplied rubric items.
2. SCORE ANCHORS INVARIANT: Each rubric item defines explicit scoring anchors for levels 0 through 4 derived from its verified source text. You may ONLY cite these anchors. You are strictly forbidden from demanding architectural details, parameters, benchmarks, or requirements beyond what is explicitly stated in the supplied rubric anchors.
3. You MUST output which rubric item IDs were matched and missed:
   - `matched_rubric_ids`: list of rubric item IDs where the candidate provided accurate technical evidence satisfying the criteria (scores 3 or 4).
   - `missed_rubric_ids`: list of rubric item IDs where the candidate failed to satisfy, omitted, or violated the required ground-truth criteria (scores 0, 1, or 2).
   - An item ID must appear in either matched_rubric_ids or missed_rubric_ids, never in both simultaneously.
4. For every rubric item ID in `missed_rubric_ids`, the critique MUST cite the rubric item anchor and explain specifically what was missing according to that rubric item's explicit criteria.
5. Off-Topic Disqualification: If the candidate's answer is about an unrelated or wrong topic (e.g. answering about DNS, Raft, or Kafka when asked about Task-Tracker API, connection pooling, or JWT), you MUST assign a score of 0 (or at most 1 if purely generic boilerplate), list the question's rubric item IDs in `missed_rubric_ids`, and explicitly state in the feedback that the answer discusses an unrelated topic.
6. Output valid JSON matching the specified schema.
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
        rubric: Optional[RubricItem | list[RubricItem]] = None,
        timeout: float = 60.0,
    ) -> GradeResult:
        """Evaluates a candidate answer against claim and supplied project rubric without exposing tier labels."""
        start_time = time.perf_counter()

        if isinstance(rubric, list):
            rubric_list = rubric
        elif rubric is not None:
            rubric_list = [rubric]
        else:
            rubric_list = []

        rubrics_blocks = []
        for r in rubric_list:
            crit_text = "\n".join(f"    Score {k}: {v}" for k, v in sorted(r.criteria.items()))
            rubrics_blocks.append(
                f"[Rubric Item ID: {r.id}]\n"
                f"Topic: {r.topic}\n"
                f"Reference: \"{r.source_quote}\" ({r.source_file}:{r.source_line})\n"
                f"Criteria:\n{crit_text}"
            )

        supplied_rubrics_text = "\n\n".join(rubrics_blocks) if rubrics_blocks else "Standard 0-4 technical scoring against claim veracity."
        fallback_ref = rubric_list[0].source_quote if rubric_list else "Project verified documentation."

        user_content = f"""Resume Claim Under Audit:
{claim.text}

Interview Question Asked:
{question_text}

Supplied Ground-Truth Rubric Items:
{supplied_rubrics_text}

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

        matched = list(validated.matched_rubric_ids)
        missed = list(validated.missed_rubric_ids)
        if validated.raw_score >= 3:
            missed = [rid for rid in missed if rid not in matched]
        else:
            matched = [rid for rid in matched if rid not in missed]

        return GradeResult(
            raw_score=validated.raw_score,
            evidence_quote=validated.evidence_quote,
            rubric_reference=validated.rubric_reference or fallback_ref,
            feedback=validated.feedback,
            matched_rubric_ids=matched,
            missed_rubric_ids=missed,
            cached=False,
            latency_ms=latency_ms,
        )
