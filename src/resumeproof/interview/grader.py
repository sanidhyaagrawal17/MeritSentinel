"""Grader using qwen2.5:7b, temperature 0, atomic rubric points, and deterministic scoring."""

import time
from typing import Optional
from pydantic import BaseModel, Field

from resumeproof.core.config import settings
from resumeproof.core.llm import LLMClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem, AtomicPoint
from resumeproof.schemas.interview import GradeResult, PointEvaluation

class PointEvaluationItem(BaseModel):
    point_id: str = Field(description="The exact atomic point ID from the prompt")
    matched: bool = Field(description="Whether the candidate answer provides factual technical evidence satisfying this atomic point")
    evidence_quote: str = Field(default="", description="Exact verbatim substring copied directly from candidate answer proving this point, or empty string if not matched")

class GradeStructuredPayload(BaseModel):
    point_evaluations: list[PointEvaluationItem] = Field(default_factory=list, description="Evaluation for every supplied atomic point ID")
    points: list[PointEvaluationItem] = Field(default_factory=list, description="Alias for point_evaluations")
    feedback: str = Field(description="Objective technical critique citing only matched/missed atomic rubric points and answer quotes")
    evidence_quote: Optional[str] = Field(default=None, description="Optional overall quote from the candidate's answer")
    rubric_reference: Optional[str] = Field(default=None, description="Optional reference citation")
    matched_rubric_ids: list[str] = Field(default_factory=list, description="IDs of rubric items that were satisfied")
    missed_rubric_ids: list[str] = Field(default_factory=list, description="IDs of rubric items that were missed")
    raw_score: Optional[int] = Field(default=None, description="Optional score from mock client (ignored in favor of deterministic calculation)")

GRADER_SYSTEM_PROMPT = """You are an objective, adversarial technical evaluator auditing candidate mock interview responses.
Evaluate the candidate's answer strictly against the supplied atomic rubric points.

CRITICAL EVALUATION RULES:
1. ONLY evaluate against the supplied atomic rubric points.
2. Evaluate each atomic rubric point independently. An answer sentence or clause CAN satisfy multiple points if it contains evidence for each.
3. For EVERY supplied atomic rubric point:
   - Determine if the candidate's answer provides factual technical evidence satisfying the point.
   - If satisfied: set `matched: true` and copy the EXACT verbatim substring from the CANDIDATE ANSWER into `evidence_quote`.
   - If not satisfied: set `matched: false` and set `evidence_quote: ""`.
4. VERBATIM SUBSTRING INVARIANT:
   - `evidence_quote` MUST be an exact verbatim substring copied directly from the CANDIDATE ANSWER text.
   - NEVER copy words from the Rubric or Fact description into `evidence_quote`. It MUST come from CANDIDATE ANSWER.
   - Do NOT omit words, acronyms, or parenthetical text when copying. Copy the phrase exactly as written in CANDIDATE ANSWER.
5. PARAMETER EQUIVALENCE & MATCHING:
   - Treat technical synonyms and equivalent phrasings as fully matching:
     * '10 connections', 'connection pool of 10', 'tuned SQLAlchemy connection pooling to 10 connections' FULLY SATISFIES 'pool_size=10'.
     * '350 read req/sec' or 'reaching 350 read req/sec in our locust test' FULLY SATISFIES 'sustained 350 concurrent read req/sec' and 'concurrent_read_req_sec=350'.
     * 'enabled WAL mode' or 'Write-Ahead Logging' FULLY SATISFIES 'SQLite WAL mode'.
   - If an atomic point specifies multiple configuration details (e.g. 'Uses SQLite WAL mode with SQLAlchemy async engine and pool_size=10'), candidate evidence demonstrating the core configurations ('WAL mode' and 'connection pooling to 10 connections') SATISFIES the point. Do NOT fail the point for minor omissions like 'async engine' when the core architectural parameters are proven.
   - If the candidate's answer satisfies multiple atomic points, mark ALL applicable points as matched: true.
   - When matched: true, copy the exact phrase from CANDIDATE ANSWER into evidence_quote.
6. NO UNMENTIONED DEMANDS: In `feedback`, you may ONLY cite the supplied atomic rubric points and verbatim quotes from the candidate answer. You are STRICTLY FORBIDDEN from demanding unmentioned trade-offs, constraints, alternative designs, or technologies not present in the atomic points.
7. NO NUMERIC CHOICES: You do NOT choose the numeric score. The score is computed deterministically from your point evaluations.
8. Off-Topic Disqualification: If the answer is about an unrelated topic, mark all atomic points as `matched: false`.
9. Output valid JSON matching the specified schema with an entry in `point_evaluations` for EVERY atomic point ID.
"""


def compute_deterministic_score(matched_count: int, total_count: int) -> int:
    """Computes 0-4 score deterministically from the share of atomic points matched.

    Deterministic Mapping:
    - total_count == 0: 0
    - M / N == 0.0: score 0 (no verified points matched)
    - 0.0 < M / N < 0.4: score 1 (surface-level or generic match)
    - 0.4 <= M / N < 0.75: score 2 (partial technical match)
    - 0.75 <= M / N < 1.0: score 3 (competent match)
    - M / N == 1.0: score 4 (complete match of all verified atomic points)
    """
    if total_count <= 0 or matched_count <= 0:
        return 0
    share = matched_count / total_count
    if share >= 1.0:
        return 4
    elif share >= 0.75:
        return 3
    elif share >= 0.4:
        return 2
    else:
        return 1

class AnswerGrader:
    """Evaluates candidate answers using qwen2.5:7b with structured atomic point evaluation."""

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
        """Evaluates a candidate answer against claim and supplied atomic rubric points."""
        start_time = time.perf_counter()

        if isinstance(rubric, list):
            rubric_list = rubric
        elif rubric is not None:
            rubric_list = [rubric]
        else:
            rubric_list = []

        # Extract and ensure atomic points exist for every rubric item
        all_atomic_points: list[AtomicPoint] = []
        point_to_rubric: dict[str, str] = {}
        for r in rubric_list:
            if not r.atomic_points:
                r.atomic_points = [
                    AtomicPoint(
                        id=f"{r.id}_p0",
                        text=r.source_quote,
                        source_file=r.source_file,
                        source_line=r.source_line,
                        source_quote=r.source_quote,
                    )
                ]
            for pt in r.atomic_points:
                all_atomic_points.append(pt)
                point_to_rubric[pt.id] = r.id

        rubrics_blocks = []
        for r in rubric_list:
            pts_text = "\n".join(
                f"    - [Atomic Point ID: {p.id}] Fact: {p.text} | Source: \"{p.source_quote}\" ({p.source_file}:{p.source_line})"
                for p in r.atomic_points
            )
            rubrics_blocks.append(
                f"[Rubric Item ID: {r.id}]\n"
                f"Topic: {r.topic}\n"
                f"Reference: \"{r.source_quote}\" ({r.source_file}:{r.source_line})\n"
                f"Atomic Checkable Points:\n{pts_text}"
            )

        supplied_rubrics_text = "\n\n".join(rubrics_blocks) if rubrics_blocks else "Standard factual evaluation against claim veracity."
        fallback_ref = rubric_list[0].source_quote if rubric_list else "Project verified documentation."

        user_content = f"""Resume Claim Under Audit:
{claim.text}

Interview Question Asked:
{question_text}

Supplied Ground-Truth Rubric Items & Atomic Points:
{supplied_rubrics_text}

### CANDIDATE ANSWER (Extract verbatim evidence_quote ONLY from this section):
\"\"\"
{answer_text}
\"\"\"
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

        # Deterministic check: verify that every evidence quote is a verbatim substring of answer_text
        # Unverifiable quotes count as not matched!
        point_evals: dict[str, PointEvaluation] = {}
        raw_evals = validated.point_evaluations if validated.point_evaluations else validated.points
        llm_point_map = {p.point_id: p for p in raw_evals}

        verified_matched_count = 0
        verified_quotes: list[str] = []

        for idx, pt in enumerate(all_atomic_points):
            llm_eval = llm_point_map.get(pt.id)
            if not llm_eval and idx < len(raw_evals):
                llm_eval = raw_evals[idx]

            if llm_eval and llm_eval.matched:
                cand_quote = (llm_eval.evidence_quote or "").strip().strip("\"'.,;:()")
                # Deterministic check: verbatim substring check
                if cand_quote and cand_quote in answer_text:
                    verified_matched = True
                    verified_quote = cand_quote
                    verified_matched_count += 1
                    verified_quotes.append(cand_quote)
                elif cand_quote and cand_quote.lower() in answer_text.lower():
                    # Exact character-for-character slice recovery from answer_text
                    idx_match = answer_text.lower().find(cand_quote.lower())
                    sub_verbatim = answer_text[idx_match : idx_match + len(cand_quote)].strip()
                    if sub_verbatim and sub_verbatim in answer_text:
                        verified_matched = True
                        verified_quote = sub_verbatim
                        verified_matched_count += 1
                        verified_quotes.append(sub_verbatim)
                    else:
                        verified_matched = False
                        verified_quote = ""
                else:
                    # Unverifiable quote counts as not matched
                    verified_matched = False
                    verified_quote = ""
            else:
                verified_matched = False
                verified_quote = ""

            point_evals[pt.id] = PointEvaluation(
                point_id=pt.id,
                matched=verified_matched,
                evidence_quote=verified_quote,
            )

        # Compute numeric score deterministically from verified matched points
        total_points_count = len(all_atomic_points)
        raw_score = compute_deterministic_score(verified_matched_count, total_points_count)

        # Determine matched and missed rubric item IDs
        matched_rubric_ids: list[str] = []
        missed_rubric_ids: list[str] = []
        for r in rubric_list:
            r_pts = r.atomic_points
            r_matched = sum(1 for p in r_pts if point_evals.get(p.id, PointEvaluation(point_id=p.id, matched=False)).matched)
            if r_matched > 0:
                matched_rubric_ids.append(r.id)
            else:
                missed_rubric_ids.append(r.id)

        # Format primary evidence quote
        if verified_quotes:
            final_evidence_quote = "; ".join(verified_quotes)
        elif validated.evidence_quote and validated.evidence_quote in answer_text:
            final_evidence_quote = validated.evidence_quote
        else:
            final_evidence_quote = answer_text[:120] if raw_score == 0 else ""

        # Construct critique strictly citing atomic rubric points and answer quotes
        # Guarantees no demands for unmentioned trade-offs, constraints, or alternatives
        if verified_matched_count == total_points_count and total_points_count > 0:
            matched_citations = [
                f"Point '{p.text}' verified by \"{point_evals[p.id].evidence_quote}\""
                for p in all_atomic_points
            ]
            feedback_str = f"All verified atomic rubric points satisfied. Evidence: {'; '.join(matched_citations)}."
        elif verified_matched_count > 0:
            matched_citations = [
                f"Point '{p.text}' verified by \"{point_evals[p.id].evidence_quote}\""
                for p in all_atomic_points
                if point_evals[p.id].matched
            ]
            missed_citations = [
                f"Point '{p.text}'"
                for p in all_atomic_points
                if not point_evals[p.id].matched
            ]
            feedback_str = (
                f"Partial technical match. Verified: {'; '.join(matched_citations)}. "
                f"Missed atomic points: {'; '.join(missed_citations)}."
            )
        else:
            missed_citations = [f"Point '{p.text}'" for p in all_atomic_points]
            feedback_str = f"No verified atomic rubric points satisfied in candidate answer. Missed: {'; '.join(missed_citations)}."

        return GradeResult(
            raw_score=raw_score,
            evidence_quote=final_evidence_quote,
            rubric_reference=validated.rubric_reference or fallback_ref,
            feedback=feedback_str,
            matched_rubric_ids=matched_rubric_ids,
            missed_rubric_ids=missed_rubric_ids,
            point_evaluations=point_evals,
            cached=False,
            latency_ms=latency_ms,
        )

