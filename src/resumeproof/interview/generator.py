import re
import uuid
from typing import Optional
from pydantic import BaseModel, Field

from resumeproof.core.config import settings
from resumeproof.core.llm import LLMClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem
from resumeproof.schemas.interview import InterviewQuestion, QuestionPriority

QUESTION_SYSTEM_PROMPT = """You are a rigorous technical interviewer conducting a mock systems engineering interview.
Your goal is to test whether the candidate genuinely implemented their resume claims by asking focused questions directly derived from verified atomic rubric points.

CRITICAL RULES:
1. Questions MUST come ONLY from the supplied atomic rubric points.
2. A question may NOT ask for trade-offs, design choices, alternatives, or hypothetical constraints UNLESS an atomic point explicitly mentions one.
3. Ground your question strictly in the specific technical terms, mechanisms, metrics, or parameters in the atomic points.
4. Keep questions direct, factual, and under 3 sentences. Do not lecture, compliment, or reveal the answer.
"""

STOPWORDS = {
    "and", "with", "for", "the", "a", "an", "in", "on", "at", "to", "of", "by",
    "from", "as", "is", "using", "built", "implemented", "sustained", "how",
    "what", "did", "you", "your", "can", "could", "would", "explain", "describe",
    "detail", "details", "approach", "handle", "handling", "were", "was", "are",
    "that", "this", "these", "those", "when", "which", "where", "why", "who", "their",
    "about", "into", "over", "under", "within", "between", "through", "project", "system",
    "tell", "discuss", "give", "more", "some", "such"
}

def check_rubric_overlap(question_text: str, rubric: Optional[RubricItem]) -> bool:
    """Deterministic check verifying question's key terms overlap the target rubric's atomic points."""
    if not rubric:
        return True

    rubric_source = f"{rubric.topic} {rubric.source_quote} " + " ".join(
        f"{p.text} {p.source_quote}" for p in rubric.atomic_points
    )
    rubric_tokens = {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9_\-\.]+", rubric_source)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }

    q_tokens = {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9_\-\.]+", question_text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }

    overlap = q_tokens.intersection(rubric_tokens)
    return len(overlap) > 0

def check_question_outside_points(question_text: str, rubric: Optional[RubricItem]) -> bool:
    """Checks if a question asks for trade-offs, alternatives, or design choices outside its points."""
    if not rubric:
        return False

    all_points_text = " ".join(pt.text + " " + pt.source_quote for pt in rubric.atomic_points) + " " + rubric.source_quote
    all_points_lower = all_points_text.lower()

    FORBIDDEN_CONCEPTS = ["trade-off", "tradeoff", "trade off", "alternative", "design choice", "other options", "why not"]
    q_lower = question_text.lower()

    for concept in FORBIDDEN_CONCEPTS:
        if concept in q_lower and concept not in all_points_lower:
            return True

    return False

def strip_outside_clauses(question_text: str) -> str:
    """Removes trailing clauses asking for trade-offs or alternatives if present."""
    patterns = [
        r",?\s*(?:and\s+)?what\s+(?:were\s+the\s+)?(?:key\s+)?trade-?offs(?:\s+did\s+you\s+make)?\??",
        r",?\s*(?:and\s+)?what\s+trade-?offs\s+did\s+you\s+encounter\??",
        r",?\s*(?:and\s+)?what\s+alternatives\s+did\s+you\s+consider\??",
    ]
    cleaned = question_text
    for pat in patterns:
        cleaned = re.sub(pat, "?", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+\?", "?", cleaned).strip()
    return cleaned

class QuestionGenerator:
    """Generates technical questions grounded strictly in atomic rubric points using 3B model."""

    def __init__(self, llm_client: LLMClient, model: Optional[str] = None):
        self.llm_client = llm_client
        self.model = model or settings.interviewer_model

    def generate_question(
        self,
        claim: Claim,
        priority: QuestionPriority,
        rubric: Optional[RubricItem] = None,
        is_followup: bool = False,
        followup_reason: Optional[str] = None,
        previous_question: Optional[str] = None,
        previous_answer: Optional[str] = None,
    ) -> InterviewQuestion:
        """Generates an interview question answerable strictly from the target atomic rubric points."""
        prompt_parts = [
            f"Candidate Claim: {claim.text}",
            f"Tools Mentioned: {', '.join(claim.tools) if claim.tools else 'None'}",
            f"Metrics Mentioned: {', '.join(claim.metric_values) if claim.metric_values else 'None'}",
        ]
        if rubric:
            pts_text = "\n".join(
                f"- [Point {idx}]: {pt.text} (Source: \"{pt.source_quote}\")"
                for idx, pt in enumerate(rubric.atomic_points, start=1)
            ) if rubric.atomic_points else f"- {rubric.source_quote}"

            prompt_parts.extend([
                f"\nTARGET ATOMIC RUBRIC POINTS (Grounded strictly in Verified Project Docs):",
                f"Topic: {rubric.topic}",
                f"Atomic Points:\n{pts_text}",
                "\nCRITICAL INSTRUCTION: You MUST generate a technical interview question answerable strictly from these atomic points.",
                "STRICT BOUNDARY: Do NOT ask for trade-offs, alternative design choices, or constraints unless an atomic point explicitly mentions one.",
                "Incorporate specific technical mechanisms, protocols, and metrics directly from the atomic points above.",
            ])

        if is_followup and previous_question and previous_answer:
            prompt_parts.extend([
                f"\nPrevious Question: {previous_question}",
                f"Candidate's Previous Answer: {previous_answer}",
                f"Follow-up Focus: {followup_reason or 'Probe for concrete atomic parameters.'}",
                "Generate a sharp, focused follow-up question pressing for the specific atomic parameters that were omitted."
            ])
        else:
            prompt_parts.append(
                "Generate a direct technical question probing how the candidate implemented or verified these specific atomic parameters."
            )

        messages = [
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(prompt_parts)},
        ]

        # First generation attempt with 3B model
        raw_question = self.llm_client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=0.2,
            timeout=60.0
        )
        clean_q = raw_question.strip().strip('"').strip("'")
        if check_question_outside_points(clean_q, rubric):
            clean_q = strip_outside_clauses(clean_q)

        # Deterministic check: verify key terms overlap the target rubric item
        if rubric and not check_rubric_overlap(clean_q, rubric):
            retry_messages = [
                {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Candidate Claim: {claim.text}\n"
                    f"Target Rubric Quote: \"{rubric.source_quote}\" ({rubric.source_file}:{rubric.source_line})\n"
                    "CRITICAL: The previous question did not reference the required rubric item.\n"
                    f"You MUST ask a specific technical interview question directly answerable from: \"{rubric.source_quote}\".\n"
                    "Include key terms (technologies, metrics, or protocols) directly from the quote without asking for external trade-offs."
                )}
            ]
            raw_retry = self.llm_client.chat_completion(
                messages=retry_messages,
                model=self.model,
                temperature=0.2,
                timeout=60.0
            )
            clean_retry = raw_retry.strip().strip('"').strip("'")
            if clean_retry and len(clean_retry) >= 10:
                clean_q = strip_outside_clauses(clean_retry)

        # Fallback if response was too verbose or empty
        if not clean_q or len(clean_q) < 10:
            if rubric:
                clean_q = f"How did you implement {rubric.source_quote} in {claim.text}?"
            elif is_followup:
                clean_q = f"Could you provide more concrete implementation details regarding {claim.text}?"
            else:
                clean_q = f"How did you configure and implement {claim.text} to meet your verified project benchmarks?"

        if check_question_outside_points(clean_q, rubric):
            clean_q = strip_outside_clauses(clean_q)


        return InterviewQuestion(
            id=str(uuid.uuid4()),
            claim_id=claim.id,
            project_id=claim.project_id or (rubric.project_id if rubric else None),
            rubric_id=rubric.id if rubric else None,
            priority=priority,
            question_text=clean_q,
            is_followup=is_followup,
            followup_reason=followup_reason,
        )
