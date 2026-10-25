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
Your goal is to test whether the candidate genuinely implemented their resume claims or merely put buzzwords.
Rules:
1. Ask one targeted, in-depth technical question probing implementation details, system constraints, or trade-offs.
2. Ground your inquiry strictly in the stated rubric item and project context.
3. If this is a follow-up, drill directly into what was vague or unverified in their previous answer.
4. Do NOT lecture, praise, or reveal the answer. Keep questions direct and under 3 sentences.
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
    """Deterministic check verifying question's key terms overlap the target rubric item."""
    if not rubric:
        return True

    rubric_source = f"{rubric.topic} {rubric.source_quote} {rubric.criteria.get(3, '')}"
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

class QuestionGenerator:
    """Generates technical questions grounded in claims and project rubrics using 3B model."""

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
        """Generates an interview question answerable from the target rubric item."""
        prompt_parts = [
            f"Candidate Claim: {claim.text}",
            f"Tools Mentioned: {', '.join(claim.tools) if claim.tools else 'None'}",
            f"Metrics Mentioned: {', '.join(claim.metric_values) if claim.metric_values else 'None'}",
        ]
        if rubric:
            prompt_parts.extend([
                f"\nTARGET RUBRIC ITEM:",
                f"Topic: {rubric.topic}",
                f"Source Quote: \"{rubric.source_quote}\" ({rubric.source_file}:{rubric.source_line})",
                f"Expected Fact Anchor: {rubric.criteria.get(3, '')}",
                "\nCRITICAL INSTRUCTION: You MUST generate a technical interview question that is directly answerable from this target rubric item.",
                "Your question MUST incorporate specific technical terms, protocols, metrics, or mechanisms from the source quote.",
            ])

        if is_followup and previous_question and previous_answer:
            prompt_parts.extend([
                f"\nPrevious Question: {previous_question}",
                f"Candidate's Previous Answer: {previous_answer}",
                f"Follow-up Focus: {followup_reason or 'Probe for concrete architectural details.'}",
                "Generate a sharp, focused follow-up question pressing for specific implementation details and verified parameters."
            ])
        else:
            prompt_parts.append(
                "Generate a focused technical question challenging the candidate on how they implemented this and handled constraints."
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

        # Deterministic check: verify key terms overlap the target rubric item
        if rubric and not check_rubric_overlap(clean_q, rubric):
            retry_messages = [
                {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Candidate Claim: {claim.text}\n"
                    f"Target Rubric Quote: \"{rubric.source_quote}\" ({rubric.source_file}:{rubric.source_line})\n"
                    "CRITICAL: The previous question did not reference the required rubric item.\n"
                    f"You MUST ask a specific technical interview question directly answerable from: \"{rubric.source_quote}\".\n"
                    "Include key terms (technologies, metrics, or protocols) directly from the quote."
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
                clean_q = clean_retry

        # Fallback if response was too verbose or empty
        if not clean_q or len(clean_q) < 10:
            if rubric:
                clean_q = f"How did you implement {rubric.source_quote} in {claim.text}?"
            elif is_followup:
                clean_q = f"Could you provide more concrete implementation details regarding {claim.text}?"
            else:
                clean_q = f"How did you implement {claim.text}, and what were the key trade-offs and constraints you encountered?"

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
