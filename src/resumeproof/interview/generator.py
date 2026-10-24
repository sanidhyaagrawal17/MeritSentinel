"""Question generator using qwen2.5:3b to produce rigorous technical interview questions."""

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
2. Ground your inquiry in the stated claim and project context.
3. If this is a follow-up, drill directly into what was vague or unverified in their previous answer.
4. Do NOT lecture, praise, or reveal the answer. Keep questions direct and under 3 sentences.
"""

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
        """Generates an interview question for the given claim."""
        prompt_parts = [
            f"Candidate Claim: {claim.text}",
            f"Tools Mentioned: {', '.join(claim.tools) if claim.tools else 'None'}",
            f"Metrics Mentioned: {', '.join(claim.metric_values) if claim.metric_values else 'None'}",
        ]
        if rubric:
            prompt_parts.append(f"Project Reference: {rubric.source_quote} ({rubric.source_file}:{rubric.source_line})")

        if is_followup and previous_question and previous_answer:
            prompt_parts.extend([
                f"\nPrevious Question: {previous_question}",
                f"Candidate's Previous Answer: {previous_answer}",
                f"Follow-up Focus: {followup_reason or 'Probe for concrete architectural details.'}",
                "Generate a sharp, focused follow-up question pressing for specific implementation details and verified parameters."
            ])
        else:
            prompt_parts.append(
                "Generate an opening technical question challenging the candidate on how they implemented this and handled constraints."
            )

        messages = [
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(prompt_parts)},
        ]

        # Use chat_completion with 3B model
        raw_question = self.llm_client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=0.2,
            timeout=60.0
        )

        # Clean question string
        clean_q = raw_question.strip().strip('"').strip("'")

        # Fallback if response was too verbose or empty
        if not clean_q or len(clean_q) < 10:
            if is_followup:
                clean_q = f"Could you provide more concrete implementation details regarding {claim.text}?"
            else:
                clean_q = f"How did you implement {claim.text}, and what were the key trade-offs and constraints you encountered?"

        return InterviewQuestion(
            id=str(uuid.uuid4()),
            claim_id=claim.id,
            project_id=claim.project_id or (rubric.project_id if rubric else None),
            priority=priority,
            question_text=clean_q,
            is_followup=is_followup,
            followup_reason=followup_reason,
        )
