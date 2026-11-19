import re
import time
import uuid
from typing import Optional
from pydantic import BaseModel, Field

from resumeproof.core.config import settings
from resumeproof.core.llm import LLMClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem
from resumeproof.schemas.interview import InterviewQuestion, QuestionPriority

QUESTION_SYSTEM_PROMPT = """You are an approachable, fair, and thorough technical interviewer interviewing an entry-level software engineering candidate (college graduate / junior developer).
Your goal is to verify that the candidate personally implemented their resume projects by asking clear, accessible questions about their hands-on coding and project implementation.

CRITICAL RULES:
1. Calibrate difficulty strictly for a college graduate / junior developer. NEVER ask senior-architect systems design questions (e.g. enterprise high-availability, scaling to tens of thousands of queries/sec, distributed bottlenecks, complex memory management) unless that exact metric is written verbatim in the candidate's claim.
2. Focus on practical, understandable engineering: What libraries or frameworks did you use? How was the database table, SQL query, or API endpoint structured? How did you test or verify it?
3. Questions MUST come ONLY from the supplied claim and atomic rubric points.
4. Keep questions direct, conversational, and under 2 sentences.
5. Do NOT ask for abstract senior trade-offs or hypothetical alternatives unless explicitly mentioned in the atomic points.
6. If asking a follow-up, provide an approachable prompt (e.g. "Can you walk through what that function or query looked like in Python/SQL?") to give the candidate an opportunity to explain their code.
"""

REPHRASE_SYSTEM_PROMPT = """You are an approachable, clear technical interviewer.
A candidate found the following technical question confusing, wordy, or overly academic.
Your task is to rephrase and simplify the question so it is direct, plain-English, and conversational (1-2 sentences maximum).

CRITICAL RULES:
1. Preserve the exact technical core: probe the same project, library, query, or mechanism from the candidate's claim.
2. Simplify the language: eliminate unnecessary academic jargon, overly convoluted preamble, and confusing clause chains.
3. Keep it approachable for an entry-level / junior software developer.
4. Output ONLY the rephrased question text. Do NOT include greetings, conversational filler, quotation marks, or explanations.
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

def is_duplicate_or_similar(new_question: str, past_questions: list[str], threshold: float = 0.65) -> bool:
    """Checks if new question is identical or has high lexical overlap with any past question."""
    if not past_questions:
        return False
    new_norm = re.sub(r"[^a-zA-Z0-9\s]", "", new_question.lower()).strip()
    new_tokens = {w for w in new_norm.split() if len(w) > 3 and w not in STOPWORDS}
    for past in past_questions:
        past_norm = re.sub(r"[^a-zA-Z0-9\s]", "", past.lower()).strip()
        if new_norm == past_norm:
            return True
        past_tokens = {w for w in past_norm.split() if len(w) > 3 and w not in STOPWORDS}
        if new_tokens and past_tokens:
            overlap = new_tokens.intersection(past_tokens)
            sim = len(overlap) / max(len(new_tokens), len(past_tokens))
            if sim >= threshold:
                return True
    return False

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
        past_questions: Optional[list[str]] = None,
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
                "\nCRITICAL INSTRUCTION: You MUST generate an accessible, entry-level technical interview question answerable strictly from these atomic points.",
                "STRICT BOUNDARY: Do NOT ask for senior-architect trade-offs, distributed scale, or enterprise constraints unless an atomic point explicitly mentions one.",
                "Incorporate specific technical mechanisms, functions, or tools directly from the atomic points above in an approachable way.",
            ])

        if past_questions:
            past_str = "\n".join(f"- {q}" for q in past_questions[-5:])
            prompt_parts.extend([
                f"\nPREVIOUSLY ASKED QUESTIONS IN THIS SESSION (FORBIDDEN TO REPEAT):",
                past_str,
                "CRITICAL DEDUPLICATION RULE: You must NEVER repeat, rephrase, or re-ask any previously asked question listed above. You must probe a completely different mechanism, parameter, or implementation step."
            ])

        if is_followup and previous_question and previous_answer:
            prompt_parts.extend([
                f"\nPrevious Question: {previous_question}",
                f"Candidate's Previous Answer: {previous_answer}",
                f"Follow-up Focus: {followup_reason or 'Guide the candidate to describe their practical implementation steps.'}",
                "Generate a clear, constructive follow-up question encouraging the candidate to explain what code, library, or steps they used."
            ])
        else:
            prompt_parts.append(
                "Generate an approachable, direct technical question probing how the candidate wrote, implemented, or tested these specific tools and parameters in their project."
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

        # Strict deduplication check against session history
        if past_questions and is_duplicate_or_similar(clean_q, past_questions):
            unused_pts = [
                pt for pt in (rubric.atomic_points if rubric else [])
                if not any(pt.text.lower() in pq.lower() for pq in past_questions)
            ]
            if unused_pts:
                clean_q = f"Could you walk through how you handled {unused_pts[0].text} in your project?"
            elif is_followup:
                clean_q = f"What specific test cases, validation checks, or scripts did you write to verify {claim.text}?"
            else:
                clean_q = f"Could you explain the concrete implementation steps you took for {claim.text}?"

        # Fallback if response was too verbose or empty
        if not clean_q or len(clean_q) < 10:
            if rubric:
                clean_q = f"Could you walk through how you implemented {rubric.source_quote} in {claim.text}?"
            elif is_followup:
                clean_q = f"Could you provide a few more details on how you wrote the code for {claim.text}?"
            else:
                clean_q = f"How did you implement and test {claim.text} in your project?"

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

    def rephrase_question(
        self,
        current_question: InterviewQuestion,
        claim: Claim,
        rubric: Optional[RubricItem] = None,
    ) -> InterviewQuestion:
        """Rephrases and simplifies an interview question on user request for greater clarity."""
        t0 = time.time()
        prompt_parts = [
            f"Original Question: {current_question.question_text}",
            f"Candidate Claim: {claim.text}",
            f"Tools Mentioned: {', '.join(claim.tools) if claim.tools else 'None'}",
            f"Metrics Mentioned: {', '.join(claim.metric_values) if claim.metric_values else 'None'}",
        ]
        if rubric:
            prompt_parts.append(f"Rubric Topic: {rubric.topic}")

        prompt_parts.append(
            "Please rephrase the original question in simpler, more direct terms so the candidate clearly understands what implementation details to explain."
        )

        messages = [
            {"role": "system", "content": REPHRASE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(prompt_parts)},
        ]

        try:
            raw_rephrase = self.llm_client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.2,
                timeout=30.0,
            )
            clean_rephrase = raw_rephrase.strip().strip('"').strip("'")
            clean_rephrase = re.sub(
                r"^(?:rephrased\s+question|simplified\s+question|here\s+is\s+the\s+rephrased\s+question|here\s+is\s+a\s+simpler\s+version|question)\s*:\s*",
                "",
                clean_rephrase,
                flags=re.IGNORECASE,
            ).strip()
            if check_question_outside_points(clean_rephrase, rubric):
                clean_rephrase = strip_outside_clauses(clean_rephrase)
        except Exception:
            clean_rephrase = ""

        if not clean_rephrase or len(clean_rephrase) < 10:
            if rubric:
                clean_rephrase = f"In simple terms, could you explain how you designed and tested {rubric.topic} in {claim.text}?"
            else:
                clean_rephrase = f"In simple terms, could you explain how you implemented and tested {claim.text}?"

        elapsed_ms = (time.time() - t0) * 1000.0

        return InterviewQuestion(
            id=str(uuid.uuid4()),
            claim_id=current_question.claim_id,
            project_id=current_question.project_id,
            rubric_id=current_question.rubric_id,
            priority=current_question.priority,
            question_text=clean_rephrase,
            is_followup=current_question.is_followup,
            followup_reason=current_question.followup_reason,
            is_rephrased=True,
            original_question_text=current_question.original_question_text or current_question.question_text,
            latency_ms=elapsed_ms,
        )
