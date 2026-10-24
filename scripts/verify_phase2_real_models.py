"""Verification script for Phase 2 gate requirements using real Ollama models.
Executes:
1. Extraction on Sam Okafor resume with real qwen2.5:3b.
2. Re-grading of Task-Tracker turn 1 answer with real qwen2.5:7b.
3. Question generation and grading for JWT claim with real models (verifying rubric isolation).
"""

import sys
from pathlib import Path
import json
from collections import Counter

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.core.telemetry import telemetry
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.interview.generator import QuestionGenerator
from resumeproof.interview.grader import AnswerGrader
from resumeproof.rubric.store import RubricStore
from resumeproof.rubric.builder import build_rubrics_for_persona
from resumeproof.core.db import Database
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus
from resumeproof.schemas.rubric import RubricItem
from resumeproof.schemas.interview import QuestionPriority

def run_real_model_verification():
    llm = OpenAICompatibleClient()

    print("==================================================")
    print("STEP 1: REAL EXTRACTION ON SAM OKAFOR RESUME (qwen2.5:3b)")
    print("==================================================")
    pdf_path = Path("data/demo/sam_okafor/resume.pdf")
    parse_result = parse_resume(pdf_path)
    facts, claims = extract_from_resume(
        parse_result,
        llm,
        persona_id="sam_okafor",
        model="qwen2.5:3b"
    )

    print(f"Total Extracted Facts: {len(facts)}")
    for f in facts:
        print(f"  - Fact [{f.fact_type.value}]: {f.value} (line {f.source_line})")

    print(f"\nTotal Deduplicated Claims: {len(claims)}")
    type_counts = Counter(c.claim_type.value for c in claims)
    for ctype, count in type_counts.items():
        print(f"  * {ctype}: {count}")

    print("\nClaims detail:")
    for idx, c in enumerate(claims, start=1):
        print(f"  {idx}. [{c.claim_type.value}] (Project: {c.project_id}, Line: {c.source_line})")
        print(f"     Text: {c.text}")
        print(f"     Tools: {c.tools} | Metrics: {c.metric_values}")

    print("\n==================================================")
    print("STEP 2: RE-GRADE TASK-TRACKER TURN 1 ANSWER (qwen2.5:7b)")
    print("==================================================")
    task_claim = Claim(
        id="c_task_tracker_wal",
        user_id="sam_u",
        persona_id="sam_okafor",
        project_id="task-tracker-api",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Built RESTful API with SQLite connection pooling sustaining 350 concurrent read req/sec.",
        tools=["FastAPI", "SQLite", "Python"],
        metric_values=["350 req/sec"],
        source_line=6,
    )

    wal_rubric = RubricItem(
        id="rubric_task_wal_pooling",
        project_id="task-tracker-api",
        persona_id="sam_okafor",
        topic="sqlite_wal_connection_pooling",
        source_file="data/demo/sam_okafor/projects/task-tracker-api/README.md",
        source_line=5,
        source_quote="Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.",
        criteria={
            0: "Irrelevant, refuses to answer, or contradicts verified facts.",
            1: "Generic mention of SQLite without connection pooling or WAL concurrency details.",
            2: "Mentions WAL or connection pooling but omits how concurrent reads were sustained at 350 req/sec.",
            3: "Accurately articulates SQLite WAL mode and SQLAlchemy connection pooling reaching 350 read req/sec.",
            4: "Deep architectural mastery: explains WAL background checkpointing, concurrent read locks, and pool sizing trade-offs.",
        }
    )

    wal_question = "How did you configure SQLite connection pooling and WAL mode to sustain 350 concurrent read req/sec in Task-Tracker API?"
    wal_answer = (
        "In Task-Tracker API, SQLite can become a write bottleneck, so we enabled WAL (Write-Ahead Logging) "
        "mode and tuned SQLAlchemy connection pooling to 10 connections. For read-heavy operations, this "
        "allowed concurrent reads while WAL checkpointing happened in the background, reaching 350 read "
        "req/sec in our locust test."
    )

    grader = AnswerGrader(llm_client=llm, model="qwen2.5:7b")
    grade_task = grader.grade_answer(
        claim=task_claim,
        question_text=wal_question,
        answer_text=wal_answer,
        rubric=wal_rubric,
    )

    print(f"Task-Tracker Grader Output:")
    print(f"  Raw Score: {grade_task.raw_score} / 4")
    print(f"  Matched Rubric IDs: {grade_task.matched_rubric_ids}")
    print(f"  Missed Rubric IDs: {grade_task.missed_rubric_ids}")
    print(f"  Evidence Quote: {grade_task.evidence_quote}")
    print(f"  Rubric Reference: {grade_task.rubric_reference}")
    print(f"  Feedback: {grade_task.feedback}")
    print(f"  Latency: {grade_task.latency_ms:.1f}ms")

    print("\n==================================================")
    print("STEP 3: JWT QUESTION GENERATION (qwen2.5:3b) & GRADING (qwen2.5:7b)")
    print("==================================================")
    # Build database and rubric store to verify selection
    db = Database(":memory:")
    rubric_store = RubricStore(db)
    sam_rubrics = build_rubrics_for_persona("sam_okafor")
    for r in sam_rubrics:
        rubric_store.save_rubric(r)

    jwt_claim = Claim(
        id="c_jwt_auth",
        user_id="sam_u",
        persona_id="sam_okafor",
        project_id="task-tracker-api",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Implemented JWT authentication and role-based access control.",
        tools=["FastAPI", "JWT", "Python"],
        metric_values=[],
        source_line=7,
    )

    # Verify best rubric match for JWT claim from built persona rubrics
    best_rubric = rubric_store.find_best_rubric_for_claim(jwt_claim)
    print(f"Rubric Selected for JWT Claim:")
    print(f"  ID: {best_rubric.id if best_rubric else 'None'}")
    print(f"  Project ID: {best_rubric.project_id if best_rubric else 'None'}")
    print(f"  Topic: {best_rubric.topic if best_rubric else 'None'}")
    assert best_rubric is not None
    assert best_rubric.project_id == "task-tracker-api", f"LEAKAGE BUG: matched {best_rubric.project_id} instead of task-tracker-api!"
    assert "dns" not in best_rubric.topic.lower(), "LEAKAGE BUG: matched mini-dns-resolver!"

    # Generate question with qwen2.5:3b
    generator = QuestionGenerator(llm_client=llm, model="qwen2.5:3b")
    jwt_question = generator.generate_question(jwt_claim, priority=QuestionPriority.UNTESTED, rubric=best_rubric)
    print(f"\nGenerated Question (qwen2.5:3b):")
    print(f"  Question ID: {jwt_question.id}")
    print(f"  Question Text: {jwt_question.question_text}")
    print(f"  Attached Rubric ID: {jwt_question.rubric_id}")

    # Grade candidate answer with qwen2.5:7b
    candidate_jwt_answer = (
        "In Task-Tracker API, we used python-jose to generate signed HMAC-SHA256 JWT tokens containing "
        "user ID and role scopes with a 30-minute expiration. For role-based access control, we implemented "
        "FastAPI dependency injection (SecurityScopes) on routes, checking whether the decoded token contained "
        "the required role before permitting access to task management endpoints."
    )

    grade_jwt = grader.grade_answer(
        claim=jwt_claim,
        question_text=jwt_question.question_text,
        answer_text=candidate_jwt_answer,
        rubric=best_rubric,
    )

    print(f"\nJWT Grader Output (qwen2.5:7b):")
    print(f"  Raw Score: {grade_jwt.raw_score} / 4")
    print(f"  Matched Rubric IDs: {grade_jwt.matched_rubric_ids}")
    print(f"  Missed Rubric IDs: {grade_jwt.missed_rubric_ids}")
    print(f"  Evidence Quote: {grade_jwt.evidence_quote}")
    print(f"  Rubric Reference: {grade_jwt.rubric_reference}")
    print(f"  Feedback: {grade_jwt.feedback}")
    print(f"  Latency: {grade_jwt.latency_ms:.1f}ms")

    print("\n==================================================")
    print("STEP 4: TELEMETRY CALL LOGS & MODEL DIGESTS")
    print("==================================================")
    telemetry_summary = telemetry.summary()
    print(f"Total LLM Calls: {telemetry_summary['total_calls']}")
    print(f"Total Tokens: {telemetry_summary['total_tokens']}")
    print(f"Total Latency: {telemetry_summary['total_latency_ms']:.1f}ms")
    print(f"Model Digests: {telemetry_summary['model_digests']}")
    print("Per-call breakdown:")
    for call in telemetry.records:
        digest_str = f"{call.digest[:16]}..." if call.digest else "None"
        print(f"  - [{call.timestamp}] Model: {call.model} (Digest: {digest_str}) | Latency: {call.latency_ms:.1f}ms | Tokens: {call.total_tokens}")

if __name__ == "__main__":
    run_real_model_verification()
