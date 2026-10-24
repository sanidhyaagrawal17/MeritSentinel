"""Tests for end-to-end interview engine, session store, and mastery synchronization."""

from pathlib import Path
from resumeproof.core.db import Database
from resumeproof.core.llm import FakeLLMClient
from resumeproof.schemas.claims import Claim, ClaimStatus
from resumeproof.store.claim_store import ClaimStore
from resumeproof.rubric.store import RubricStore
from resumeproof.rubric.builder import build_rubrics_for_persona
from resumeproof.interview.generator import QuestionGenerator
from resumeproof.interview.grader import AnswerGrader
from resumeproof.interview.caching import GradingCache
from resumeproof.interview.background import BackgroundGrader
from resumeproof.interview.session_store import SessionStore
from resumeproof.interview.engine import InterviewEngine

DEMO_DIR = Path("data/demo")

def test_interview_engine_session_flow_and_mastery_persistence():
    """Tests complete interview session lifecycle with QuestionGenerator, Grader, and Store."""
    db = Database(":memory:")
    claim_store = ClaimStore(db)
    rubric_store = RubricStore(db)
    session_store = SessionStore(db, claim_store)

    # Seed confirmed claims for sam_okafor
    c1 = Claim(
        id="c_wal",
        user_id="user_test",
        persona_id="sam_okafor",
        project_id="task-tracker-api",
        text="FastAPI REST API with SQLite connection pooling sustaining 350 concurrent read req/sec",
        tools=["SQLite", "SQLAlchemy", "FastAPI"],
        metric_values=["350 req/sec"],
        status=ClaimStatus.CONFIRMED,
    )
    c2 = Claim(
        id="c_dns",
        user_id="user_test",
        persona_id="sam_okafor",
        project_id="mini-dns-resolver",
        text="Recursive UDP DNS resolver resolving queries in under 45ms for cached records",
        tools=["Python sockets", "struct"],
        metric_values=["45ms"],
        status=ClaimStatus.CONFIRMED,
    )
    claim_store.save_claim(c1)
    claim_store.save_claim(c2)

    # Load rubrics
    rubrics = build_rubrics_for_persona("sam_okafor", base_dir=DEMO_DIR)
    rubric_store.save_rubrics(rubrics)

    # Setup LLM and Engine
    fake_client = FakeLLMClient()
    generator = QuestionGenerator(llm_client=fake_client, model="qwen2.5:3b")
    grader = AnswerGrader(llm_client=fake_client, model="qwen2.5:7b")
    cache = GradingCache(db)
    bg_grader = BackgroundGrader(grader=grader, cache=cache)

    engine = InterviewEngine(
        claim_store=claim_store,
        rubric_store=rubric_store,
        session_store=session_store,
        generator=generator,
        background_grader=bg_grader,
    )

    session = engine.start_session(persona_id="sam_okafor", user_id="user_test", budget=3)
    assert session.status == "active"
    assert session.budget == 3

    # Turn 1: Question 1
    q1 = engine.get_next_question(session.id)
    assert q1 is not None
    assert len(q1.question_text) > 0

    strong_ans = (
        "In Task-Tracker API, SQLite can become a write bottleneck, so we enabled WAL (Write-Ahead Logging) "
        "mode and tuned SQLAlchemy connection pooling to 10 connections reaching 350 read req/sec in our locust test."
    )
    turn1 = engine.submit_answer(session.id, q1, strong_ans, wait_for_grade=True)
    assert turn1.grade is not None
    assert turn1.grade.raw_score == 4

    # Check mastery was updated in claim store
    m1 = claim_store.get_mastery_state(q1.claim_id)
    assert m1 is not None
    assert m1.total_answers == 1
    assert m1.aggregate_score == 3.0  # Capped at prior 2.0 + 1.0 jump cap!

    # Turn 2: Question 2
    q2 = engine.get_next_question(session.id)
    assert q2 is not None

    terse_ans = "Enabled WAL mode and SQLAlchemy connection pool of 10 to hit 350 req/sec."
    turn2 = engine.submit_answer(session.id, q2, terse_ans, wait_for_grade=True)
    assert turn2.grade is not None

    # Turn 3: Question 3
    q3 = engine.get_next_question(session.id)
    assert q3 is not None
    turn3 = engine.submit_answer(session.id, q3, strong_ans, wait_for_grade=True)
    assert turn3.grade is not None

    # Turn 4: Should return None because budget of 3 reached
    q4 = engine.get_next_question(session.id)
    assert q4 is None

    # Finalize session
    finalized = engine.finalize_session(session.id)
    assert finalized.status == "completed"
    assert finalized.total_questions == 3
    assert finalized.completed_at is not None

    # Verify session persisted in SQLite
    reloaded = session_store.get_session(session.id)
    assert reloaded is not None
    assert len(reloaded.turns) == 3
    assert reloaded.turns[0].grade.raw_score == 4

    bg_grader.shutdown()
