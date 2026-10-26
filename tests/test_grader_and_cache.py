"""Tests for AnswerGrader, zero-tier-label invariant, caching, and background worker."""

import pytest
from resumeproof.core.db import Database
from resumeproof.core.llm import FakeLLMClient
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.rubric import RubricItem
from resumeproof.interview.grader import AnswerGrader
from resumeproof.interview.caching import GradingCache
from resumeproof.interview.background import BackgroundGrader

def test_grader_zero_tier_label_invariant():
    """CRITICAL INVARIANT: The grader must never see any answer-quality tier labels.

    NOTE: This test uses FakeLLMClient to test plumbing and prompt construction mechanics,
    NOT grader quality or real LLM scoring accuracy.
    """
    client = FakeLLMClient()
    grader = AnswerGrader(llm_client=client, model="qwen2.5:7b")

    claim = Claim(
        id="c_test",
        user_id="user_test",
        text="FastAPI REST API with SQLite connection pooling sustaining 350 concurrent read req/sec",
    )
    rubric = RubricItem(
        project_id="task-tracker-api",
        topic="sqlite_wal",
        source_file="README.md",
        source_line=5,
        source_quote="Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.",
        criteria={
            0: "Irrelevant or hallucinated claims.",
            1: "Surface-level without parameters.",
            2: "Partial understanding of pooling.",
            3: "Accurate WAL mode and 350 req/sec benchmark.",
            4: "Deep architectural trade-offs of single-writer multi-reader.",
        }
    )

    question = "How did you scale SQLite to handle 350 concurrent read requests per second?"
    answer = "In Task-Tracker API, we enabled WAL mode and tuned SQLAlchemy connection pooling to 10 connections reaching 350 read req/sec in our locust test."

    grade = grader.grade_answer(claim=claim, question_text=question, answer_text=answer, rubric=rubric)

    # Inspect all messages sent to the LLM client
    assert len(client.call_history) > 0
    last_call = client.call_history[-1]
    all_content = " ".join(m["content"] for m in last_call["messages"])

    # Strict assertion: The word "tier" MUST NOT appear anywhere in the prompt!
    assert "tier" not in all_content.lower(), f"Grader prompt violated invariant by containing tier labels: {all_content}"

    # Check valid grade output
    assert 0 <= grade.raw_score <= 4
    assert len(grade.evidence_quote) > 0
    assert len(grade.feedback) > 0

def test_grader_scores_trap_answer_zero():
    """Verifies that an unsupported trap answer receives raw_score = 0.

    NOTE: This test uses FakeLLMClient to test plumbing and pipeline mechanics,
    NOT grader quality or real LLM scoring accuracy.
    """
    client = FakeLLMClient()
    grader = AnswerGrader(llm_client=client, model="qwen2.5:7b")

    claim = Claim(id="c_trap", user_id="u", text="Junior API with SQLite")
    trap_answer = "We used an Apache Kafka event bus and BGP scrubbing centers to stream financial transactions into our Byzantine fault-tolerant Raft cluster with zero downtime."

    grade = grader.grade_answer(
        claim=claim,
        question_text="How did you build the event pipeline?",
        answer_text=trap_answer,
    )

    assert grade.raw_score == 0
    assert "Kafka" in grade.evidence_quote or "financial" in grade.evidence_quote

def test_grading_cache_hit_avoids_recomputation():
    """Verifies that duplicate answers are returned from cache with cached=True.

    NOTE: This test uses FakeLLMClient to test plumbing and cache mechanics,
    NOT grader quality.
    """
    db = Database(":memory:")
    cache = GradingCache(db)
    client = FakeLLMClient()
    grader = AnswerGrader(llm_client=client, model="qwen2.5:7b")

    claim = Claim(id="c_cache", user_id="u", text="SQLite scaling")
    q = "How does WAL mode work?"
    a = "WAL allows concurrent readers while writers write sequentially."

    # First call: cache miss
    assert cache.get(claim.id, q, a, grader.model) is None
    res1 = grader.grade_answer(claim, q, a)
    cache.set(claim.id, q, a, grader.model, res1)

    # Second call: cache hit
    res2 = cache.get(claim.id, q, a, grader.model)
    assert res2 is not None
    assert res2.cached is True
    assert res2.raw_score == res1.raw_score
    assert res2.latency_ms == 0.0

def test_background_grader_async_execution_and_drain():
    """Verifies background grading runs asynchronously on thread pool and drains cleanly.

    NOTE: This test uses FakeLLMClient to test plumbing and background worker mechanics,
    NOT grader quality.
    """
    db = Database(":memory:")
    cache = GradingCache(db)
    client = FakeLLMClient()
    grader = AnswerGrader(llm_client=client, model="qwen2.5:7b")
    bg = BackgroundGrader(grader=grader, cache=cache, max_workers=2)

    claim = Claim(id="c_bg", user_id="u", text="Background task test")
    q = "Explain SQLite pooling."
    a = "SQLAlchemy async pool with pool_size=10."

    completed_grades = []
    future = bg.grade_async(
        task_id="task_1",
        claim=claim,
        question_text=q,
        answer_text=a,
        on_complete=lambda g: completed_grades.append(g)
    )

    # Wait for completion
    grade = future.result(timeout=5.0)
    assert grade.raw_score in (0, 1, 2, 3, 4)
    assert len(completed_grades) == 1

    # Draining completed pool
    drained = bg.drain_all()
    assert isinstance(drained, dict)

    bg.shutdown(wait=True)

def test_confident_wrong_topic_answer_scores_zero_against_rubric():
    """Verifies that a confident, well-written answer about the wrong topic scores 0 or 1 against the asked rubric.

    Scenario:
    Question asks about Task-Tracker API JWT authentication and role-based access control.
    Candidate responds with a confident, technically detailed answer about Raft consensus and Byzantine fault tolerance.
    The grader MUST evaluate strictly against the supplied JWT rubric:
    - Raw score must be 0 or 1 (specifically 0 for off-topic/unsubstantiated claims).
    - Matched rubric IDs must be empty.
    - The JWT rubric item ID must be listed in missed_rubric_ids.
    - Feedback must cite that the answer addresses an unrelated topic.

    NOTE: This test uses FakeLLMClient to test pipeline wiring and schema validation,
    NOT grader quality or real LLM scoring accuracy.
    """
    client = FakeLLMClient()
    grader = AnswerGrader(llm_client=client, model="qwen2.5:7b")

    claim = Claim(
        id="c_jwt",
        user_id="u_sam",
        persona_id="sam_okafor",
        project_id="task-tracker-api",
        text="Implemented JWT authentication and role-based access control.",
    )
    rubric = RubricItem(
        id="rubric_jwt_42",
        project_id="task-tracker-api",
        topic="jwt_authentication",
        source_file="README.md",
        source_line=8,
        source_quote="Implemented JWT authentication and role-based access control.",
        criteria={
            0: "Irrelevant, refuses to answer, or makes claims that contradict verified facts.",
            1: "Generic explanation of auth tokens without implementation mechanics.",
            2: "Mentions JWT payload signing but omits role-based access control.",
            3: "Accurately articulates JWT token generation, expiration, and RBAC middleware.",
            4: "Expert mastery explaining token revocation, HMAC-SHA256 vs RS256 trade-offs, and RBAC scopes.",
        }
    )

    question_text = "How did you implement JWT authentication and verify user roles in Task-Tracker API?"
    # A confident, well-written technical answer about the completely wrong topic (Raft consensus)
    confident_wrong_topic_answer = (
        "We implemented a distributed Raft consensus cluster with Python asyncio sockets, "
        "electing leaders via randomized election timeouts and replicating log entries across "
        "5 quorum nodes with Byzantine fault tolerance."
    )

    grade = grader.grade_answer(
        claim=claim,
        question_text=question_text,
        answer_text=confident_wrong_topic_answer,
        rubric=rubric,
    )

    # Invariant assertions
    assert grade.raw_score in (0, 1), f"Expected score 0 or 1 for off-topic answer, got {grade.raw_score}"
    assert grade.raw_score == 0
    assert rubric.id in grade.missed_rubric_ids, f"Expected {rubric.id} in missed_rubric_ids, got {grade.missed_rubric_ids}"
    assert rubric.id not in grade.matched_rubric_ids
    assert len(grade.matched_rubric_ids) == 0
    assert len(grade.evidence_quote) > 0
    assert len(grade.feedback) > 0


def test_grader_numeric_and_alias_quote_recovery():
    """Validates numeric and synonym alias recovery (0.9999 <-> 99.99%, [150, 300] <-> 150-300ms, dict <-> dictionary)."""
    from resumeproof.interview.grader import resolve_alias_or_numeric_quote

    # 1. 0.9999 -> 99.99%
    ans1 = "We sustained 45,000 req/sec maintaining 99.99% linearizability under Raft consensus."
    recovered1 = resolve_alias_or_numeric_quote("0.9999", ans1)
    assert recovered1 == "99.99%"
    assert recovered1 in ans1

    # 2. [150, 300] -> 150-300ms
    ans2 = "Tuned heartbeat election timeouts to 150-300ms to balance stability and failover speed."
    recovered2 = resolve_alias_or_numeric_quote("[150, 300]", ans2)
    assert recovered2 == "150-300ms"
    assert recovered2 in ans2

    # 3. dict -> dictionary
    ans3 = "Implemented an in-memory dictionary caching recently decoded packet headers."
    recovered3 = resolve_alias_or_numeric_quote("dict", ans3)
    assert recovered3 == "dictionary"
    assert recovered3 in ans3

    # 4. Unmatched alias returns None without false positive
    ans4 = "Configured Cassandra cluster across three availability zones."
    recovered4 = resolve_alias_or_numeric_quote("0.9999", ans4)
    assert recovered4 is None

