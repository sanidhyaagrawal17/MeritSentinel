"""Tests for interview question planning, priority ordering, budget, and follow-ups."""

from datetime import datetime, timezone, timedelta
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.mastery import MasteryState
from resumeproof.schemas.interview import QuestionPriority
from resumeproof.interview.planner import QuestionPlanner

def test_planner_priority_untested_over_weak_over_stale():
    """Verifies that QuestionPlanner strictly follows untested > weak > stale priority order."""
    now = datetime.now(timezone.utc)

    # Claim 1: Untested (0 answers)
    c_untested = Claim(id="c_untested", user_id="u", text="Untested claim")
    s_untested = MasteryState(claim_id="c_untested", total_answers=0, aggregate_score=2.0, effective_score=2.0, confidence=0.0, is_eligible=False, last_updated=now)

    # Claim 2: Weak (tested but effective_score < 2.5)
    c_weak = Claim(id="c_weak", user_id="u", text="Weak claim")
    s_weak = MasteryState(claim_id="c_weak", total_answers=1, aggregate_score=1.5, effective_score=1.8, confidence=0.33, is_eligible=False, last_updated=now - timedelta(days=2))

    # Claim 3: Stale (eligible, but tested 10 days ago)
    c_stale = Claim(id="c_stale", user_id="u", text="Stale eligible claim")
    s_stale = MasteryState(claim_id="c_stale", total_answers=3, aggregate_score=3.5, effective_score=3.2, confidence=0.6, is_eligible=True, last_updated=now - timedelta(days=10))

    claims = [c_stale, c_weak, c_untested]  # Passed in non-priority order
    mastery = {
        "c_untested": s_untested,
        "c_weak": s_weak,
        "c_stale": s_stale,
    }

    planner = QuestionPlanner(claims=claims, mastery_states=mastery, budget=5)

    # Turn 1 must pick untested
    p1 = planner.plan_next_claim()
    assert p1 is not None
    claim1, prio1, is_fu1, _ = p1
    assert claim1.id == "c_untested"
    assert prio1 == QuestionPriority.UNTESTED
    assert is_fu1 is False

    # Simulate updated mastery for c_untested making it strong
    mastery["c_untested"] = MasteryState(claim_id="c_untested", total_answers=2, aggregate_score=3.5, effective_score=3.0, confidence=0.5, is_eligible=True, last_updated=now)

    # Turn 2 must pick weak (using substantive answer >= 15 words to avoid terse follow-up)
    substantive_ans1 = "In Task-Tracker API we enabled WAL mode with SQLAlchemy connection pooling to 10 connections sustaining 350 req/sec in our locust test benchmarks."
    p2 = planner.plan_next_claim(last_answer=substantive_ans1)
    assert p2 is not None
    claim2, prio2, is_fu2, _ = p2
    assert claim2.id == "c_weak"
    assert prio2 == QuestionPriority.WEAK
    assert is_fu2 is False

    # Simulate updated mastery for c_weak making it strong
    mastery["c_weak"] = MasteryState(claim_id="c_weak", total_answers=3, aggregate_score=3.5, effective_score=3.0, confidence=0.6, is_eligible=True, last_updated=now)

    # Turn 3 must pick stale
    substantive_ans2 = "Comprehensive explanation of distributed caching and WAL checkpoints sustaining high concurrent throughput and robust fault injection verification."
    p3 = planner.plan_next_claim(last_answer=substantive_ans2)
    assert p3 is not None
    claim3, prio3, is_fu3, _ = p3
    assert claim3.id == "c_stale"
    assert prio3 == QuestionPriority.STALE

def test_planner_triggers_followup_on_vague_or_terse_answer():
    """Verifies that QuestionPlanner triggers a follow-up when candidate gives a vague or terse answer."""
    c1 = Claim(id="c1", user_id="u", text="FastAPI SQLite read scaling")
    planner = QuestionPlanner(claims=[c1], mastery_states={}, budget=3)

    # Turn 1: Initial question
    p1 = planner.plan_next_claim()
    assert p1[2] is False  # Not follow-up

    # Candidate gives terse answer (< 15 words)
    terse_ans = "We used SQLite with connection pooling."
    p2 = planner.plan_next_claim(last_answer=terse_ans)
    assert p2 is not None
    claim2, prio2, is_fu2, reason2 = p2
    assert claim2.id == "c1"
    assert is_fu2 is True
    assert "terse" in reason2.lower()

    # Second answer given to follow-up: should NOT trigger a second follow-up (max 1 follow-up cap)
    p3 = planner.plan_next_claim(last_answer="Still terse.")
    # Either returns None or moves to next claim because max follow-up per claim is 1
    if p3 is not None:
        assert p3[2] is False

def test_planner_enforces_question_budget():
    """Verifies that QuestionPlanner stops immediately once budget is reached."""
    claims = [Claim(id=f"c{i}", user_id="u", text=f"Claim {i}") for i in range(10)]
    planner = QuestionPlanner(claims=claims, mastery_states={}, budget=3)

    assert planner.plan_next_claim() is not None
    assert planner.plan_next_claim(last_answer="Solid detailed technical answer.") is not None
    assert planner.plan_next_claim(last_answer="Another solid answer with parameters.") is not None

    # Budget of 3 is reached
    assert planner.plan_next_claim(last_answer="Next answer.") is None
    assert planner.is_budget_exhausted() is True

def test_planner_rotates_across_multiple_untested_claims():
    """Verifies that QuestionPlanner does not ask the same claim repeatedly when multiple untested claims exist."""
    c1 = Claim(id="c1", user_id="u", text="SQLite WAL scaling")
    c2 = Claim(id="c2", user_id="u", text="Mini DNS UDP resolver")
    c3 = Claim(id="c3", user_id="u", text="Raw socket packet sniffer")

    planner = QuestionPlanner(claims=[c1, c2, c3], mastery_states={}, budget=3)

    p1 = planner.plan_next_claim()
    assert p1 is not None
    assert p1[0].id == "c1"

    # Non-terse answer (>= 15 words) to avoid follow-up
    p2 = planner.plan_next_claim(last_answer="In Task-Tracker API we tuned connection pooling to 10 connections and enabled WAL mode for locust benchmarking.")
    assert p2 is not None
    assert p2[0].id == "c2", f"Expected c2 to be picked next among untested claims, got {p2[0].id}"

    p3 = planner.plan_next_claim(last_answer="In Mini-DNS-Resolver cached DNS queries resolve via in-memory dictionary under 45ms while uncached queries parse RFC 1035 headers.")
    assert p3 is not None
    assert p3[0].id == "c3", f"Expected c3 to be picked next among untested claims, got {p3[0].id}"

def test_planner_update_mastery_state():
    """Verifies that update_mastery_state dynamically updates planner prioritization."""
    c1 = Claim(id="c1", user_id="u", text="Claim 1")
    planner = QuestionPlanner(claims=[c1], mastery_states={}, budget=3)

    p1 = planner.plan_next_claim()
    assert p1[1] == QuestionPriority.UNTESTED

    now = datetime.now(timezone.utc)
    new_state = MasteryState(
        claim_id="c1", total_answers=2, aggregate_score=3.5,
        effective_score=3.0, confidence=0.6, is_eligible=True, last_updated=now
    )
    planner.update_mastery_state("c1", new_state)

    assert planner.mastery_states["c1"].is_eligible is True
    assert planner.mastery_states["c1"].effective_score == 3.0
