"""Tests for SQLite Claim Store, dual categories, and ACID persistence."""

import uuid
from datetime import datetime, timezone
import pytest
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus
from resumeproof.schemas.mastery import AnswerRecord, MasteryState

def test_store_profile_facts_and_claims_isolation(claim_store):
    user_id = "test_user_isolation"

    # Category 1: Profile Fact
    fact = ProfileFact(
        id=str(uuid.uuid4()),
        user_id=user_id,
        fact_type=ProfileFactType.NAME,
        value="Sam Okafor",
        source_line=1,
        is_confirmed=True
    )
    claim_store.save_profile_fact(fact)

    # Category 2: Mastery Claim
    claim = Claim(
        id=str(uuid.uuid4()),
        user_id=user_id,
        claim_type=ClaimType.PROJECT_METRIC,
        text="Built REST API sustaining 350 concurrent read req/sec.",
        tools=["FastAPI", "SQLite"],
        metric_values=["350 req/sec"],
        source_line=6,
        status=ClaimStatus.CONFIRMED
    )
    claim_store.save_claim(claim)

    facts = claim_store.get_profile_facts(user_id, only_confirmed=True)
    claims = claim_store.get_claims(user_id)

    assert len(facts) == 1
    assert facts[0].value == "Sam Okafor"

    assert len(claims) == 1
    assert claims[0].text == "Built REST API sustaining 350 concurrent read req/sec."

    # Eligible claims should be empty because claim is not yet verified through interview gate
    eligible = claim_store.get_eligible_claims(user_id)
    assert len(eligible) == 0

def test_claim_mastery_gating_and_eligibility(claim_store):
    user_id = "test_user_mastery"
    claim_id = str(uuid.uuid4())

    claim = Claim(
        id=claim_id,
        user_id=user_id,
        claim_type=ClaimType.SKILL,
        text="Python backend programming with FastAPI and SQLite.",
        tools=["Python", "FastAPI"],
        metric_values=[],
        source_line=18,
        status=ClaimStatus.CONFIRMED
    )
    claim_store.save_claim(claim)

    # Initially ineligible
    initial_state = claim_store.get_mastery_state(claim_id)
    assert initial_state is not None
    assert initial_state.is_eligible is False

    # Simulate passing mastery threshold (effective_score >= 2.5, confidence >= 0.5)
    answer = AnswerRecord(
        id=str(uuid.uuid4()),
        claim_id=claim_id,
        raw_score=4,
        evidence_quote="Candidate explained SQLite WAL mode clearly."
    )
    mastery = MasteryState(
        claim_id=claim_id,
        aggregate_score=3.0,
        effective_score=2.6,
        confidence=0.55,
        n_effective=2.44,
        total_answers=2,
        is_eligible=True,
        last_updated=datetime.now(timezone.utc)
    )
    claim_store.save_answer_and_mastery(answer, mastery)

    # Now verify claim is returned in eligible claims query
    eligible = claim_store.get_eligible_claims(user_id)
    assert len(eligible) == 1
    assert eligible[0].id == claim_id

    # Check answers history
    answers = claim_store.get_answer_records(claim_id)
    assert len(answers) == 1
    assert answers[0].raw_score == 4
    assert answers[0].evidence_quote == "Candidate explained SQLite WAL mode clearly."
