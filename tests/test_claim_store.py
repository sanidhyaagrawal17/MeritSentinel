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


def test_clear_user_data_preserves_confirmed_claims_and_mastery(claim_store):
    user_id = "test_user_clear_safety"

    # Profile facts: 1 confirmed, 1 unconfirmed
    fact_confirmed = ProfileFact(
        id=str(uuid.uuid4()),
        user_id=user_id,
        fact_type=ProfileFactType.NAME,
        value="Sanidhya Agrawal",
        source_line=1,
        is_confirmed=True
    )
    fact_unconfirmed = ProfileFact(
        id=str(uuid.uuid4()),
        user_id=user_id,
        fact_type=ProfileFactType.LOCATION,
        value="Indore, India",
        source_line=2,
        is_confirmed=False
    )
    claim_store.save_profile_fact(fact_confirmed)
    claim_store.save_profile_fact(fact_unconfirmed)

    # Claims: 1 confirmed, 1 unconfirmed (extracted)
    confirmed_claim_id = str(uuid.uuid4())
    claim_confirmed = Claim(
        id=confirmed_claim_id,
        user_id=user_id,
        claim_type=ClaimType.PROJECT_METRIC,
        text="Trained defect inspection pipeline reaching 61.7% mAP50 on NEU-DET.",
        tools=["PyTorch", "YOLO11"],
        metric_values=["61.7% mAP50"],
        source_line=10,
        status=ClaimStatus.CONFIRMED
    )
    extracted_claim_id = str(uuid.uuid4())
    claim_extracted = Claim(
        id=extracted_claim_id,
        user_id=user_id,
        claim_type=ClaimType.SKILL,
        text="Python programming",
        tools=["Python"],
        metric_values=[],
        source_line=15,
        status=ClaimStatus.EXTRACTED
    )
    claim_store.save_claim(claim_confirmed)
    claim_store.save_claim(claim_extracted)

    # Add answer record and mastery state for confirmed claim
    answer = AnswerRecord(
        id=str(uuid.uuid4()),
        claim_id=confirmed_claim_id,
        raw_score=4,
        evidence_quote="Candidate explained anchor-free detection mechanics."
    )
    mastery = MasteryState(
        claim_id=confirmed_claim_id,
        aggregate_score=3.5,
        effective_score=3.2,
        confidence=0.8,
        n_effective=3.0,
        total_answers=3,
        is_eligible=True,
        last_updated=datetime.now(timezone.utc)
    )
    claim_store.save_answer_and_mastery(answer, mastery)

    # Now clear user data (simulating resume re-upload)
    claim_store.clear_user_data(user_id)

    # Verify unconfirmed profile fact was deleted, but confirmed profile fact remains
    facts = claim_store.get_profile_facts(user_id)
    assert len(facts) == 1
    assert facts[0].id == fact_confirmed.id
    assert facts[0].value == "Sanidhya Agrawal"

    # Verify extracted claim was deleted, but confirmed claim remains
    claims = claim_store.get_claims(user_id)
    assert len(claims) == 1
    assert claims[0].id == confirmed_claim_id
    assert claims[0].status == ClaimStatus.CONFIRMED

    # Verify mastery state and answer records are intact
    saved_mastery = claim_store.get_mastery_state(confirmed_claim_id)
    assert saved_mastery is not None
    assert saved_mastery.is_eligible is True
    assert saved_mastery.effective_score == 3.2

    saved_answers = claim_store.get_answer_records(confirmed_claim_id)
    assert len(saved_answers) == 1
    assert saved_answers[0].raw_score == 4

