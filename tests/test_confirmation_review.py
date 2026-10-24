"""Tests for the mandatory user confirmation & review step and audit log recording."""

import uuid
from resumeproof.schemas.facts import ProfileFact, ProfileFactType
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus
from resumeproof.review.flow import ReviewSession, ReviewDecision

def test_confirm_profile_fact(review_session, claim_store):
    fact = ProfileFact(
        id=str(uuid.uuid4()),
        user_id="user_1",
        fact_type=ProfileFactType.NAME,
        value="Alex Rivera",
        source_line=1
    )
    # User confirms fact
    reviewed = review_session.review_profile_fact(fact, confirm=True)
    assert reviewed.is_confirmed is True
    assert reviewed.confirmed_at is not None

    # Check store
    stored = claim_store.get_profile_facts("user_1", only_confirmed=True)
    assert len(stored) == 1
    assert stored[0].value == "Alex Rivera"

def test_edit_profile_fact_correction(review_session, claim_store):
    fact = ProfileFact(
        id=str(uuid.uuid4()),
        user_id="user_1",
        fact_type=ProfileFactType.CGPA,
        value="3.7 / 4.0",
        source_line=3
    )
    # User corrects CGPA to 3.8
    reviewed = review_session.review_profile_fact(fact, confirm=True, correction="3.8 / 4.0")
    assert reviewed.is_confirmed is True
    assert reviewed.user_correction == "3.8 / 4.0"

    stored = claim_store.get_profile_facts("user_1", only_confirmed=True)
    assert stored[0].user_correction == "3.8 / 4.0"
    assert "3.8" in stored[0].allowed_tokens()

def test_confirm_claim_as_is(review_session, claim_store):
    claim = Claim(
        id=str(uuid.uuid4()),
        user_id="user_1",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Sustained 45,000 req/sec write throughput.",
        tools=["Go", "Raft"],
        metric_values=["45,000 req/sec"],
        source_line=11
    )
    reviewed = review_session.review_claim(claim, ReviewDecision(action=ClaimStatus.CONFIRMED))

    assert reviewed.status == ClaimStatus.CONFIRMED
    assert reviewed.audit is not None
    assert reviewed.audit.action_taken == ClaimStatus.CONFIRMED
    assert reviewed.audit.extractor_text == "Sustained 45,000 req/sec write throughput."
    assert reviewed.audit.user_correction_text is None

    # Verify audit trail in SQLite
    audit = claim_store.get_audit_trail(claim.id)
    assert audit is not None
    assert audit.action_taken == ClaimStatus.CONFIRMED
    assert audit.extractor_metrics == ["45,000 req/sec"]

def test_edit_claim_with_user_correction(review_session, claim_store):
    # Candidate corrects inflated claim to true verified number
    claim = Claim(
        id=str(uuid.uuid4()),
        user_id="user_1",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Handled 100,000 req/sec write throughput.",
        tools=["Go", "Raft"],
        metric_values=["100,000 req/sec"],
        source_line=12
    )

    decision = ReviewDecision(
        action=ClaimStatus.EDITED,
        user_text="Handled 45,000 req/sec write throughput under 3-node benchmark.",
        user_tools=["Go", "Raft"],
        user_metrics=["45,000 req/sec"],
        notes="Downscoped from inflated 100k claim to verified 45k metric."
    )

    reviewed = review_session.review_claim(claim, decision)

    assert reviewed.status == ClaimStatus.EDITED
    assert reviewed.text == "Handled 45,000 req/sec write throughput under 3-node benchmark."
    assert reviewed.effective_text() == "Handled 45,000 req/sec write throughput under 3-node benchmark."
    assert reviewed.effective_metrics() == ["45,000 req/sec"]

    # Verify audit record records BOTH the extractor output AND user correction
    audit = claim_store.get_audit_trail(claim.id)
    assert audit is not None
    assert audit.action_taken == ClaimStatus.EDITED
    assert audit.extractor_text == "Handled 100,000 req/sec write throughput."
    assert audit.extractor_metrics == ["100,000 req/sec"]
    assert audit.user_correction_text == "Handled 45,000 req/sec write throughput under 3-node benchmark."
    assert audit.user_correction_metrics == ["45,000 req/sec"]
    assert audit.notes == "Downscoped from inflated 100k claim to verified 45k metric."

def test_reject_claim_audit(review_session, claim_store):
    # Candidate rejects planted hallucination / unsupported claim
    claim = Claim(
        id=str(uuid.uuid4()),
        user_id="user_1",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Built Paxos consensus engine with Istio service mesh.",
        tools=["Paxos", "Istio"],
        metric_values=[],
        source_line=13
    )

    decision = ReviewDecision(
        action=ClaimStatus.REJECTED,
        notes="Never implemented Paxos; remove from interview pool."
    )

    reviewed = review_session.review_claim(claim, decision)
    assert reviewed.status == ClaimStatus.REJECTED

    # Ensure store records it as rejected
    claims = claim_store.get_claims("user_1", statuses=[ClaimStatus.REJECTED])
    assert len(claims) == 1
    assert claims[0].id == claim.id

    audit = claim_store.get_audit_trail(claim.id)
    assert audit.action_taken == ClaimStatus.REJECTED
    assert audit.extractor_text == "Built Paxos consensus engine with Istio service mesh."
