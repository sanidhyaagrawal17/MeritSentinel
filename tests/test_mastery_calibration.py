"""Calibration test on demo personas showing claim eligibility progression across 1, 2, and 3 interview answers."""

from datetime import datetime, timezone, timedelta
from resumeproof.core.llm import FakeLLMClient
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.schemas.claims import ClaimStatus
from resumeproof.schemas.mastery import AnswerRecord, MasteryState
from resumeproof.mastery.calculator import compute_mastery_state
from pathlib import Path

DEMO_DIR = Path("data/demo")

def test_mastery_calibration_across_personas():
    """Simulates 1, 2, and 3 interview answers per claim across all 3 demo personas and tracks eligibility count."""
    client = FakeLLMClient()
    personas = ["alex_rivera", "priya_nair", "sam_okafor"]

    # Target stats to record for reporting
    calibration_stats = {}

    for persona in personas:
        parsed = parse_resume(DEMO_DIR / persona / "resume.pdf")
        facts, claims = extract_from_resume(parsed, client, persona_id=persona)
        assert len(claims) > 0

        # Map claims to performance profiles based on keywords:
        # Supported claims get strong answers (4, 4, 4)
        # Partial claims get mediocre answers (2, 2, 2)
        # Unsupported traps get failing answers (0, 1, 0)
        now = datetime.now(timezone.utc)

        eligible_after_1 = 0
        eligible_after_2 = 0
        eligible_after_3 = 0

        for claim in claims:
            txt_lower = claim.text.lower()
            if any(k in txt_lower for k in ["paxos", "70b", "byzantine", "ddos", "microservices", "100,000", "10 tb"]):
                # Trap or partial: lower scores
                if any(k in txt_lower for k in ["paxos", "70b", "byzantine"]):
                    scores = [0, 0, 1]  # Unsupported trap
                else:
                    scores = [2, 2, 2]  # Partial claim
            else:
                scores = [4, 4, 4]      # Supported claim

            # Answer 1
            ans1 = AnswerRecord(id="a1", claim_id=claim.id, raw_score=scores[0], timestamp=now)
            s1 = compute_mastery_state(claim.id, [ans1], evaluation_time=now)
            if s1.is_eligible:
                eligible_after_1 += 1

            # Answer 2
            ans2 = AnswerRecord(id="a2", claim_id=claim.id, raw_score=scores[1], timestamp=now + timedelta(minutes=5))
            s2 = compute_mastery_state(claim.id, [ans1, ans2], previous_state=s1, evaluation_time=now + timedelta(minutes=5))
            if s2.is_eligible:
                eligible_after_2 += 1

            # Answer 3
            ans3 = AnswerRecord(id="a3", claim_id=claim.id, raw_score=scores[2], timestamp=now + timedelta(minutes=10))
            s3 = compute_mastery_state(claim.id, [ans1, ans2, ans3], previous_state=s2, evaluation_time=now + timedelta(minutes=10))
            if s3.is_eligible:
                eligible_after_3 += 1

        calibration_stats[persona] = {
            "total_claims": len(claims),
            "eligible_after_1": eligible_after_1,
            "eligible_after_2": eligible_after_2,
            "eligible_after_3": eligible_after_3,
        }

    # Invariants to verify:
    # 1. After 1 answer, confidence is 1.0/(1.0+2.0) = 0.333 < 0.5. Exactly 0 claims should be eligible!
    for p, stat in calibration_stats.items():
        assert stat["eligible_after_1"] == 0, f"{p} had eligible claims after only 1 answer (confidence should be < 0.5)"

    # 2. After 2 answers, confidence reaches 2.0/(2.0+2.0) = 0.50. Supported claims become eligible!
    for p, stat in calibration_stats.items():
        assert stat["eligible_after_2"] > 0, f"{p} had no eligible claims after 2 solid answers"
        assert stat["eligible_after_2"] < stat["total_claims"], f"{p} admitted all claims (including unsupported traps!)"

    # 3. After 3 answers, confidence reaches 3.0/(3.0+2.0) = 0.60. Supported claims solidify eligibility.
    for p, stat in calibration_stats.items():
        assert stat["eligible_after_3"] >= stat["eligible_after_2"]
        # Unsupported traps must STILL be blocked
        assert stat["eligible_after_3"] < stat["total_claims"]

    print("\nCalibration Results Across Demo Personas:")
    for p, stat in calibration_stats.items():
        print(f"  {p}: {stat['eligible_after_1']}/{stat['total_claims']} eligible after 1 ans, "
              f"{stat['eligible_after_2']}/{stat['total_claims']} after 2 ans, "
              f"{stat['eligible_after_3']}/{stat['total_claims']} after 3 ans.")

def test_configurable_threshold_sensitivity():
    """Tests that loosening or tightening the threshold cleanly modulates eligibility."""
    claim_id = "test_sensitivity"
    now = datetime.now(timezone.utc)
    # Two good answers: raw scores 3 and 3
    ans1 = AnswerRecord(id="a1", claim_id=claim_id, raw_score=3, timestamp=now)
    ans2 = AnswerRecord(id="a2", claim_id=claim_id, raw_score=3, timestamp=now + timedelta(minutes=5))

    # At default thresholds (effective >= 2.5, conf >= 0.5):
    # n_eff = 2, conf = 0.5. Agg score: prior 2.0 -> capped 3.0. Effective = 2.0 + (3.0-2.0)*0.5 = 2.5.
    default_state = compute_mastery_state(claim_id, [ans1, ans2], evaluation_time=now + timedelta(minutes=5))
    assert default_state.is_eligible is True

    # At strict thresholds (effective >= 3.0, conf >= 0.5):
    strict_state = compute_mastery_state(
        claim_id,
        [ans1, ans2],
        evaluation_time=now + timedelta(minutes=5),
        effective_threshold=3.0,
        confidence_threshold=0.5
    )
    # Effective score is 2.5 < 3.0, so it is blocked!
    assert strict_state.is_eligible is False
