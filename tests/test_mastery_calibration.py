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
    # 1. After 1 answer, confidence is 1.0/(1.0+1.5) = 0.40 < 0.5. Exactly 0 claims should be eligible!
    for p, stat in calibration_stats.items():
        assert stat["eligible_after_1"] == 0, f"{p} had eligible claims after only 1 answer (confidence should be < 0.5)"

    # 2. After 2 answers, confidence reaches 2.0/(2.0+1.5) = 0.5714 >= 0.50. Supported claims become eligible!
    for p, stat in calibration_stats.items():
        assert stat["eligible_after_2"] > 0, f"{p} had no eligible claims after 2 solid answers"
        assert stat["eligible_after_2"] < stat["total_claims"], f"{p} admitted all claims (including unsupported traps!)"

    # 3. After 3 answers, confidence reaches 3.0/(3.0+1.5) = 0.6667 >= 0.50. Supported claims solidify eligibility.
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

    # At default thresholds (effective >= 2.5, conf >= 0.5, k=1.5):
    # n_eff = 2, conf = 2/(2+1.5) = 0.5714. Agg score: prior 2.0 -> capped 3.0. Effective = 2.0 + (3.0-2.0)*0.5714 = 2.5714 >= 2.5.
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
    # Effective score is 2.5714 < 3.0, so it is blocked!
    assert strict_state.is_eligible is False

def test_realistic_day_gaps_and_config_comparison():
    """Calibration tests with answers spread over realistic gaps (Day 0, Day 3, Day 7).

    Evaluates and records eligible-claim counts across 3 configurations:
    - Config 1: Previous baseline (conf >= 0.5, k = 2.0)
    - Config 2: Flawed confidence-only adjustment (conf >= 0.45, k = 2.0)
    - Config 3: Adopted default (conf >= 0.5, k = 1.5)
    """
    client = FakeLLMClient()
    personas = ["alex_rivera", "priya_nair", "sam_okafor"]

    configs = {
        "Config 1 (k=2.0, conf>=0.5)": {"conf": 0.5, "k": 2.0},
        "Config 2 (conf>=0.45, k=2.0)": {"conf": 0.45, "k": 2.0},
        "Config 3 (adopted default: conf>=0.5, k=1.5)": {"conf": 0.5, "k": 1.5},
    }

    now = datetime.now(timezone.utc)
    t_day0 = now
    t_day3 = now + timedelta(days=3)
    t_day7 = now + timedelta(days=7)

    results = {}

    for cfg_name, cfg in configs.items():
        conf_thresh = cfg["conf"]
        k_prior = cfg["k"]
        results[cfg_name] = {}

        for persona in personas:
            parsed = parse_resume(DEMO_DIR / persona / "resume.pdf")
            facts, claims = extract_from_resume(parsed, client, persona_id=persona)

            el_day0 = 0
            el_day3 = 0
            el_day7 = 0

            for claim in claims:
                txt = claim.text.lower()
                if any(k in txt for k in ["paxos", "70b", "byzantine"]):
                    scores = [0, 0, 1]  # Unsupported trap
                elif any(k in txt for k in ["ddos", "microservices", "100,000", "10 tb"]):
                    scores = [2, 2, 2]  # Partial claim
                else:
                    scores = [4, 4, 4]  # Supported claim

                # Day 0: 1 answer
                ans1 = AnswerRecord(id=f"{claim.id}_a1", claim_id=claim.id, raw_score=scores[0], timestamp=t_day0)
                s1 = compute_mastery_state(
                    claim.id, [ans1], evaluation_time=t_day0,
                    confidence_threshold=conf_thresh, prior_weight=k_prior
                )
                if s1.is_eligible:
                    el_day0 += 1

                # Day 3: 2 answers (Day 0 + Day 3)
                ans2 = AnswerRecord(id=f"{claim.id}_a2", claim_id=claim.id, raw_score=scores[1], timestamp=t_day3)
                s2 = compute_mastery_state(
                    claim.id, [ans1, ans2], previous_state=s1, evaluation_time=t_day3,
                    confidence_threshold=conf_thresh, prior_weight=k_prior
                )
                if s2.is_eligible:
                    el_day3 += 1

                # Day 7: 3 answers (Day 0 + Day 3 + Day 7)
                ans3 = AnswerRecord(id=f"{claim.id}_a3", claim_id=claim.id, raw_score=scores[2], timestamp=t_day7)
                s3 = compute_mastery_state(
                    claim.id, [ans1, ans2, ans3], previous_state=s2, evaluation_time=t_day7,
                    confidence_threshold=conf_thresh, prior_weight=k_prior
                )
                if s3.is_eligible:
                    el_day7 += 1

            results[cfg_name][persona] = {
                "total": len(claims),
                "day0": el_day0,
                "day3": el_day3,
                "day7": el_day7,
            }

            # Invariant: Day 0 (1 answer) must never make any claim eligible across all configs
            assert el_day0 == 0

            # Invariant: Unsupported traps must remain blocked at Day 7 across all configs
            assert el_day7 < len(claims)

    # Specific behavior verification:
    # Under Config 1 (conf>=0.5, k=2.0), a 3-day gap pushes n_eff to ~1.933 < 2.0, so confidence < 0.50 (Day 3 yields 0)
    for p in personas:
        assert results["Config 1 (k=2.0, conf>=0.5)"][p]["day3"] == 0
        assert results["Config 1 (k=2.0, conf>=0.5)"][p]["day7"] == 5

    # Under Config 2 (conf >= 0.45, k=2.0), 2 answers with 3-day gap yield confidence ~0.4915 >= 0.45.
    # Note: If scores are 4.0, effective_score = 2.0 + 2.0 * 0.4915 = 2.983 >= 2.5 (5 claims qualify).
    # But if scores are 3.0, effective_score = 2.4915 < 2.50 (0 claims qualify).
    for p in personas:
        assert results["Config 2 (conf>=0.45, k=2.0)"][p]["day3"] == 5
        assert results["Config 2 (conf>=0.45, k=2.0)"][p]["day7"] == 5

    # Under Config 3 (adopted default: conf>=0.5, k=1.5), 2 answers with 3-day gap yield confidence ~0.5630 >= 0.50.
    for p in personas:
        assert results["Config 3 (adopted default: conf>=0.5, k=1.5)"][p]["day3"] == 5
        assert results["Config 3 (adopted default: conf>=0.5, k=1.5)"][p]["day7"] == 5

def test_answers_three_days_apart():
    """Explicitly verifies Config 3 vs Config 2 on answers 3 days apart for competent candidate (raw score 3.0).

    Demonstrates the exact mathematical flaw in Config 2:
    - Day 0: Answer scoring raw 3.0 (n_eff = 1.0)
    - Day 3: Answer scoring raw 3.0 (age = 3 days, w = 0.5**(3/30) = 0.93303, n_eff = 1.93303)
    - Under Config 2 (prior_weight = 2.0, conf >= 0.45):
      conf = 1.93303 / (1.93303 + 2.0) = 0.4915 >= 0.45 (passes confidence cutoff!)
      effective_score = 2.0 + (3.0 - 2.0) * 0.4915 = 2.4915 < 2.50 (FAILS effective score cutoff!)
      Result: is_eligible is False! (The previously reported 5/7 figure required unachievable 4/4 scores).
    - Under Config 3 (adopted default: prior_weight = 1.5, conf >= 0.50):
      conf = 1.93303 / (1.93303 + 1.5) = 0.56307 >= 0.50 (passes confidence cutoff!)
      effective_score = 2.0 + (3.0 - 2.0) * 0.56307 = 2.56307 >= 2.50 (PASSES effective score cutoff!)
      Result: is_eligible is True! Competent candidate with 3 days gap is correctly verified.
    """
    claim_id = "test_three_day_gap_math"
    t_day0 = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    t_day3 = t_day0 + timedelta(days=3)

    ans1 = AnswerRecord(id="ans1", claim_id=claim_id, raw_score=3, timestamp=t_day0)
    ans2 = AnswerRecord(id="ans2", claim_id=claim_id, raw_score=3, timestamp=t_day3)

    # Config 2 test: k=2.0, conf_threshold=0.45
    s_config2 = compute_mastery_state(
        claim_id=claim_id,
        answers=[ans1, ans2],
        evaluation_time=t_day3,
        effective_threshold=2.5,
        confidence_threshold=0.45,
        prior_weight=2.0,
    )
    assert round(s_config2.confidence, 4) == 0.4915
    assert round(s_config2.effective_score, 4) == 2.4915
    assert s_config2.effective_score < 2.50
    # Demonstrates that Config 2 FAILS to admit competent (score 3.0) claims!
    assert s_config2.is_eligible is False

    # Config 3 test (Adopted Default): k=1.5, conf_threshold=0.50
    s_config3 = compute_mastery_state(
        claim_id=claim_id,
        answers=[ans1, ans2],
        evaluation_time=t_day3,
        effective_threshold=2.5,
        confidence_threshold=0.50,
        prior_weight=1.5,
    )
    assert round(s_config3.confidence, 4) == 0.5631
    assert round(s_config3.effective_score, 4) == 2.5631
    assert s_config3.confidence >= 0.50
    assert s_config3.effective_score >= 2.50
    # Demonstrates that Config 3 correctly ADMITS competent (score 3.0) claims!
    assert s_config3.is_eligible is True
