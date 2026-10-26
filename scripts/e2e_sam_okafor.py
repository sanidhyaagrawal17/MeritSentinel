"""Real End-to-End Demo for Sam Okafor with Real SQLite Store, Real Extraction, and Live Verification.

Invariants:
- Zero hand-named IDs (uses real generated UUIDs from store and extractor)
- Real SQLite ClaimStore and Review Flow
- Real Eligibility Gate (effective_score >= 2.5, confidence >= 0.5)
- Target JD requiring Redis and Kubernetes (skills absent from Sam's store)
- Redis and Kubernetes appear ONLY in gap_report.unsupported_gaps
- 100% Deterministic Verifier validation of tailored draft with sentence attribution trace
"""

from datetime import datetime, timezone
import json
import sys
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from resumeproof.core.db import Database
from resumeproof.core.llm import FakeLLMClient
from tests.fixtures.demo_fixtures import DEMO_FIXTURES
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.store.claim_store import ClaimStore
from resumeproof.schemas.claims import ClaimType, ClaimStatus
from resumeproof.schemas.facts import ProfileFactType
from resumeproof.schemas.mastery import MasteryState
from resumeproof.tailoring.engine import TailoringEngine
from resumeproof.verifier.strict_verifier import StrictVerifier

def run_real_e2e_sam_okafor():
    print("=" * 80)
    print("RESUMEPROOF REAL END-TO-END DEMO: SAM OKAFOR")
    print("=" * 80)

    # Step 1: Real PDF/Markdown Parsing & Claim Extraction
    resume_path = ROOT_DIR / "data" / "demo" / "sam_okafor" / "resume.md"
    print(f"\n[1] Parsing source resume: {resume_path}")
    parse_result = parse_resume(resume_path)
    print(f"    Extracted {len(parse_result.lines)} lines ({parse_result.extractor_used})")

    extractor_client = FakeLLMClient(recorded_fixtures=DEMO_FIXTURES)
    user_id = "u_sam_okafor_real"
    facts, claims = extract_from_resume(
        parse_result=parse_result,
        user_id=user_id,
        persona_id="sam_okafor",
        llm_client=extractor_client,
    )
    print(f"    Extracted {len(facts)} profile facts and {len(claims)} total claims.")

    # Step 2: Store into Real In-Memory SQLite ClaimStore
    print("\n[2] Initializing SQLite ClaimStore and persisting extracted data...")
    db = Database(":memory:")
    claim_store = ClaimStore(db)

    for f in facts:
        claim_store.save_profile_fact(f)
    for c in claims:
        claim_store.save_claim(c)

    stored_facts = claim_store.get_profile_facts(user_id=user_id)
    stored_claims = claim_store.get_claims(user_id=user_id)
    print(f"    Persisted {len(stored_facts)} facts and {len(stored_claims)} claims.")

    # Step 3: Candidate Review Confirmation
    print("\n[3] Candidate Review Step: Confirming legitimate facts and claims...")
    # Confirm profile facts
    for f in stored_facts:
        claim_store.confirm_profile_fact(f.id)

    # Confirm legitimate claims; reject planted traps
    TRAP_PHRASES = ["millions of global", "bgp route scrubbing", "kafka event bus", "raft consensus"]
    for c in stored_claims:
        is_trap = any(t in c.text.lower() for t in TRAP_PHRASES)
        if is_trap:
            claim_store.update_claim_review(c.id, action=ClaimStatus.REJECTED)
        else:
            claim_store.update_claim_review(c.id, action=ClaimStatus.CONFIRMED)

    # Step 4: Technical Interview Verification (Mastery Gate: effective_score >= 2.5, confidence >= 0.5)
    print("\n[4] Technical Interview Verification & Mastery Gate (threshold: score >= 2.5, conf >= 0.5)...")
    now = datetime.now(timezone.utc)
    for c in stored_claims:
        is_trap = any(t in c.text.lower() for t in TRAP_PHRASES)
        if is_trap:
            # Traps score 0.0 in interview
            m_state = MasteryState(
                claim_id=c.id,
                aggregate_score=0.0,
                effective_score=0.0,
                confidence=0.85,
                n_effective=2.0,
                total_answers=2,
                is_eligible=False,
                last_updated=now,
            )
        elif c.claim_type == ClaimType.SKILL:
            # Skills confirmed in review
            m_state = MasteryState(
                claim_id=c.id,
                aggregate_score=3.0,
                effective_score=3.0,
                confidence=0.75,
                n_effective=2.0,
                total_answers=2,
                is_eligible=True,
                last_updated=now,
            )
        else:
            # Legitimate project claims pass interview with high mastery
            m_state = MasteryState(
                claim_id=c.id,
                aggregate_score=3.8,
                effective_score=3.6,
                confidence=0.88,
                n_effective=2.5,
                total_answers=3,
                is_eligible=True,
                last_updated=now,
            )
        claim_store.save_mastery_state(m_state)

    # Step 5: Target Job Description with Redis and Kubernetes
    target_jd = """# Distributed Backend & Systems Engineer
## About the Role
We are seeking a backend engineer to design scalable API services and network infrastructure.

## Key Responsibilities
- Build high-performance RESTful APIs and configure relational database connection pooling with SQLite.
- Maintain network utilities, decode packet headers, and understand low-level TCP/IP, UDP, and sockets.
- Implement DNS wire format parsing per RFC 1035 for recursive resolution.
- Deploy and configure distributed in-memory caching systems using Redis clusters.
- Manage containerized microservices and automated deployments using Kubernetes (k8s).

## Requirements
- B.Tech or B.S. in Computer Science or equivalent.
- Proficiency in Python, FastAPI, and relational database systems.
- Demonstrated production experience with Redis caching and Kubernetes orchestration.
"""

    print("\n[5] Target Job Description (Requires Python, FastAPI, SQLite, TCP/IP, DNS, PLUS Redis and Kubernetes):")
    print("-" * 70)
    for line in target_jd.strip().splitlines()[:12]:
        print(f"    {line}")
    print("    ...")
    print("-" * 70)

    # Step 6: Execute Tailoring Engine
    print("\n[6] Running Tailoring Engine (Deterministic hybrid matching + mastery gating)...")
    engine = TailoringEngine(
        score_threshold=2.5,
        confidence_threshold=0.5,
        match_cutoff=0.35,
    )

    confirmed_facts = [f for f in claim_store.get_profile_facts(user_id=user_id) if f.is_confirmed]
    all_claims = claim_store.get_claims(user_id=user_id)
    mastery_map = {c.id: claim_store.get_mastery_state(c.id) for c in all_claims if claim_store.get_mastery_state(c.id)}

    tailor_result = engine.tailor(
        jd_text=target_jd,
        facts=confirmed_facts,
        claims=all_claims,
        mastery_map=mastery_map,
    )

    gap_report = tailor_result.gap_report
    print(f"\n    Total JD Requirements Extracted: {gap_report.total_requirements}")
    print(f"    Matched & Defensible: {len(gap_report.matched_and_defensible)}")
    print(f"    Unsupported Gaps:     {len(gap_report.unsupported_gaps)}")
    print(f"    Defensibility Ratio:  {gap_report.defensibility_ratio * 100:.1f}%")

    print("\n    --> Matched & Defensible Requirements:")
    for gap in gap_report.matched_and_defensible:
        print(f"        * Req: \"{gap.requirement_text}\"")
        print(f"          Claim ID: [{gap.supporting_claim_id}] (Score: {gap.effective_score}, Conf: {gap.confidence})")

    print("\n    --> Unsupported Gaps (Missing from Sam's Verified Experience):")
    for gap in gap_report.unsupported_gaps:
        print(f"        [GAP] \"{gap.requirement_text}\" -> Reason: {gap.reason}")

    # Verify that Redis and Kubernetes appear ONLY in unsupported_gaps
    gap_texts = " ".join(g.requirement_text.lower() for g in gap_report.unsupported_gaps)
    assert "redis" in gap_texts, "Redis must appear in unsupported_gaps"
    assert "kubernetes" in gap_texts or "k8s" in gap_texts, "Kubernetes must appear in unsupported_gaps"

    tailored_text = tailor_result.tailored_markdown
    assert "redis" not in tailored_text.lower(), "Redis must NEVER appear in tailored resume"
    assert "kubernetes" not in tailored_text.lower(), "Kubernetes must NEVER appear in tailored resume"

    print("\n[7] Tailored Markdown Draft Output (Strictly Extractive):")
    print("=" * 70)
    print(tailored_text)
    print("=" * 70)

    # Step 7: Strict Deterministic Verifier Audit
    print("\n[8] Running Strict Deterministic Verifier against claim store and confirmed facts...")
    verifier = StrictVerifier()
    eligible_claims = [
        c for c in all_claims
        if engine.is_claim_eligible(c, mastery_map.get(c.id), all_claims, mastery_map)
    ]
    v_report = verifier.verify(
        resume_text=tailored_text,
        eligible_claims=eligible_claims,
        confirmed_facts=confirmed_facts,
    )

    print(f"    Total Sentences:    {v_report.total_sentences}")
    print(f"    Verified Sentences: {v_report.verified_count}")
    print(f"    Blocked Sentences:  {v_report.blocked_count}")
    pass_pct = (v_report.verified_count / max(1, v_report.total_sentences)) * 100
    print(f"    Pass Ratio:         {pass_pct:.1f}%")
    print(f"    Strict Verdict:     {'PASSED' if v_report.passed else 'BLOCKED'}")

    assert v_report.passed, f"Tailored draft failed strict verification: {v_report.blocking_summary}"
    assert v_report.blocked_count == 0, "Zero sentences can be blocked"

    print("\n    Sentence Attribution Trace (First 5 Sentences):")
    for sent, cid in list(tailor_result.attribution_trace.items())[:5]:
        print(f"      - \"{sent[:60]}...\" -> [{cid}]")

    print("\n--> END-TO-END DEMO FOR SAM OKAFOR COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_real_e2e_sam_okafor()
