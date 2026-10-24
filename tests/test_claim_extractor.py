"""Tests for structured claim and profile fact extraction across demo personas."""

from pathlib import Path
from resumeproof.core.llm import FakeLLMClient
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.schemas.facts import ProfileFactType
from resumeproof.schemas.claims import ClaimType, ClaimStatus

DEMO_DIR = Path("data/demo")

def test_extract_alex_rivera():
    parsed = parse_resume(DEMO_DIR / "alex_rivera" / "resume.pdf")
    client = FakeLLMClient()
    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="alex_rivera",
        user_id="user_alex"
    )

    # Category 1: Profile facts check
    assert len(facts) >= 8
    fact_types = {f.fact_type for f in facts}
    assert ProfileFactType.NAME in fact_types
    assert ProfileFactType.EMAIL in fact_types
    assert ProfileFactType.PHONE in fact_types
    assert ProfileFactType.EDUCATION_INSTITUTION in fact_types
    assert ProfileFactType.CGPA in fact_types

    name_fact = next(f for f in facts if f.fact_type == ProfileFactType.NAME)
    assert name_fact.value == "Alex Rivera"
    assert name_fact.is_confirmed is False

    # Category 2: Mastery claims check
    assert len(claims) >= 6
    for c in claims:
        assert c.status == ClaimStatus.EXTRACTED
        assert c.audit is not None
        assert c.audit.extractor_text == c.text
        assert c.audit.action_taken == ClaimStatus.EXTRACTED

    # Check that tools and metrics are parsed
    p99_claim = next((c for c in claims if "1.13ms" in c.metric_values), None)
    assert p99_claim is not None
    assert "POSIX" in p99_claim.tools or "C++" in p99_claim.tools

    # Check planted trap exists (e.g. Paxos)
    paxos_trap = next((c for c in claims if "Paxos" in c.text or "Paxos" in c.tools), None)
    assert paxos_trap is not None

def test_extract_priya_nair():
    parsed = parse_resume(DEMO_DIR / "priya_nair" / "resume.pdf")
    client = FakeLLMClient()
    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="priya_nair",
        user_id="user_priya"
    )

    assert any(f.value == "Priya Nair" for f in facts)
    assert any(f.fact_type == ProfileFactType.CGPA and "3.9" in f.value for f in facts)

    # Check tabular boost claims
    assert any("4.2ms" in c.metric_values for c in claims)
    # Check planted trap (e.g. 70B LLM)
    assert any("70B" in c.metric_values or "70B" in c.text for c in claims)

def test_extract_sam_okafor():
    parsed = parse_resume(DEMO_DIR / "sam_okafor" / "resume.pdf")
    client = FakeLLMClient()
    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="sam_okafor",
        user_id="user_sam"
    )

    assert any(f.value == "Sam Okafor" for f in facts)
    assert any(f.fact_type == ProfileFactType.CGPA and "7.3" in f.value for f in facts)

    # Check junior claims
    assert any("350 req/sec" in c.metric_values or "350" in c.text for c in claims)
    # Check planted trap (e.g. Kafka or Raft Byzantine trap)
    assert any("Kafka" in c.tools or "Kafka" in c.text for c in claims)
