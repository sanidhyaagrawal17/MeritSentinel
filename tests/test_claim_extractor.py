"""Parsing-Pipeline Tests: Claim & Profile Fact Extraction.

NOTE: These tests use FakeLLMClient with recorded fixtures to validate the
pipeline mechanics, schema serialization, profile fact isolation, and audit trail
initialization. They do NOT validate real-model extraction accuracy or LLM quality.
Real-model extraction quality is verified manually with Ollama (qwen2.5:3b).
"""

import pytest
from pathlib import Path
from resumeproof.core.llm import FakeLLMClient
from tests.fixtures.demo_fixtures import DEMO_FIXTURES
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.schemas.facts import ProfileFactType
from resumeproof.schemas.claims import ClaimType, ClaimStatus

DEMO_DIR = Path("data/demo")

@pytest.mark.parsing_pipeline
def test_parsing_pipeline_alex_rivera():
    """Parsing-pipeline test for Alex Rivera: validates schema adherence and fact separation via FakeLLMClient."""
    parsed = parse_resume(DEMO_DIR / "alex_rivera" / "resume.pdf")
    client = FakeLLMClient(recorded_fixtures=DEMO_FIXTURES)
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

@pytest.mark.parsing_pipeline
def test_parsing_pipeline_priya_nair():
    """Parsing-pipeline test for Priya Nair: validates schema adherence and fact separation via FakeLLMClient."""
    parsed = parse_resume(DEMO_DIR / "priya_nair" / "resume.pdf")
    client = FakeLLMClient(recorded_fixtures=DEMO_FIXTURES)
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

@pytest.mark.parsing_pipeline
def test_parsing_pipeline_sam_okafor():
    """Parsing-pipeline test for Sam Okafor: validates schema adherence and fact separation via FakeLLMClient."""
    parsed = parse_resume(DEMO_DIR / "sam_okafor" / "resume.pdf")
    client = FakeLLMClient(recorded_fixtures=DEMO_FIXTURES)
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

def test_sam_okafor_has_no_employer_or_employment_dates_without_experience_section():
    """Requirement 5(a): Profile facts must never be inferred.
    employer_name and employment_dates only exist if the resume has an experience section.
    For Sam Okafor (education, projects, skills only), employer_name and employment_dates must not exist.
    """
    parsed = parse_resume(DEMO_DIR / "sam_okafor" / "resume.pdf")
    client = FakeLLMClient(recorded_fixtures=DEMO_FIXTURES)
    facts, _ = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="sam_okafor",
        user_id="user_sam"
    )

    fact_types = {f.fact_type for f in facts}
    assert ProfileFactType.EMPLOYER_NAME not in fact_types, "employer_name was erroneously extracted for Sam Okafor"
    assert ProfileFactType.EMPLOYMENT_DATES not in fact_types, "employment_dates was erroneously extracted for Sam Okafor"

def test_skills_split_into_individual_claims():
    """Requirement 5(b): Split skill lines into one skill claim per skill (Python, SQL, FastAPI, ...)."""
    parsed = parse_resume(DEMO_DIR / "sam_okafor" / "resume.pdf")
    client = FakeLLMClient()
    _, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="sam_okafor",
        user_id="user_sam"
    )

    skill_claims = [c for c in claims if c.claim_type == ClaimType.SKILL]
    skill_texts = {c.text.strip() for c in skill_claims}

    # Verify individual skills are present as distinct claims
    expected_individual_skills = ["Python", "SQL", "FastAPI", "SQLite", "Sockets", "TCP/IP", "UDP", "HTTP", "DNS", "Git", "Linux", "Docker"]
    for s in expected_individual_skills:
        assert s in skill_texts, f"Expected individual skill claim '{s}' not found in {skill_texts}"

    # Verify no multi-skill lines remain as single claims
    assert not any("Languages:" in s for s in skill_texts)
    assert not any("Networking & Backend:" in s for s in skill_texts)
    assert not any("Tools:" in s for s in skill_texts)
    # Verify section headings are never extracted as skill claims
    assert "## Skills" not in skill_texts
    assert "SKILLS" not in skill_texts
    assert "Skills" not in skill_texts

def test_project_and_experience_claims_never_typed_as_skill_or_tool():
    """Requirement 5(c): Experience or project claims must never be typed as skill or tool."""
    parsed = parse_resume(DEMO_DIR / "sam_okafor" / "resume.pdf")
    client = FakeLLMClient()
    _, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="sam_okafor",
        user_id="user_sam"
    )

    for c in claims:
        if c.project_id is not None:
            assert c.claim_type not in (ClaimType.SKILL, ClaimType.TOOL), (
                f"Project claim '{c.text}' under project '{c.project_id}' was misclassified as {c.claim_type}"
            )

