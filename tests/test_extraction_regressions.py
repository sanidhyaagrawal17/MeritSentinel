"""Regression Tests for Resume Parsing & Claim Extraction across Diverse Layouts.

Tests all 5 layout variants required to prevent regression on non-demo resume structures:
(a) wrapped bullet lines and wrapped skill lines
(b) headers named 'FEATURED PROJECTS', 'TECHNICAL SKILLS', 'COURSEWORK & COMPETITIONS',
    'WORK EXPERIENCE', 'ACHIEVEMENTS' in both uppercase and title case
(c) project title lines followed by a tech-stack line and a repository link
(d) two-column or template-style resumes
(e) pure Markdown resumes with nested formatting
"""

import pytest
from pathlib import Path
import re

from resumeproof.core.llm import FakeLLMClient
from resumeproof.extractor.pdf_parser import parse_resume, normalize_text
from resumeproof.extractor.claim_extractor import (
    extract_from_resume,
    detect_section_header,
    is_tech_stack_line,
    split_tech_stack_line,
    _map_resume_lines,
)
from resumeproof.schemas.claims import ClaimType, ClaimStatus
from resumeproof.schemas.facts import ProfileFactType

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "synthetic_resumes"


def test_regression_wrapped_bullet_lines_and_skill_lines():
    """Layout (a): Verifies unwrapping of multi-line bullets and multi-line skill lists without metric or tool truncation."""
    resume_path = FIXTURES_DIR / "wrapped_lines_resume.txt"
    parsed = parse_resume(resume_path)
    client = FakeLLMClient()

    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="synthetic_marcus",
        user_id="user_marcus",
    )

    # 1. Verify wrapped bullets are unwrapped and metric phrases remain contiguous
    claim_texts = [c.text for c in claims]
    assert any("400 concurrent read req/sec" in t for t in claim_texts), (
        "Multi-line bullet was severed; '400 concurrent read req/sec' was truncated or missing"
    )
    assert any("250,000 document vectors" in t for t in claim_texts), (
        "Multi-line bullet was severed; '250,000 document vectors' missing"
    )
    assert any("94% branch coverage" in t for t in claim_texts), (
        "Multi-line bullet was severed; '94% branch coverage' missing"
    )

    # 2. Verify wrapped skill lines are cleanly parsed into individual skill claims
    skill_claims = [c for c in claims if c.claim_type == ClaimType.SKILL]
    skill_names = {c.text.strip() for c in skill_claims}

    expected_skills = [
        "Python", "Go", "TypeScript", "SQL", "C++",
        "FastAPI", "PostgreSQL", "ChromaDB", "Redis", "React",
        "Docker", "Git", "Linux", "pytest", "Postman",
    ]
    for s in expected_skills:
        assert s in skill_names, f"Expected individual skill claim '{s}' missing from wrapped skills list: {skill_names}"

    # 3. Verify no category headers remain as skill claims
    assert not any("Languages:" in s for s in skill_names)
    assert not any("Frameworks & Databases:" in s for s in skill_names)


def test_regression_headers_upper_and_title_case_detection():
    """Layout (b): Verifies that section headers named 'FEATURED PROJECTS', 'TECHNICAL SKILLS',

    'COURSEWORK & COMPETITIONS', 'WORK EXPERIENCE', 'ACHIEVEMENTS' are recognized in both
    uppercase and title case across resume formats.
    """
    test_headers = [
        ("FEATURED PROJECTS", "PROJECTS"),
        ("Featured Projects", "PROJECTS"),
        ("KEY PROJECTS", "PROJECTS"),
        ("Key Projects", "PROJECTS"),
        ("TECHNICAL SKILLS", "SKILLS"),
        ("Technical Skills", "SKILLS"),
        ("SKILLS & EXPERTISE", "SKILLS"),
        ("Skills & Expertise", "SKILLS"),
        ("COURSEWORK & COMPETITIONS", "ACHIEVEMENTS"),
        ("Coursework & Competitions", "ACHIEVEMENTS"),
        ("WORK EXPERIENCE", "EXPERIENCE"),
        ("Work Experience", "EXPERIENCE"),
        ("PROFESSIONAL EXPERIENCE", "EXPERIENCE"),
        ("Professional Experience", "EXPERIENCE"),
        ("ACHIEVEMENTS", "ACHIEVEMENTS"),
        ("Achievements", "ACHIEVEMENTS"),
        ("EDUCATION", "EDUCATION"),
        ("Education", "EDUCATION"),
    ]

    for raw_header, expected_section in test_headers:
        detected = detect_section_header(raw_header)
        assert detected == expected_section, (
            f"Failed to detect header '{raw_header}'. Expected '{expected_section}', got '{detected}'"
        )

    # Test full document mapping with Title Case headers
    resume_path = FIXTURES_DIR / "title_case_headers_resume.md"
    parsed = parse_resume(resume_path)
    line_sections, line_projects, has_exp = _map_resume_lines(parsed)

    assert has_exp is True
    # Verify that project lines are mapped to PROJECTS and have correct project_id
    project_lines = [l for l in parsed.lines if line_sections.get(l.line_number) == "PROJECTS"]
    assert len(project_lines) >= 6
    assert any("graphrouter" in (line_projects.get(l.line_number) or "") for l in project_lines)
    assert any("logpulse" in (line_projects.get(l.line_number) or "") for l in project_lines)


def test_regression_stack_lines_with_links_extraction():
    """Layout (c): Verifies that project tech-stack lines with links extract all tools as

    ClaimType.TOOL linked to that project (with project_id set), and links are not treated as tools.
    """
    resume_path = FIXTURES_DIR / "stack_lines_with_links_resume.md"
    parsed = parse_resume(resume_path)
    client = FakeLLMClient()

    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="synthetic_tariq",
        user_id="user_tariq",
    )

    # 1. Verify stack-line tools exist with exact project_id
    tool_claims = [c for c in claims if c.claim_type == ClaimType.TOOL]
    assert len(tool_claims) >= 15

    # Check EdgeVision stack tools
    edge_tools = {c.text for c in tool_claims if c.project_id == "edgevision"}
    for expected in ["Ultralytics YOLO11", "PyTorch", "ONNX Runtime", "AWS EC2"]:
        assert expected in edge_tools, f"Expected tool '{expected}' missing from edgevision tools: {edge_tools}"

    # Check NetAudit MCP stack tools
    mcp_tools = {c.text for c in tool_claims if c.project_id == "netaudit-mcp"}
    for expected in ["Python", "Model Context Protocol", "SQLite", "pytest", "JSON-RPC 2.0"]:
        assert expected in mcp_tools, f"Expected tool '{expected}' missing from netaudit-mcp tools: {mcp_tools}"

    # Check CitationLink stack tools
    cite_tools = {c.text for c in tool_claims if c.project_id == "citationlink"}
    for expected in ["FastAPI", "ChromaDB", "Next.js 14", "TypeScript", "NetworkX"]:
        assert expected in cite_tools, f"Expected tool '{expected}' missing from citationlink tools: {cite_tools}"

    # Check PortalShield stack tools
    portal_tools = {c.text for c in tool_claims if c.project_id == "portalshield"}
    for expected in ["Python", "FastAPI", "React", "Cisco Meraki API", "JWT", "SQLite"]:
        assert expected in portal_tools, f"Expected tool '{expected}' missing from portalshield tools: {portal_tools}"

    # 2. Verify links are NOT extracted as tools or claims
    all_texts = [c.text for c in claims]
    for text in all_texts:
        assert "github.com" not in text.lower(), f"Repo URL was falsely extracted as claim/tool: '{text}'"
        assert not text.startswith("http"), f"URL was falsely extracted as claim/tool: '{text}'"

    # 3. Verify bullets under projects are metrics or experience claims, NOT tools or skills
    bullet_claims = [c for c in claims if c.project_id is not None and c.claim_type != ClaimType.TOOL]
    for bc in bullet_claims:
        assert bc.claim_type in (ClaimType.PROJECT_METRIC, ClaimType.EXPERIENCE_CLAIM), (
            f"Project bullet was misclassified: '{bc.text}' as {bc.claim_type}"
        )
        assert len(bc.text.split()) >= 4


def test_regression_two_column_template_resume_extraction():
    """Layout (d): Verifies extraction from two-column or template-style resumes."""
    resume_path = FIXTURES_DIR / "two_column_template_resume.txt"
    parsed = parse_resume(resume_path)
    client = FakeLLMClient()

    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="synthetic_kavya",
        user_id="user_kavya",
    )

    # Verify skills extracted
    skill_claims = [c for c in claims if c.claim_type == ClaimType.SKILL]
    skill_names = {c.text.strip() for c in skill_claims}
    for s in ["Python", "Go", "FastAPI", "Redis", "PyTorch", "Docker"]:
        assert s in skill_names, f"Expected skill '{s}' missing from template resume: {skill_names}"

    # Verify project tools and bullets extracted
    cachemesh_tools = {c.text for c in claims if c.project_id == "cachemesh" and c.claim_type == ClaimType.TOOL}
    assert "Go" in cachemesh_tools or "Redis" in cachemesh_tools

    # Verify metrics extracted
    metric_claims = [c for c in claims if c.claim_type == ClaimType.PROJECT_METRIC]
    assert any("45,000 req/sec" in c.text or "63.5% mAP50" in c.text for c in metric_claims)


def test_regression_markdown_resume_extraction():
    """Layout (e): Verifies pure Markdown resumes with headings, bolding, and bullet lists."""
    resume_path = FIXTURES_DIR / "markdown_resume.md"
    parsed = parse_resume(resume_path)
    client = FakeLLMClient()

    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id="synthetic_leo",
        user_id="user_leo",
    )

    # Verify markdown parsed
    assert len(parsed.lines) >= 20

    # Verify project stack-line tools extracted from 'Tech Stack:' prefix
    chronokv_tools = {c.text for c in claims if c.project_id == "chronokv" and c.claim_type == ClaimType.TOOL}
    for t in ["Go", "gRPC", "Redis", "Docker"]:
        assert t in chronokv_tools, f"Expected tool '{t}' missing from chronokv tools: {chronokv_tools}"

    fastproto_tools = {c.text for c in claims if c.project_id == "fastproto" and c.claim_type == ClaimType.TOOL}
    for t in ["C++", "Python", "CMake"]:
        assert t in fastproto_tools, f"Expected tool '{t}' missing from fastproto tools: {fastproto_tools}"

    # Verify project metrics
    p_metrics = [c for c in claims if c.claim_type == ClaimType.PROJECT_METRIC]
    assert any("32,000 writes/sec" in c.text for c in p_metrics)
    assert any("1,000,000" in c.text for c in p_metrics)
