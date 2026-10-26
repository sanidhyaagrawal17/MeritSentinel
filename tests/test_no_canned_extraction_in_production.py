"""Test ensuring zero canned or hardcoded extraction fallbacks exist in production code."""

from pathlib import Path
import pytest

from resumeproof.core.llm import FakeLLMClient, OpenAICompatibleClient
from resumeproof.extractor.claim_extractor import FullExtractionPayload


def test_production_src_contains_no_canned_fixtures():
    """Ensures src/ contains zero hardcoded persona extraction fixtures or canned dictionaries."""
    src_dir = Path("src/resumeproof")
    assert src_dir.exists(), "src/resumeproof must exist"
    
    # Assert fixtures directory is absent from src
    fixtures_dir = src_dir / "fixtures"
    assert not fixtures_dir.exists(), "src/resumeproof/fixtures must not exist; fixtures belong in tests/fixtures/"

    forbidden_tokens = [
        "DEMO_FIXTURES",
        "ALEX_RIVERA_FIXTURE",
        "PRIYA_NAIR_FIXTURE",
        "SAM_OKAFOR_FIXTURE",
        "Test Candidate",
    ]

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in content, (
                f"Forbidden canned extraction token '{token}' found in production file {py_file}"
            )


def test_fake_llm_client_has_no_implicit_canned_extraction_fallback():
    """Ensures an unconfigured FakeLLMClient returns empty extraction payload, never canned personas."""
    client = FakeLLMClient()
    messages = [
        {"role": "user", "content": "Resume Text with Line Numbers:\n[1] Sam Okafor | Chicago, IL"}
    ]
    resp = client.structured_completion(
        messages=messages,
        schema=FullExtractionPayload,
        model="test-model"
    )
    assert isinstance(resp, dict)
    assert resp.get("profile_facts") == []
    assert resp.get("claims") == []
