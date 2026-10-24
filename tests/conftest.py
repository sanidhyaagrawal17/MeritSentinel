"""Pytest test configuration and fixtures."""

import pytest
from resumeproof.core.db import Database
from resumeproof.core.llm import FakeLLMClient
from resumeproof.store.claim_store import ClaimStore
from resumeproof.review.flow import ReviewSession

@pytest.fixture
def memory_db():
    """In-memory SQLite database for isolated unit tests."""
    db = Database(":memory:")
    yield db
    db.close()

@pytest.fixture
def claim_store(memory_db):
    """Claim store backed by in-memory SQLite database."""
    store = ClaimStore(memory_db)
    yield store
    store.close()

@pytest.fixture
def review_session(claim_store):
    """Review session backed by in-memory claim store."""
    return ReviewSession(claim_store)

@pytest.fixture
def fake_llm():
    """Deterministic offline FakeLLMClient."""
    return FakeLLMClient()
