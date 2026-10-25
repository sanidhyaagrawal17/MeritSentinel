"""Unit and integration tests for FastAPI + Jinja2 + HTMX web UI."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from resumeproof.core.db import Database
from resumeproof.core.llm import FakeLLMClient
from resumeproof.ui.app import app, init_app_services
from resumeproof.schemas.claims import ClaimStatus

@pytest.fixture
def test_client(tmp_path: Path):
    """Initializes TestClient with isolated test database and FakeLLMClient."""
    db_file = tmp_path / "test_ui.db"
    db = Database(str(db_file))
    llm = FakeLLMClient()

    # Re-initialize services with isolated db and fake LLM
    app.state.services = init_app_services(database=db, llm=llm)

    with TestClient(app) as client:
        yield client

def test_home_page(test_client: TestClient):
    """GET / renders home page with demo persona selector and upload form."""
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert "ResumeProof" in resp.text
    assert "Adversarial Technical Verification" in resp.text
    assert "Sam Okafor" in resp.text
    assert "Alex Rivera" in resp.text
    assert "Priya Nair" in resp.text

def test_upload_demo_persona(test_client: TestClient):
    """POST /upload loads demo persona and redirects to review."""
    resp = test_client.post(
        "/upload",
        data={"user_id": "u_test_sam", "persona": "sam_okafor"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "/review/u_test_sam?persona=sam_okafor" in resp.headers["location"]

    # Verify review page renders extracted claims
    review_resp = test_client.get(resp.headers["location"])
    assert review_resp.status_code == 200
    assert "Candidate Claim Verification" in review_resp.text
    assert "Technical Claims" in review_resp.text

def test_confirm_and_reject_claim_actions(test_client: TestClient):
    """POST /review/{user_id}/confirm_claim and reject_claim update status and return HTMX row."""
    # First load claims for user
    test_client.post("/upload", data={"user_id": "u_actions", "persona": "sam_okafor"})
    claim_store = test_client.app.state.services["claim_store"]
    claims = claim_store.get_claims(user_id="u_actions")
    assert len(claims) > 0

    c0 = claims[0]

    # Confirm claim
    conf_resp = test_client.post(
        "/review/u_actions/confirm_claim",
        data={"claim_id": c0.id, "persona": "sam_okafor"},
    )
    assert conf_resp.status_code == 200
    assert "Confirmed" in conf_resp.text

    c0_updated = claim_store.get_claim(c0.id)
    assert c0_updated.status == ClaimStatus.CONFIRMED

    # Reject claim
    rej_resp = test_client.post(
        "/review/u_actions/reject_claim",
        data={"claim_id": c0.id, "persona": "sam_okafor"},
    )
    assert rej_resp.status_code == 200
    assert "Rejected" in rej_resp.text

    c0_updated2 = claim_store.get_claim(c0.id)
    assert c0_updated2.status == ClaimStatus.REJECTED

def test_full_interview_and_dashboard_flow(test_client: TestClient):
    """Tests the full mock interview loop: start session, get question, answer, poll grade, and dashboard."""
    # 1. Load persona
    test_client.post("/upload", data={"user_id": "u_interview", "persona": "sam_okafor"})
    claim_store = test_client.app.state.services["claim_store"]
    claims = claim_store.get_claims(user_id="u_interview")
    # Confirm first 3 claims
    for c in claims[:3]:
        claim_store.update_claim_review(claim_id=c.id, action=ClaimStatus.CONFIRMED)

    # 2. Start Interview
    start_resp = test_client.post(
        "/review/u_interview/start_interview",
        data={"persona": "sam_okafor", "budget": 2},
        follow_redirects=False,
    )
    assert start_resp.status_code == 303
    session_url = start_resp.headers["location"]
    session_id = session_url.split("/interview/")[1]

    # 3. View Interview Room
    room_resp = test_client.get(session_url)
    assert room_resp.status_code == 200
    assert "Technical Verification Interview" in room_resp.text
    assert "Turn 1 of 2" in room_resp.text

    # Extract current question form fields
    engine = test_client.app.state.services["engine"]
    q1 = engine.last_questions.get(session_id)
    assert q1 is not None

    # 4. Submit Answer (async background grading)
    ans_resp = test_client.post(
        f"/interview/{session_id}/answer",
        data={
            "question_id": "q_test_1",
            "claim_id": claims[0].id,
            "priority": "untested",
            "question_text": q1,
            "is_followup": "0",
            "rubric_id": "",
            "project_id": "task-tracker-api",
            "answer_text": "We enabled SQLite WAL mode with SQLAlchemy connection pooling to 10 connections.",
        },
        follow_redirects=False,
    )
    assert ans_resp.status_code == 303

    # Drain background grader to ensure grade completes
    engine.background_grader.drain_all(timeout=5.0)

    # 5. Check Grade Status HTMX Endpoint
    status_resp = test_client.get(f"/interview/{session_id}/grade_status")
    assert status_resp.status_code == 200
    assert "All Turns Graded" in status_resp.text

    # 6. View Dashboard
    dash_resp = test_client.get("/dashboard/u_interview")
    assert dash_resp.status_code == 200
    assert "Defensibility & Mastery Dashboard" in dash_resp.text
    assert "Claim Verification Matrix" in dash_resp.text

def test_custom_file_upload(test_client: TestClient, tmp_path: Path):
    """POST /upload handles custom markdown resume file upload."""
    md_content = """# Alex Rivera
Full Stack Engineer
Email: alex@example.com

## Skills
- Python, TypeScript, Docker, PostgreSQL

## Projects
- Multi-Tenant SaaS: Scaled PostgreSQL partitioned database to 25000 users.
"""
    resume_file = tmp_path / "custom_resume.md"
    resume_file.write_text(md_content, encoding="utf-8")

    with open(resume_file, "rb") as f:
        resp = test_client.post(
            "/upload",
            data={"user_id": "u_custom"},
            files={"file": ("custom_resume.md", f, "text/markdown")},
            follow_redirects=False,
        )

    assert resp.status_code == 303
    assert "/review/u_custom" in resp.headers["location"]
