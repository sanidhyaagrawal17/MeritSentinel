"""Unit and integration tests for FastAPI + Jinja2 + HTMX web UI."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from resumeproof.core.db import Database
from resumeproof.core.llm import FakeLLMClient
from tests.fixtures.demo_fixtures import DEMO_FIXTURES
from resumeproof.ui.app import app, init_app_services
from resumeproof.schemas.claims import ClaimStatus

@pytest.fixture
def test_client(tmp_path: Path):
    """Initializes TestClient with isolated test database and FakeLLMClient."""
    db_file = tmp_path / "test_ui.db"
    db = Database(str(db_file))
    llm = FakeLLMClient(recorded_fixtures=DEMO_FIXTURES)

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


def test_upload_path_traversal_filename(test_client: TestClient, tmp_path: Path):
    """POST /upload sanitizes path traversal characters in filename and stays in upload_dir."""
    content = b"# Traversal Test\n- Python skill"
    # Test unix traversal and windows traversal
    for bad_name in ["../../../../etc/passwd.md", "..\\..\\windows\\system32\\hosts.txt"]:
        resp = test_client.post(
            "/upload",
            data={"user_id": "u_traversal"},
            files={"file": (bad_name, content, "text/markdown")},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        # Ensure file was saved inside .data_uploads and did not traverse outside
        upload_dir = Path(".data_uploads")
        assert upload_dir.exists()
        # Verify no file was created outside upload_dir
        assert not Path("passwd.md").exists()
        assert not Path("hosts.txt").exists()


def test_upload_oversized_file_rejected_with_413(test_client: TestClient):
    """POST /upload rejects files exceeding 5MB with HTTP 413."""
    # 5.5MB dummy content
    oversized_content = b"0" * (5 * 1024 * 1024 + 1024 * 512)
    resp = test_client.post(
        "/upload",
        data={"user_id": "u_oversized"},
        files={"file": ("large_resume.txt", oversized_content, "text/plain")},
        follow_redirects=False,
    )
    assert resp.status_code == 413
    assert "exceeds maximum" in resp.json()["detail"].lower()


def test_upload_unsupported_file_type_rejected_with_400(test_client: TestClient):
    """POST /upload rejects disallowed extensions (e.g. .exe, .sh) with HTTP 400."""
    resp = test_client.post(
        "/upload",
        data={"user_id": "u_bad_ext"},
        files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "unsupported file type" in resp.json()["detail"].lower()


def test_jinja_autoescape_configured():
    """Verifies Jinja2 templates environment has autoescape enabled."""
    from resumeproof.ui.app import templates
    assert templates.env.autoescape is True


def test_tailor_page_get(test_client: TestClient):
    """GET /tailor/{user_id} renders tailoring page with sample JD and action button."""
    resp = test_client.get("/tailor/u_test_view")
    assert resp.status_code == 200
    assert "Resume Tailoring & Strict Verifier" in resp.text
    assert "Target Job Description" in resp.text
    assert "Tailor &amp; Verify Resume" in resp.text or "Tailor & Verify Resume" in resp.text


def test_tailor_post_and_export_flow(test_client: TestClient):
    """POST /tailor/{user_id} runs tailoring, strict verifier, and enables MD and PDF download."""
    from resumeproof.schemas.mastery import MasteryState

    # 1. Load persona
    test_client.post("/upload", data={"user_id": "u_tailor_user", "persona": "sam_okafor"})
    claim_store = test_client.app.state.services["claim_store"]
    claims = claim_store.get_claims(user_id="u_tailor_user")
    assert len(claims) > 0

    # 2. Confirm claims and record high mastery
    for c in claims:
        claim_store.update_claim_review(claim_id=c.id, action=ClaimStatus.CONFIRMED)
        claim_store.save_mastery_state(MasteryState(
            claim_id=c.id,
            aggregate_score=3.5,
            effective_score=3.5,
            confidence=0.85,
            total_answers=2,
            is_eligible=True,
        ))

    # 3. Post JD matching backend systems
    jd_sample = """Looking for a Backend Engineer:
- RESTful API development with FastAPI and database connection pooling
- Low-latency socket networking and packet inspection
- SQLite performance tuning and caching
"""
    tailor_resp = test_client.post(
        "/tailor/u_tailor_user",
        data={"jd_text": jd_sample},
    )
    assert tailor_resp.status_code == 200
    assert "Matched &amp; Defensible" in tailor_resp.text or "Matched & Defensible" in tailor_resp.text
    assert "Strict Deterministic Verifier: PASSED" in tailor_resp.text
    assert "Download ATS PDF" in tailor_resp.text

    # 4. Download Markdown
    md_resp = test_client.get("/tailor/u_tailor_user/download_md")
    assert md_resp.status_code == 200
    assert "text/markdown" in md_resp.headers["content-type"]
    assert "attachment; filename=\"resume_u_tailor_user.md\"" in md_resp.headers["content-disposition"]
    assert "# Sam Okafor" in md_resp.text or "# Candidate" in md_resp.text

    # 5. Download ATS PDF
    pdf_resp = test_client.get("/tailor/u_tailor_user/download_pdf")
    assert pdf_resp.status_code == 200
    assert "application/pdf" in pdf_resp.headers["content-type"]
    assert pdf_resp.content.startswith(b"%PDF-")


def test_tailor_pdf_export_blocked_when_unverified(test_client: TestClient):
    """GET /tailor/{user_id}/download_pdf returns HTTP 400 when strict verification fails."""
    from resumeproof.verifier.strict_verifier import VerificationReport, SentenceVerificationResult

    # Set up blocked report in cache
    test_client.app.state.tailoring_cache["u_blocked_user"] = {
        "tailored_markdown": "# Blocked User\n- Unsupported hallucinated metric 999k req/sec",
        "verifier_report": VerificationReport(
            passed=False,
            total_sentences=2,
            verified_count=1,
            blocked_count=1,
            sentence_results=[
                SentenceVerificationResult(
                    sentence_idx=0,
                    sentence_text="Unsupported hallucinated metric 999k req/sec",
                    status="BLOCKED",
                    blocking_reasons=["Ungrounded metric 999k req/sec"],
                )
            ],
            blocking_summary=["Sentence 0: Ungrounded metric 999k req/sec"],
        ),
    }

    resp_pdf = test_client.get("/tailor/u_blocked_user/download_pdf")
    assert resp_pdf.status_code == 400
    assert "Export blocked" in resp_pdf.json()["detail"]
    assert "failed strict verification" in resp_pdf.json()["detail"]

    # Invariant: download_md must also be strictly blocked (no force-export)
    resp_md = test_client.get("/tailor/u_blocked_user/download_md")
    assert resp_md.status_code == 400
    assert "Export blocked" in resp_md.json()["detail"]
    assert "failed strict verification" in resp_md.json()["detail"]


