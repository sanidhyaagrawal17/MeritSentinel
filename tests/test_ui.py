"""Unit and integration tests for FastAPI + Jinja2 + HTMX web UI."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from resumeproof.core.db import Database
from resumeproof.core.llm import FakeLLMClient
from tests.fixtures.demo_fixtures import DEMO_FIXTURES
from resumeproof.ui.app import app, init_app_services
from resumeproof.schemas.claims import ClaimStatus, ClaimType

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
    assert ("ResumeProof" in resp.text or "MeritSentinel" in resp.text)
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

    # Reset claim back to pending (Undo)
    reset_resp = test_client.post(
        "/review/u_actions/reset_claim",
        data={"claim_id": c0.id, "persona": "sam_okafor"},
    )
    assert reset_resp.status_code == 200
    assert "Pending" in reset_resp.text

    c0_updated3 = claim_store.get_claim(c0.id)
    assert c0_updated3.status == ClaimStatus.EXTRACTED

    # Test individual profile fact confirm and undo (reject_fact)
    facts = claim_store.get_profile_facts(user_id="u_actions")
    assert len(facts) > 0
    f0 = facts[0]
    confirm_f_resp = test_client.post(
        "/review/u_actions/confirm_fact",
        data={"fact_id": f0.id},
    )
    assert confirm_f_resp.status_code == 200
    assert "Confirmed" in confirm_f_resp.text
    assert "Undo" in confirm_f_resp.text
    assert claim_store.get_profile_fact(f0.id).is_confirmed is True

    undo_f_resp = test_client.post(
        "/review/u_actions/reject_fact",
        data={"fact_id": f0.id},
    )
    assert undo_f_resp.status_code == 200
    assert "Pending" in undo_f_resp.text
    assert "Confirm" in undo_f_resp.text
    assert claim_store.get_profile_fact(f0.id).is_confirmed is False

    # Test bulk confirm all facts and reset all facts (Undo All Facts)
    test_client.post("/review/u_actions/confirm_all_facts")
    for f in claim_store.get_profile_facts(user_id="u_actions"):
        assert f.is_confirmed is True

    test_client.post("/review/u_actions/reset_all_facts")
    for f in claim_store.get_profile_facts(user_id="u_actions"):
        assert f.is_confirmed is False

    # Test 3-way toggle on skill badge: Pending -> Confirmed -> Rejected -> Pending
    skills = [c for c in claims if c.claim_type == ClaimType.SKILL]
    if skills:
        s0 = skills[0]
        # Step 1: Pending -> Confirmed
        t1 = test_client.post("/review/u_actions/toggle_skill", data={"claim_id": s0.id})
        assert t1.status_code == 200
        assert "skill-badge active" in t1.text
        assert claim_store.get_claim(s0.id).status == ClaimStatus.CONFIRMED

        # Step 2: Confirmed -> Rejected
        t2 = test_client.post("/review/u_actions/toggle_skill", data={"claim_id": s0.id})
        assert t2.status_code == 200
        assert "skill-badge inactive" in t2.text
        assert claim_store.get_claim(s0.id).status == ClaimStatus.REJECTED

        # Step 3: Rejected -> Pending
        t3 = test_client.post("/review/u_actions/toggle_skill", data={"claim_id": s0.id})
        assert t3.status_code == 200
        assert "skill-badge pending" in t3.text
        assert claim_store.get_claim(s0.id).status == ClaimStatus.EXTRACTED

    # Test project bulk confirm and undo (reset_project)
    project_claims = [c for c in claims if c.project_id and c.claim_type != ClaimType.SKILL]
    if project_claims:
        proj_id = project_claims[0].project_id
        test_client.post("/review/u_actions/confirm_project", data={"project_id": proj_id})
        for c in claim_store.get_claims(user_id="u_actions"):
            if c.project_id == proj_id and c.claim_type != ClaimType.SKILL:
                assert c.status == ClaimStatus.CONFIRMED

        test_client.post("/review/u_actions/reset_project", data={"project_id": proj_id})
        for c in claim_store.get_claims(user_id="u_actions"):
            if c.project_id == proj_id and c.claim_type != ClaimType.SKILL:
                assert c.status == ClaimStatus.EXTRACTED

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


def test_interview_question_rephrase_flow(test_client: TestClient):
    """POST /interview/{session_id}/rephrase returns simplified question with HTMX swap and preserves state."""
    # 1. Upload demo candidate and confirm all claims
    test_client.post("/upload", data={"user_id": "u_rephrase_test", "persona": "sam_okafor"})
    claim_store = test_client.app.state.services["claim_store"]
    for c in claim_store.get_claims(user_id="u_rephrase_test"):
        test_client.post(
            "/review/u_rephrase_test/confirm_claim",
            data={"claim_id": c.id, "persona": "sam_okafor"},
        )

    # 2. Start interview
    start_resp = test_client.post(
        "/review/u_rephrase_test/start_interview",
        data={"persona": "sam_okafor"},
        follow_redirects=False,
    )
    assert start_resp.status_code == 303
    session_url = start_resp.headers["location"]
    session_id = session_url.split("/interview/")[-1]

    # 3. Load initial interview room
    room_resp = test_client.get(session_url)
    assert room_resp.status_code == 200
    assert "btn-rephrase-question" in room_resp.text
    assert "active-question-container" in room_resp.text

    # Extract original question text
    session_store = test_client.app.state.services["session_store"]
    session_before = session_store.get_session(session_id)
    assert session_before.current_question is not None
    original_text = session_before.current_question.question_text

    # 4. Trigger rephrase via HTMX POST
    rephrase_resp = test_client.post(
        f"/interview/{session_id}/rephrase",
        headers={"HX-Request": "true"},
    )
    assert rephrase_resp.status_code == 200
    assert "active-question-container" in rephrase_resp.text
    assert "Rephrased for Clarity" in rephrase_resp.text
    assert "Show original phrasing" in rephrase_resp.text

    # Verify session store was updated
    session_after = session_store.get_session(session_id)
    assert session_after.current_question is not None
    assert session_after.current_question.is_rephrased is True
    assert session_after.current_question.original_question_text == original_text

    # 5. Fallback GET returns 303 redirect
    get_resp = test_client.get(f"/interview/{session_id}/rephrase", follow_redirects=False)
    assert get_resp.status_code == 303


def test_upload_multipage_report_rejected_with_400(test_client: TestClient, tmp_path: Path):
    """POST /upload rejects documents exceeding max resume page limit (e.g. 35 pages) with HTTP 400."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(12):  # 12 pages exceeds max_resume_pages=5 limit
        writer.add_blank_page(width=612, height=792)
    pdf_file = tmp_path / "research_report.pdf"
    with open(pdf_file, "wb") as f:
        writer.write(f)

    with open(pdf_file, "rb") as f:
        resp = test_client.post(
            "/upload",
            data={"user_id": "u_report_test"},
            files={"file": ("research_report.pdf", f, "application/pdf")},
            follow_redirects=False,
        )

    assert resp.status_code == 400
    assert "exceeds maximum resume page limit" in resp.json()["detail"].lower()
    assert "12 pages" in resp.json()["detail"]


def test_rephrase_then_answer_advances_to_different_question(test_client: TestClient):
    """Answering a rephrased question advances session to a distinct subsequent question without repetition."""
    # 1. Upload demo candidate and confirm claims
    test_client.post("/upload", data={"user_id": "u_distinct_flow", "persona": "sam_okafor"})
    claim_store = test_client.app.state.services["claim_store"]
    for c in claim_store.get_claims(user_id="u_distinct_flow"):
        test_client.post(
            "/review/u_distinct_flow/confirm_claim",
            data={"claim_id": c.id, "persona": "sam_okafor"},
        )

    # 2. Start interview
    start_resp = test_client.post(
        "/review/u_distinct_flow/start_interview",
        data={"persona": "sam_okafor"},
        follow_redirects=False,
    )
    session_url = start_resp.headers["location"]
    session_id = session_url.split("/interview/")[-1]

    # 3. Load question 1
    test_client.get(session_url)
    session_store = test_client.app.state.services["session_store"]
    s1 = session_store.get_session(session_id)
    q1 = s1.current_question
    assert q1 is not None

    # 4. Rephrase question 1
    rephrase_resp = test_client.post(
        f"/interview/{session_id}/rephrase",
        headers={"HX-Request": "true"},
    )
    assert rephrase_resp.status_code == 200
    s1_rephrased = session_store.get_session(session_id)
    q1_rephrased = s1_rephrased.current_question
    assert q1_rephrased.is_rephrased is True

    # 5. Submit candidate answer (e.g., stating no knowledge)
    ans_resp = test_client.post(
        f"/interview/{session_id}/answer",
        data={
            "question_id": q1_rephrased.id,
            "claim_id": q1_rephrased.claim_id,
            "priority": q1_rephrased.priority.value,
            "question_text": q1_rephrased.question_text,
            "is_followup": "0",
            "rubric_id": q1_rephrased.rubric_id or "",
            "project_id": q1_rephrased.project_id or "",
            "answer_text": "I do not know or have no idea about this specific module.",
        },
        follow_redirects=False,
    )
    assert ans_resp.status_code == 303

    # 6. Fetch next question in room
    room_resp = test_client.get(session_url)
    assert room_resp.status_code == 200

    s2 = session_store.get_session(session_id)
    q2 = s2.current_question
    assert q2 is not None

    # Assert question 2 is NOT identical to question 1 or rephrased question 1
    assert q2.id != q1.id
    assert q2.id != q1_rephrased.id
    assert q2.question_text.strip() != q1.question_text.strip()
    assert q2.question_text.strip() != q1_rephrased.question_text.strip()


def test_claim_edit_xss_escaped(test_client: TestClient):
    """Inline claim edit and partial swap escape HTML/script payloads via Jinja2 autoescaping."""
    test_client.post("/upload", data={"user_id": "u_xss_test", "persona": "sam_okafor"})
    claim_store = test_client.app.state.services["claim_store"]
    c0 = claim_store.get_claims(user_id="u_xss_test")[0]

    payload = "<script>alert('xss')</script> Built API"
    save_resp = test_client.post(
        "/review/u_xss_test/save_claim_edit",
        data={"claim_id": c0.id, "edited_text": payload, "persona": "sam_okafor"},
    )
    assert save_resp.status_code == 200
    assert "<script>" not in save_resp.text
    assert "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in save_resp.text


def test_upload_browser_html_error_banner(test_client: TestClient):
    """Browser form uploads with Accept: text/html render styled error banner instead of raw JSON."""
    resp = test_client.post(
        "/upload",
        data={"user_id": "u_bad_ext"},
        files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
        headers={"Accept": "text/html,application/xhtml+xml"},
    )
    assert resp.status_code == 400
    assert "Upload Rejected:" in resp.text
    assert "Unsupported file type" in resp.text


def test_grade_status_triggers_hx_refresh_when_done(test_client: TestClient):
    """grade_status with was_pending=1 emits HX-Refresh header when all turns are graded."""
    test_client.post("/upload", data={"user_id": "u_refresh_test", "persona": "sam_okafor"})
    claim_store = test_client.app.state.services["claim_store"]
    for c in claim_store.get_claims(user_id="u_refresh_test"):
        test_client.post("/review/u_refresh_test/confirm_claim", data={"claim_id": c.id})
    start_resp = test_client.post(
        "/review/u_refresh_test/start_interview",
        data={"persona": "sam_okafor"},
        follow_redirects=False,
    )
    session_id = start_resp.headers["location"].split("/interview/")[-1]
    resp = test_client.get(f"/interview/{session_id}/grade_status?was_pending=1")
    assert resp.status_code == 200
    assert "All Turns Graded" in resp.text
    assert resp.headers.get("HX-Refresh") == "true"





