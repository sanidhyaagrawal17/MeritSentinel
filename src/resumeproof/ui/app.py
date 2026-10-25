"""FastAPI + Jinja2 + HTMX single-port application for ResumeProof."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
from typing import Optional
import uuid

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from resumeproof.core.config import settings
from resumeproof.core.db import Database
from resumeproof.core.llm import OpenAICompatibleClient, FakeLLMClient, LLMClient
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.schemas.claims import Claim, ClaimStatus
from resumeproof.schemas.facts import ProfileFact
from resumeproof.schemas.interview import (
    InterviewQuestion,
    QuestionPriority,
    InterviewSession,
)
from resumeproof.store.claim_store import ClaimStore
from resumeproof.rubric.builder import build_rubrics_for_persona
from resumeproof.rubric.store import RubricStore
from resumeproof.interview.generator import QuestionGenerator
from resumeproof.interview.grader import AnswerGrader
from resumeproof.interview.caching import GradingCache
from resumeproof.interview.background import BackgroundGrader
from resumeproof.interview.session_store import SessionStore
from resumeproof.interview.engine import InterviewEngine

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="ResumeProof", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def init_app_services(
    database: Optional[Database] = None,
    llm: Optional[LLMClient] = None,
) -> dict:
    """Initializes and returns application stores and engine services."""
    db = database or Database(settings.db_path)
    claim_store = ClaimStore(db)
    rubric_store = RubricStore(db)
    session_store = SessionStore(db, claim_store)

    if llm:
        llm_client = llm
    elif settings.use_fake_llm:
        llm_client = FakeLLMClient()
    else:
        llm_client = OpenAICompatibleClient(base_url=settings.ollama_base_url)

    generator = QuestionGenerator(llm_client=llm_client)
    grader = AnswerGrader(llm_client=llm_client)
    cache = GradingCache(db)
    background_grader = BackgroundGrader(grader=grader, cache=cache, max_workers=2)
    engine = InterviewEngine(
        claim_store=claim_store,
        rubric_store=rubric_store,
        session_store=session_store,
        generator=generator,
        background_grader=background_grader,
    )

    return {
        "db": db,
        "claim_store": claim_store,
        "rubric_store": rubric_store,
        "session_store": session_store,
        "llm_client": llm_client,
        "generator": generator,
        "grader": grader,
        "background_grader": background_grader,
        "engine": engine,
    }


# Attach services to app.state
app.state.services = init_app_services()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with demo persona selector and file upload form."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"nav_active": "home", "user_id": None},
    )


@app.post("/upload")
async def upload_resume(
    request: Request,
    user_id: str = Form("default_user"),
    persona: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """Uploads resume or loads demo persona, extracts facts/claims, and redirects to review."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    rubric_store: RubricStore = services["rubric_store"]
    llm_client = services["llm_client"]

    resume_path: Optional[Path] = None
    temp_dir: Optional[str] = None
    effective_persona = persona or "sam_okafor"

    if persona:
        demo_dir = Path("data/demo") / persona
        if (demo_dir / "resume.pdf").exists():
            resume_path = demo_dir / "resume.pdf"
        elif (demo_dir / "resume.md").exists():
            resume_path = demo_dir / "resume.md"
        else:
            raise HTTPException(status_code=404, detail=f"Demo persona resume not found in {demo_dir}")
    elif file and file.filename:
        temp_dir = tempfile.mkdtemp()
        resume_path = Path(temp_dir) / file.filename
        with open(resume_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        # Default fallback to sam_okafor
        resume_path = Path("data/demo/sam_okafor/resume.pdf")

    # Parse and extract
    parse_result = parse_resume(resume_path)
    facts, claims = extract_from_resume(
        parse_result=parse_result,
        llm_client=llm_client,
        user_id=user_id,
        persona_id=effective_persona,
    )

    # Persist in DB
    for f in facts:
        claim_store.save_profile_fact(f)
    for c in claims:
        claim_store.save_claim(c)

    # Build and persist rubrics if persona exists
    try:
        rubrics = build_rubrics_for_persona(effective_persona)
        rubric_store.save_rubrics(rubrics)
    except Exception:
        pass

    if temp_dir:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    return RedirectResponse(
        url=f"/review/{user_id}?persona={effective_persona}",
        status_code=303,
    )


@app.get("/review/{user_id}", response_class=HTMLResponse)
async def review_claims(request: Request, user_id: str, persona: Optional[str] = None):
    """Review screen to confirm or reject parsed candidate claims."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]

    claims = claim_store.get_claims(user_id=user_id)
    facts = claim_store.get_profile_facts(user_id=user_id)
    persona_id = persona or (claims[0].persona_id if claims else "default_persona")

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "nav_active": "review",
            "user_id": user_id,
            "persona_id": persona_id,
            "claims": claims,
            "facts": facts,
        },
    )


@app.post("/review/{user_id}/confirm_claim", response_class=HTMLResponse)
async def confirm_claim(
    request: Request,
    user_id: str,
    claim_id: str = Form(...),
    persona: Optional[str] = Form(None),
):
    """Confirms a technical claim."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.update_claim_review(claim_id=claim_id, action=ClaimStatus.CONFIRMED)
    c = claim_store.get_claim(claim_id)

    # Return partial row for HTMX swap
    return HTMLResponse(
        f"""
        <tr id="claim-{c.id}">
          <td>
            <span class="badge badge-neutral">{c.claim_type.value}</span>
            {f'<div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;"><code>{c.project_id}</code></div>' if c.project_id else ''}
          </td>
          <td>
            <div style="color: #fff; font-weight: 500;">{c.text}</div>
            {f'<div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">Source line: {c.source_line}</div>' if c.source_line else ''}
          </td>
          <td>
            <div style="display: flex; flex-wrap: wrap; gap: 0.25rem;">
              {''.join(f'<code>{t}</code>' for t in c.tools)}
              {''.join(f'<span class="badge badge-warning" style="font-size: 0.7rem;">{m}</span>' for m in c.metric_values)}
            </div>
          </td>
          <td>
            <span class="badge badge-success">Confirmed</span>
          </td>
          <td style="text-align: right;">
            <div style="display: inline-flex; gap: 0.4rem;">
              <button 
                class="btn btn-secondary btn-sm"
                hx-post="/review/{user_id}/reject_claim"
                hx-vals='{{"claim_id": "{c.id}", "persona": "{persona or ""}"}}'
                hx-target="#claim-{c.id}"
                hx-swap="outerHTML">
                Reject
              </button>
            </div>
          </td>
        </tr>
        """
    )


@app.post("/review/{user_id}/reject_claim", response_class=HTMLResponse)
async def reject_claim(
    request: Request,
    user_id: str,
    claim_id: str = Form(...),
    persona: Optional[str] = Form(None),
):
    """Rejects a technical claim."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.update_claim_review(claim_id=claim_id, action=ClaimStatus.REJECTED)
    c = claim_store.get_claim(claim_id)

    return HTMLResponse(
        f"""
        <tr id="claim-{c.id}">
          <td>
            <span class="badge badge-neutral">{c.claim_type.value}</span>
            {f'<div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;"><code>{c.project_id}</code></div>' if c.project_id else ''}
          </td>
          <td>
            <div style="color: #fff; font-weight: 500;">{c.text}</div>
            {f'<div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">Source line: {c.source_line}</div>' if c.source_line else ''}
          </td>
          <td>
            <div style="display: flex; flex-wrap: wrap; gap: 0.25rem;">
              {''.join(f'<code>{t}</code>' for t in c.tools)}
              {''.join(f'<span class="badge badge-warning" style="font-size: 0.7rem;">{m}</span>' for m in c.metric_values)}
            </div>
          </td>
          <td>
            <span class="badge badge-danger">Rejected</span>
          </td>
          <td style="text-align: right;">
            <div style="display: inline-flex; gap: 0.4rem;">
              <button 
                class="btn btn-success btn-sm"
                hx-post="/review/{user_id}/confirm_claim"
                hx-vals='{{"claim_id": "{c.id}", "persona": "{persona or ""}"}}'
                hx-target="#claim-{c.id}"
                hx-swap="outerHTML">
                Confirm
              </button>
            </div>
          </td>
        </tr>
        """
    )


@app.post("/review/{user_id}/start_interview")
async def start_interview(
    request: Request,
    user_id: str,
    persona: Optional[str] = Form(None),
    budget: int = Form(5),
):
    """Creates a new interview session and redirects to the interview room."""
    services = app.state.services
    engine: InterviewEngine = services["engine"]
    effective_persona = persona or "sam_okafor"

    session = engine.start_session(
        persona_id=effective_persona,
        user_id=user_id,
        budget=budget,
    )
    return RedirectResponse(
        url=f"/interview/{session.id}",
        status_code=303,
    )


@app.get("/interview/{session_id}", response_class=HTMLResponse)
async def interview_room(request: Request, session_id: str):
    """Interview room displaying current question, answer form, and past turns."""
    services = app.state.services
    engine: InterviewEngine = services["engine"]
    session_store: SessionStore = services["session_store"]
    claim_store: ClaimStore = services["claim_store"]

    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Interview session {session_id} not found")

    turns = session_store.get_turns_for_session(session_id)
    has_pending_grades = any(t.grade is None for t in turns)

    current_question: Optional[InterviewQuestion] = None
    target_claim_text = ""

    if len(turns) < session.budget:
        current_question = engine.get_next_question(session_id)
        if current_question:
            claim_obj = claim_store.get_claim(current_question.claim_id)
            target_claim_text = claim_obj.text if claim_obj else "Technical Claim"

    return templates.TemplateResponse(
        request=request,
        name="interview.html",
        context={
            "nav_active": "interview",
            "session": session,
            "turns": turns,
            "current_question": current_question,
            "target_claim_text": target_claim_text,
            "has_pending_grades": has_pending_grades,
            "user_id": session.user_id,
        },
    )


@app.post("/interview/{session_id}/answer")
async def submit_answer(
    request: Request,
    session_id: str,
    question_id: str = Form(...),
    claim_id: str = Form(...),
    priority: str = Form("untested"),
    question_text: str = Form(...),
    is_followup: str = Form("0"),
    rubric_id: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    answer_text: str = Form(...),
):
    """Submits answer with async background grading and instantly advances the session."""
    services = app.state.services
    engine: InterviewEngine = services["engine"]

    question = InterviewQuestion(
        id=question_id,
        claim_id=claim_id,
        priority=QuestionPriority(priority),
        question_text=question_text,
        is_followup=(is_followup == "1"),
        rubric_id=rubric_id if rubric_id else None,
        project_id=project_id if project_id else None,
    )

    # Notice: wait_for_grade=False submits to background worker and returns immediately!
    engine.submit_answer(
        session_id=session_id,
        question=question,
        answer_text=answer_text,
        wait_for_grade=False,
    )

    return RedirectResponse(
        url=f"/interview/{session_id}",
        status_code=303,
    )


@app.get("/interview/{session_id}/grade_status", response_class=HTMLResponse)
async def grade_status(session_id: str):
    """HTMX polling endpoint returning background grading badge."""
    services = app.state.services
    session_store: SessionStore = services["session_store"]

    turns = session_store.get_turns_for_session(session_id)
    has_pending = any(t.grade is None for t in turns)

    if has_pending:
        return HTMLResponse(
            f"""
            <div id="grade-status" hx-get="/interview/{session_id}/grade_status" hx-trigger="every 1s" hx-swap="outerHTML" class="badge badge-pending">
              ⏳ Background Grading Active
            </div>
            """
        )
    else:
        return HTMLResponse(
            """
            <div id="grade-status" class="badge badge-success">
              ✓ All Turns Graded
            </div>
            """
        )


@app.get("/dashboard/{user_id}", response_class=HTMLResponse)
async def dashboard(request: Request, user_id: str):
    """Mastery Matrix Dashboard displaying claims, verification badges, scores, and defensibility."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]

    claims = claim_store.get_claims(user_id=user_id)
    matrix_rows = []

    verified_count = 0
    partial_count = 0
    unverified_count = 0

    for c in claims:
        m = claim_store.get_mastery_state(c.id)
        answers = claim_store.get_answer_records(c.id)
        if m and m.total_answers > 0:
            score = round(m.effective_score, 1)
        elif answers:
            score = answers[-1].raw_score
        else:
            score = 0
        last_quote = answers[-1].evidence_quote if answers and answers[-1].evidence_quote else ""

        if score >= 3:
            verified_count += 1
        elif score >= 1:
            partial_count += 1
        else:
            unverified_count += 1

        matrix_rows.append({
            "claim": c,
            "mastery": m,
            "score": score,
            "answers_count": len(answers),
            "last_quote": last_quote,
        })

    total_claims = len(claims)
    defensibility_pct = int(round((verified_count / total_claims * 100))) if total_claims > 0 else 0

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "nav_active": "dashboard",
            "user_id": user_id,
            "total_claims": total_claims,
            "verified_count": verified_count,
            "partial_count": partial_count,
            "unverified_count": unverified_count,
            "defensibility_pct": defensibility_pct,
            "matrix_rows": matrix_rows,
        },
    )
