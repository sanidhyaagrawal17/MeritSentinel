"""FastAPI + Jinja2 + HTMX single-port application for ResumeProof."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
from typing import Optional
import uuid
import re
import urllib.parse

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from resumeproof.core.config import settings
from resumeproof.core.db import Database
from resumeproof.core.llm import OpenAICompatibleClient, FakeLLMClient, LLMClient
from resumeproof.extractor.pdf_parser import (
    parse_resume,
    ResumePageLimitExceededError,
    ResumeLengthExceededError,
)
from resumeproof.extractor.claim_extractor import extract_from_resume, infer_skill_category
from resumeproof.schemas.claims import Claim, ClaimStatus, ClaimType
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
from resumeproof.interview.session_logger import SessionLogger
from resumeproof.interview.engine import InterviewEngine
from resumeproof.core.security import (
    create_signed_session_token,
    verify_signed_session_token,
    screen_candidate_answer,
    global_rate_limiter,
)
from resumeproof.tailoring.engine import TailoringEngine, GapReport, TailoringResult
from resumeproof.verifier.strict_verifier import (
    StrictVerifier,
    VerificationReport,
    VerificationBlockedError,
)
from resumeproof.exporter.export import export_markdown, export_pdf

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="MeritSentinel", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.autoescape = True
app.state.tailoring_cache = {}

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}

def sanitize_filename(filename: str) -> str:
    """Sanitizes upload filename by stripping directory traversal and keeping [a-zA-Z0-9_.-]."""
    # Strip any directory path components (both unix / and windows \)
    base = Path(filename.replace("\\", "/")).name
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    clean = clean.lstrip("._")
    if not clean:
        clean = f"upload_{uuid.uuid4().hex[:8]}"
    orig_ext = Path(base).suffix.lower()
    if orig_ext in ALLOWED_EXTENSIONS and not clean.lower().endswith(orig_ext):
        clean += orig_ext
    return clean


def init_app_services(
    database: Optional[Database] = None,
    llm: Optional[LLMClient] = None,
) -> dict:
    """Initializes and returns application stores and engine services."""
    db = database or Database(settings.db_path)
    claim_store = ClaimStore(db)
    rubric_store = RubricStore(db)
    session_store = SessionStore(db, claim_store)
    session_logger = SessionLogger(db)

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
        session_logger=session_logger,
    )

    return {
        "db": db,
        "claim_store": claim_store,
        "rubric_store": rubric_store,
        "session_store": session_store,
        "session_logger": session_logger,
        "llm_client": llm_client,
        "generator": generator,
        "grader": grader,
        "background_grader": background_grader,
        "engine": engine,
    }


# Attach services to app.state
app.state.services = init_app_services()


# ---------------------------------------------------------------------------
# Security Headers & Throttling Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enforces rate limiting on POST routes and attaches robust security headers."""
    client_ip = request.client.host if request.client else "127.0.0.1"

    # Enforce CORS Origin check on POST requests
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin:
            parsed_origin = urllib.parse.urlparse(origin)
            host = request.headers.get("host")
            if host and parsed_origin.netloc != host:
                return Response("Forbidden: cross-origin request rejected.", status_code=403)

    if request.method == "POST" and client_ip != "testclient":
        if not global_rate_limiter.is_allowed(client_ip):
            return Response("Too Many Requests. Submission rate limit exceeded.", status_code=429)

    response = await call_next(request)

    # Security Headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response


def get_current_user_and_session(request: Request) -> tuple[Optional[str], Optional[InterviewSession]]:
    """Extracts verified user_id from signed cookie and retrieves any active interview session."""
    cookie = request.cookies.get("meritsentinel_session")
    user_id = None
    if cookie:
        payload = verify_signed_session_token(cookie)
        if payload and "user_id" in payload:
            user_id = payload["user_id"]

    active_session = None
    services = getattr(request.app.state, "services", None)
    if user_id and services and "session_store" in services:
        active_session = services["session_store"].get_active_session_for_user(user_id)

    return user_id, active_session


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with demo persona selector, resume upload form, and active session resumption."""
    curr_user, active_session = get_current_user_and_session(request)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "nav_active": "home",
            "user_id": curr_user,
            "active_session": active_session,
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serves application favicon."""
    fav_path = STATIC_DIR / "favicon.svg"
    if fav_path.exists():
        return FileResponse(fav_path, media_type="image/svg+xml")
    return Response(status_code=204)


def _upload_error_response(request: Request, status_code: int, detail: str):
    """Returns a styled HTML error on browser form submits, or raises HTTPException for API clients."""
    accept = request.headers.get("accept", "").lower()
    if "text/html" in accept:
        services = app.state.services
        session_store: SessionStore = services["session_store"]
        _, active_session = get_current_user_and_session(request)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "nav_active": "home",
                "active_session": active_session,
                "error_message": detail,
            },
            status_code=status_code,
        )
    raise HTTPException(status_code=status_code, detail=detail)


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
        ext = Path(file.filename.replace("\\", "/")).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _upload_error_response(
                request,
                400,
                f"Unsupported file type '{ext}'. Only .pdf, .md, and .txt files are allowed.",
            )

        safe_filename = sanitize_filename(file.filename)
        upload_dir = settings.upload_dir
        upload_dir.mkdir(parents=True, exist_ok=True)
        resume_path = upload_dir / f"{uuid.uuid4().hex}_{safe_filename}"

        total_bytes = 0
        chunk_size = 64 * 1024
        with open(resume_path, "wb") as f_out:
            while chunk := await file.read(chunk_size):
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_size_bytes:
                    f_out.close()
                    resume_path.unlink(missing_ok=True)
                    return _upload_error_response(
                        request,
                        413,
                        "File size exceeds maximum allowed limit of 5MB.",
                    )
                f_out.write(chunk)
    else:
        # Default fallback to sam_okafor
        resume_path = Path("data/demo/sam_okafor/resume.pdf")

    # Parse and extract with strict resume boundaries (max pages and lines)
    try:
        parse_result = parse_resume(
            resume_path,
            max_pages=settings.max_resume_pages,
            max_lines=settings.max_resume_lines,
        )
    except (ResumePageLimitExceededError, ResumeLengthExceededError, ValueError) as err:
        if file and file.filename and resume_path and resume_path.exists():
            resume_path.unlink(missing_ok=True)
        return _upload_error_response(request, 400, str(err))
    facts, claims = extract_from_resume(
        parse_result=parse_result,
        llm_client=llm_client,
        user_id=user_id,
        persona_id=effective_persona,
    )

    if persona:
        existing_claims = claim_store.get_claims(user_id=user_id)
        if existing_claims and any(c.persona_id and c.persona_id != effective_persona for c in existing_claims):
            claim_store.clear_claims(user_id)

    claim_store.clear_user_data(user_id)
    # Persist in DB
    for f in facts:
        claim_store.save_profile_fact(f)
    for c in claims:
        claim_store.save_claim(c)

    # Build and persist rubrics (demo persona or dynamically synthesized for custom resume)
    try:
        from resumeproof.rubric.builder import synthesize_rubrics_from_claims
        rubric_store.clear_rubrics_for_persona(effective_persona)
        rubrics = build_rubrics_for_persona(effective_persona)
        if not rubrics:
            rubrics = synthesize_rubrics_from_claims(claims, effective_persona)
        if rubrics:
            rubric_store.save_rubrics(rubrics)
    except Exception:
        pass

    if temp_dir:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    response = RedirectResponse(
        url=f"/review/{user_id}?persona={effective_persona}",
        status_code=303,
    )
    # Set persistent cryptographically signed session cookie
    token = create_signed_session_token(user_id=user_id, extra_data={"persona": effective_persona})
    session_store: SessionStore = services["session_store"]
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    session_store.save_user_session(
        session_token=token,
        user_id=user_id,
        client_ip=client_ip,
        user_agent=user_agent,
    )
    response.set_cookie(
        key="meritsentinel_session",
        value=token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=604800,
    )
    return response


@app.get("/review/{user_id}", response_class=HTMLResponse)
async def review_claims(request: Request, user_id: str, persona: Optional[str] = None):
    """Review screen to confirm or reject parsed candidate claims."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    session_store: SessionStore = services["session_store"]

    claims = claim_store.get_claims(user_id=user_id)
    facts = claim_store.get_profile_facts(user_id=user_id)
    persona_id = persona or (claims[0].persona_id if claims else "default_persona")
    active_session = session_store.get_active_session_for_user(user_id)

    # Separate skills from substantive audit claims
    skill_claims = [c for c in claims if c.claim_type == ClaimType.SKILL]
    audit_claims = [c for c in claims if c.claim_type != ClaimType.SKILL]

    # Group skills by category
    skills_by_category: dict[str, list[Claim]] = {}
    for sc in skill_claims:
        cat = sc.project_id or infer_skill_category(sc.raw_sentence or "", sc.text)
        if cat not in skills_by_category:
            skills_by_category[cat] = []
        skills_by_category[cat].append(sc)

    # Group substantive claims into sections with drop tables
    project_claims: dict[str, list[Claim]] = {}
    experience_claims: list[Claim] = []
    achievement_claims: list[Claim] = []
    other_claims: list[Claim] = []

    for c in audit_claims:
        if c.claim_type == ClaimType.EXPERIENCE_CLAIM:
            experience_claims.append(c)
        elif c.claim_type == ClaimType.ACHIEVEMENT:
            achievement_claims.append(c)
        elif c.project_id:
            project_claims.setdefault(c.project_id, []).append(c)
        else:
            other_claims.append(c)

    facts_by_type = {f.fact_type.value: f for f in facts}
    total_project_claims_count = sum(len(plist) for plist in project_claims.values())

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "nav_active": "review",
            "user_id": user_id,
            "persona_id": persona_id,
            "claims": audit_claims,
            "all_claims": claims,
            "skill_claims": skill_claims,
            "skills_by_category": skills_by_category,
            "project_claims": project_claims,
            "total_project_claims_count": total_project_claims_count,
            "experience_claims": experience_claims,
            "achievement_claims": achievement_claims,
            "other_claims": other_claims,
            "facts": facts,
            "facts_by_type": facts_by_type,
            "active_session": active_session,
        },
    )


@app.post("/review/{user_id}/toggle_skill", response_class=HTMLResponse)
async def toggle_skill(request: Request, user_id: str, claim_id: str = Form(...)):
    """Toggles skill confirmation state cycling: Pending -> Confirmed -> Rejected -> Pending."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim = claim_store.get_claim(claim_id)
    if not claim:
        return HTMLResponse("")

    if claim.status == ClaimStatus.EXTRACTED:
        new_status = ClaimStatus.CONFIRMED
    elif claim.status == ClaimStatus.CONFIRMED:
        new_status = ClaimStatus.REJECTED
    else:
        new_status = ClaimStatus.EXTRACTED

    claim_store.update_claim_review(claim_id=claim_id, action=new_status)
    claim.status = new_status

    html = templates.get_template("partials/skill_badge.html").render(
        s=claim,
        user_id=user_id,
    )
    return HTMLResponse(html)


@app.post("/review/{user_id}/confirm_skill_category")
async def confirm_skill_category(request: Request, user_id: str, category: str = Form(...)):
    """Bulk confirms all skills within a specific category."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claims = claim_store.get_claims(user_id=user_id)
    for c in claims:
        cat = c.project_id or infer_skill_category(c.raw_sentence or "", c.text)
        if c.claim_type == ClaimType.SKILL and cat == category:
            claim_store.update_claim_review(claim_id=c.id, action=ClaimStatus.CONFIRMED)
    return RedirectResponse(url=f"/review/{user_id}", status_code=303)


@app.post("/review/{user_id}/reset_skill_category")
async def reset_skill_category(request: Request, user_id: str, category: str = Form(...)):
    """Resets all skills in a specific category back to pending (Undo Group)."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claims = claim_store.get_claims(user_id=user_id)
    for c in claims:
        cat = c.project_id or infer_skill_category(c.raw_sentence or "", c.text)
        if c.claim_type == ClaimType.SKILL and cat == category:
            claim_store.update_claim_review(claim_id=c.id, action=ClaimStatus.EXTRACTED)
    return RedirectResponse(url=f"/review/{user_id}", status_code=303)


@app.post("/review/{user_id}/confirm_project")
async def confirm_project(request: Request, user_id: str, project_id: str = Form(...)):
    """Bulk confirms all claims for a specific project."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claims = claim_store.get_claims(user_id=user_id)
    for c in claims:
        if c.project_id == project_id and c.claim_type != ClaimType.SKILL:
            claim_store.update_claim_review(claim_id=c.id, action=ClaimStatus.CONFIRMED)
    return RedirectResponse(url=f"/review/{user_id}", status_code=303)


@app.post("/review/{user_id}/reset_project")
async def reset_project(request: Request, user_id: str, project_id: str = Form(...)):
    """Resets all claims for a specific project back to pending (Undo Project)."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claims = claim_store.get_claims(user_id=user_id)
    for c in claims:
        if c.project_id == project_id and c.claim_type != ClaimType.SKILL:
            claim_store.update_claim_review(claim_id=c.id, action=ClaimStatus.EXTRACTED)
    return RedirectResponse(url=f"/review/{user_id}", status_code=303)


def _render_claim_row(c: Claim, user_id: str, persona: Optional[str] = None) -> str:
    """Renders a single resume bullet row via autoescaped Jinja2 partial."""
    return templates.get_template("partials/claim_row.html").render(
        c=c,
        user_id=user_id,
        persona_id=persona or "",
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
    return HTMLResponse(_render_claim_row(c, user_id, persona))


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
    return HTMLResponse(_render_claim_row(c, user_id, persona))


@app.post("/review/{user_id}/reset_claim", response_class=HTMLResponse)
async def reset_claim(
    request: Request,
    user_id: str,
    claim_id: str = Form(...),
    persona: Optional[str] = Form(None),
):
    """Resets a technical claim back to pending (extracted) status (Undo)."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.update_claim_review(claim_id=claim_id, action=ClaimStatus.EXTRACTED)
    c = claim_store.get_claim(claim_id)
    return HTMLResponse(_render_claim_row(c, user_id, persona))


def _render_fact_row(f: ProfileFact, user_id: str) -> str:
    """Renders a single profile fact chip via autoescaped Jinja2 partial."""
    return templates.get_template("partials/fact_row.html").render(
        f=f,
        user_id=user_id,
    )


@app.get("/review/{user_id}/claim_edit_form", response_class=HTMLResponse)
async def claim_edit_form(request: Request, user_id: str, claim_id: str, persona: Optional[str] = None):
    """Renders inline edit box for a claim bullet."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    c = claim_store.get_claim(claim_id)
    if not c:
        return HTMLResponse("")
    html = templates.get_template("partials/claim_edit_form.html").render(
        c=c,
        user_id=user_id,
        persona_id=persona or "",
    )
    return HTMLResponse(html)


@app.post("/review/{user_id}/save_claim_edit", response_class=HTMLResponse)
async def save_claim_edit(request: Request, user_id: str, claim_id: str = Form(...), edited_text: str = Form(...), persona: Optional[str] = Form(None)):
    """Saves user edit to a technical claim and returns updated resume bullet."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    c = claim_store.get_claim(claim_id)
    if not c:
        return HTMLResponse("")
    clean_text = edited_text.strip()
    if clean_text:
        claim_store.update_claim_review(claim_id=claim_id, action=c.status, user_correction_text=clean_text)
        c = claim_store.get_claim(claim_id)
    return HTMLResponse(_render_claim_row(c, user_id, persona))


@app.get("/review/{user_id}/cancel_claim_edit", response_class=HTMLResponse)
async def cancel_claim_edit(request: Request, user_id: str, claim_id: str, persona: Optional[str] = None):
    """Cancels inline editing and restores standard resume bullet row."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    c = claim_store.get_claim(claim_id)
    if not c:
        return HTMLResponse("")
    return HTMLResponse(_render_claim_row(c, user_id, persona))


@app.post("/review/{user_id}/confirm_fact", response_class=HTMLResponse)
async def confirm_fact(
    request: Request,
    user_id: str,
    fact_id: str = Form(...),
):
    """Confirms an individual profile fact."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.confirm_profile_fact(fact_id=fact_id)
    f = claim_store.get_profile_fact(fact_id)
    if not f:
        return HTMLResponse("")
    return HTMLResponse(_render_fact_row(f, user_id))


@app.post("/review/{user_id}/reject_fact", response_class=HTMLResponse)
async def reject_fact(
    request: Request,
    user_id: str,
    fact_id: str = Form(...),
):
    """Unconfirms/resets an individual profile fact to pending (Undo)."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.reject_profile_fact(fact_id=fact_id)
    f = claim_store.get_profile_fact(fact_id)
    if not f:
        return HTMLResponse("")
    return HTMLResponse(_render_fact_row(f, user_id))


@app.post("/review/{user_id}/confirm_all_facts")
async def confirm_all_facts(request: Request, user_id: str):
    """Bulk confirms all profile facts for the user."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.confirm_all_facts(user_id=user_id)
    return RedirectResponse(url=f"/review/{user_id}", status_code=303)


@app.post("/review/{user_id}/reset_all_facts")
async def reset_all_facts(request: Request, user_id: str):
    """Resets all profile facts back to pending (Undo All Facts)."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.reset_all_facts(user_id=user_id)
    return RedirectResponse(url=f"/review/{user_id}", status_code=303)


@app.post("/review/{user_id}/confirm_all")
async def confirm_all(
    request: Request,
    user_id: str,
    persona: Optional[str] = Form(None),
):
    """Bulk confirms all claims for the user. Profile facts must be confirmed individually."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.bulk_update_claims(user_id=user_id, action=ClaimStatus.CONFIRMED)
    return RedirectResponse(url=f"/review/{user_id}?persona={persona or ''}", status_code=303)


@app.post("/review/{user_id}/reject_all")
async def reject_all(
    request: Request,
    user_id: str,
    persona: Optional[str] = Form(None),
):
    """Bulk rejects all claims for the user."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.bulk_update_claims(user_id=user_id, action=ClaimStatus.REJECTED)
    return RedirectResponse(url=f"/review/{user_id}?persona={persona or ''}", status_code=303)


@app.post("/review/{user_id}/reset_all")
async def reset_all(
    request: Request,
    user_id: str,
    persona: Optional[str] = Form(None),
):
    """Resets all claims back to extracted (pending)."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    claim_store.bulk_update_claims(user_id=user_id, action=ClaimStatus.EXTRACTED)
    return RedirectResponse(url=f"/review/{user_id}?persona={persona or ''}", status_code=303)


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
    session_store: SessionStore = services["session_store"]
    effective_persona = persona or "sam_okafor"

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    session = engine.start_session(
        persona_id=effective_persona,
        user_id=user_id,
        budget=budget,
        client_ip=client_ip,
        user_agent=user_agent,
    )

    response = RedirectResponse(
        url=f"/interview/{session.id}",
        status_code=303,
    )

    # Persist or update session token
    cookie_token = request.cookies.get("meritsentinel_session")
    token_valid = False
    if cookie_token:
        try:
            payload = verify_signed_session_token(cookie_token)
            if payload.get("user_id") == user_id:
                token_valid = True
                session_store.touch_user_session(cookie_token, active_interview_id=session.id)
        except Exception:
            token_valid = False

    if not token_valid:
        token = create_signed_session_token(user_id=user_id, extra_data={"persona": effective_persona})
        session_store.save_user_session(
            session_token=token,
            user_id=user_id,
            active_interview_id=session.id,
            client_ip=client_ip,
            user_agent=user_agent,
        )
        response.set_cookie(
            key="meritsentinel_session",
            value=token,
            httponly=True,
            samesite="lax",
            path="/",
            max_age=604800,
        )

    return response


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
    else:
        # Finalize if not already finalized
        if session.status == "active":
            engine.finalize_session(session_id)
            session = session_store.get_session(session_id) or session
            turns = session_store.get_turns_for_session(session_id)
            has_pending_grades = any(t.grade is None for t in turns)

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
            "active_session": session if session.status == "active" else None,
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
    """Submits answer with async background grading, security screening, and advances the session."""
    services = app.state.services
    engine: InterviewEngine = services["engine"]
    session_store: SessionStore = services["session_store"]

    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Interview session {session_id} not found")
    if session.status != "active" or len(session.turns) >= session.budget:
        return RedirectResponse(url=f"/interview/{session_id}", status_code=303)

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Security screen candidate answer for adversarial injection / obfuscation
    adv_scan = screen_candidate_answer(answer_text)
    is_adversarial = adv_scan["is_adversarial"]
    adversarial_reasons = adv_scan["reasons"]

    if session.current_question is not None:
        question = session.current_question
    else:
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
        client_ip=client_ip,
        user_agent=user_agent,
        is_adversarial=is_adversarial,
        adversarial_reasons=adversarial_reasons,
    )

    return RedirectResponse(
        url=f"/interview/{session_id}",
        status_code=303,
    )


@app.post("/interview/{session_id}/rephrase", response_class=HTMLResponse)
async def rephrase_interview_question(request: Request, session_id: str):
    """Rephrases the current pending interview question upon candidate request."""
    services = app.state.services
    engine: InterviewEngine = services["engine"]
    session_store: SessionStore = services["session_store"]
    claim_store: ClaimStore = services["claim_store"]

    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Interview session {session_id} not found")

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    rephrased_q = engine.rephrase_current_question(
        session_id=session_id,
        client_ip=client_ip,
        user_agent=user_agent,
    )

    if not rephrased_q:
        rephrased_q = engine.get_next_question(session_id)

    if not rephrased_q:
        if request.headers.get("hx-request") == "true":
            return HTMLResponse("")
        return RedirectResponse(url=f"/interview/{session_id}", status_code=303)

    # Check if this is an HTMX request
    is_htmx = request.headers.get("hx-request") == "true"
    if not is_htmx:
        return RedirectResponse(url=f"/interview/{session_id}", status_code=303)

    turns = session_store.get_turns_for_session(session_id)
    claim_obj = claim_store.get_claim(rephrased_q.claim_id) if rephrased_q else None
    target_claim_text = claim_obj.text if claim_obj else "Technical Claim"

    return templates.TemplateResponse(
        request=request,
        name="partials/question_container.html",
        context={
            "session": session,
            "current_question": rephrased_q,
            "target_claim_text": target_claim_text,
            "turns": turns,
        },
    )


@app.get("/interview/{session_id}/rephrase")
async def rephrase_get_fallback(session_id: str):
    """Fallback redirect for GET requests to rephrase endpoint."""
    return RedirectResponse(url=f"/interview/{session_id}", status_code=303)


@app.get("/interview/{session_id}/logs", response_class=HTMLResponse)
async def interview_logs(request: Request, session_id: str):
    """Audit log viewer for an interview session."""
    services = app.state.services
    session_store: SessionStore = services["session_store"]
    session_logger: SessionLogger = services["session_logger"]

    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Interview session {session_id} not found")

    logs = session_logger.get_session_logs(session_id)
    security_event_count = sum(1 for log in logs if "SECURITY" in log.get("event_type", ""))

    return templates.TemplateResponse(
        request=request,
        name="session_logs.html",
        context={
            "nav_active": "interview",
            "session": session,
            "events": logs,
            "logs": logs,
            "security_event_count": security_event_count,
            "user_id": session.user_id,
            "active_session": session if session.status == "active" else None,
        },
    )


@app.get("/interview/{session_id}/logs/export")
async def export_interview_logs(session_id: str):
    """Exports structured audit logs as JSON."""
    services = app.state.services
    session_logger: SessionLogger = services["session_logger"]
    session_store: SessionStore = services["session_store"]

    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Interview session {session_id} not found")

    json_data = session_logger.export_session_logs_json(session_id)
    return Response(
        content=json_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=meritsentinel_audit_{session_id[:8]}.json"
        },
    )


@app.get("/interview/{session_id}/grade_status", response_class=HTMLResponse)
async def grade_status(session_id: str, was_pending: Optional[str] = None):
    """HTMX polling endpoint returning background grading badge and refreshing turns when finished."""
    services = app.state.services
    session_store: SessionStore = services["session_store"]

    turns = session_store.get_turns_for_session(session_id)
    has_pending = any(t.grade is None for t in turns)

    if has_pending:
        return HTMLResponse(
            f"""
            <div id="grade-status" hx-get="/interview/{session_id}/grade_status?was_pending=1" hx-trigger="every 1s" hx-swap="outerHTML" class="badge badge-pending">
              ⏳ Background Grading Active
            </div>
            """
        )
    else:
        headers = {"HX-Refresh": "true"} if was_pending == "1" else {}
        return HTMLResponse(
            """
            <div id="grade-status" class="badge badge-success">
              ✓ All Turns Graded
            </div>
            """,
            headers=headers,
        )


@app.get("/dashboard/{user_id}", response_class=HTMLResponse)
async def dashboard(request: Request, user_id: str):
    """Mastery Matrix Dashboard displaying claims, verification badges, scores, and defensibility."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    session_store: SessionStore = services["session_store"]
    active_session = session_store.get_active_session_for_user(user_id)

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
    interviewed_claims = [r for r in matrix_rows if r["answers_count"] > 0]
    interviewed_verified = sum(1 for r in interviewed_claims if r["score"] >= 3)
    interviewed_defensibility_pct = (
        int(round((interviewed_verified / len(interviewed_claims) * 100)))
        if interviewed_claims
        else 0
    )
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
            "interviewed_count": len(interviewed_claims),
            "interviewed_defensibility_pct": interviewed_defensibility_pct,
            "matrix_rows": matrix_rows,
            "active_session": active_session,
        },
    )


SAMPLE_JD = """Looking for a Systems / Backend Software Engineer with experience in:
- High-throughput RESTful API microservices with FastAPI and database connection pooling
- Low-latency network programming, socket handling, and custom packet decoding
- Concurrent distributed systems, in-memory caching, and performance benchmarking
- Python, SQLite, Redis, and Linux systems programming
"""


@app.get("/tailor/{user_id}", response_class=HTMLResponse)
async def tailor_page(request: Request, user_id: str):
    """Resume Tailoring and Strict Verifier screen."""
    services = app.state.services
    session_store: SessionStore = services["session_store"]
    active_session = session_store.get_active_session_for_user(user_id)

    cached = getattr(app.state, "tailoring_cache", {}).get(user_id)
    jd_text = cached.get("jd_text") if cached else SAMPLE_JD
    gap_report = cached.get("gap_report") if cached else None
    verifier_report = cached.get("verifier_report") if cached else None
    tailored_markdown = cached.get("tailored_markdown") if cached else None

    return templates.TemplateResponse(
        request=request,
        name="tailor.html",
        context={
            "nav_active": "tailor",
            "user_id": user_id,
            "jd_text": jd_text,
            "sample_jd": SAMPLE_JD,
            "gap_report": gap_report,
            "verifier_report": verifier_report,
            "tailored_markdown": tailored_markdown,
            "active_session": active_session,
        },
    )


@app.post("/tailor/{user_id}", response_class=HTMLResponse)
async def run_tailoring(
    request: Request,
    user_id: str,
    jd_text: str = Form(...),
):
    """Runs deterministic requirement matching, tailoring engine, and strict verification."""
    services = app.state.services
    claim_store: ClaimStore = services["claim_store"]
    session_store: SessionStore = services["session_store"]
    active_session = session_store.get_active_session_for_user(user_id)

    claims = claim_store.get_claims(user_id=user_id)
    facts = claim_store.get_profile_facts(user_id=user_id)

    # Auto-confirm profile facts if none are confirmed yet
    if facts and not any(f.is_confirmed for f in facts):
        for f in facts:
            claim_store.confirm_profile_fact(f.id)
        facts = claim_store.get_profile_facts(user_id=user_id)

    mastery_map = {}
    for c in claims:
        m = claim_store.get_mastery_state(c.id)
        if m:
            mastery_map[c.id] = m

    engine = TailoringEngine()
    tailored_result = engine.tailor(
        jd_text=jd_text,
        facts=facts,
        claims=claims,
        mastery_map=mastery_map,
    )

    verifier = StrictVerifier()
    verifier_report = verifier.verify(
        resume_text=tailored_result.tailored_markdown,
        eligible_claims=tailored_result.included_claims,
        confirmed_facts=tailored_result.included_facts,
    )

    if not hasattr(app.state, "tailoring_cache"):
        app.state.tailoring_cache = {}

    app.state.tailoring_cache[user_id] = {
        "jd_text": jd_text,
        "gap_report": tailored_result.gap_report,
        "tailored_result": tailored_result,
        "tailored_markdown": tailored_result.tailored_markdown,
        "verifier_report": verifier_report,
        "eligible_claims": tailored_result.included_claims,
        "confirmed_facts": tailored_result.included_facts,
    }

    return templates.TemplateResponse(
        request=request,
        name="tailor.html",
        context={
            "nav_active": "tailor",
            "user_id": user_id,
            "jd_text": jd_text,
            "sample_jd": SAMPLE_JD,
            "gap_report": tailored_result.gap_report,
            "verifier_report": verifier_report,
            "tailored_markdown": tailored_result.tailored_markdown,
            "active_session": active_session,
        },
    )


@app.get("/tailor/{user_id}/download_md")
async def download_markdown(user_id: str):
    """Downloads tailored resume draft in Markdown format."""
    cached = getattr(app.state, "tailoring_cache", {}).get(user_id)
    if not cached or not cached.get("tailored_markdown"):
        raise HTTPException(
            status_code=404,
            detail=f"No tailored resume draft found for user {user_id}. Run tailoring first."
        )

    verifier_report = cached.get("verifier_report")
    if not verifier_report or not verifier_report.passed:
        blocked_count = verifier_report.blocked_count if verifier_report else 1
        raise HTTPException(
            status_code=400,
            detail=f"Export blocked: {blocked_count} sentence(s) failed strict verification. Zero unverified claims permitted."
        )

    md_content = cached["tailored_markdown"]
    safe_uid = sanitize_filename(user_id)
    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="resume_{safe_uid}.md"'},
    )


@app.get("/tailor/{user_id}/download_pdf")
async def download_pdf(user_id: str):
    """Compiles and downloads single-page ATS PDF if and only if strict verification passed."""
    cached = getattr(app.state, "tailoring_cache", {}).get(user_id)
    if not cached or not cached.get("tailored_markdown"):
        raise HTTPException(
            status_code=404,
            detail=f"No tailored resume draft found for user {user_id}. Run tailoring first."
        )

    verifier_report = cached.get("verifier_report")
    if not verifier_report or not verifier_report.passed:
        blocked_count = verifier_report.blocked_count if verifier_report else 1
        raise HTTPException(
            status_code=400,
            detail=f"Export blocked: {blocked_count} sentence(s) failed strict verification. Zero unverified claims permitted."
        )

    try:
        export_dir = settings.export_dir
        export_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        export_dir = Path(tempfile.gettempdir())

    safe_uid = sanitize_filename(user_id)
    pdf_path = export_dir / f"resume_{safe_uid}.pdf"
    export_pdf(
        markdown_content=cached["tailored_markdown"],
        verifier_report=verifier_report,
        output_path=pdf_path,
    )

    pdf_bytes = pdf_path.read_bytes()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="resume_{safe_uid}.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("resumeproof.ui.app:app", host=settings.ui_host, port=settings.ui_port, reload=False)

