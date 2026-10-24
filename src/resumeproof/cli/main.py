"""CLI interface for ResumeProof: parse resume, review claims, and store audit logs."""

from pathlib import Path
import sys
import time
import click

from resumeproof.core.config import settings
from resumeproof.core.db import Database
from resumeproof.core.llm import OpenAICompatibleClient, FakeLLMClient
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.schemas.claims import ClaimStatus
from resumeproof.store.claim_store import ClaimStore
from resumeproof.review.flow import ReviewSession, ReviewDecision
from resumeproof.rubric.builder import build_rubrics_for_persona
from resumeproof.rubric.store import RubricStore
from resumeproof.interview.generator import QuestionGenerator
from resumeproof.interview.grader import AnswerGrader
from resumeproof.interview.caching import GradingCache
from resumeproof.interview.background import BackgroundGrader
from resumeproof.interview.session_store import SessionStore
from resumeproof.interview.engine import InterviewEngine

@click.group()
def cli():
    """ResumeProof: Truth-First Agentic Resume & Mock Interview Engine."""
    pass

@cli.command("extract")
@click.option("--resume", "-r", required=True, type=click.Path(exists=True), help="Path to resume PDF or Markdown file")
@click.option("--user", "-u", default="default_user", help="User ID")
@click.option("--persona", "-p", default=None, help="Demo Persona ID (alex_rivera, priya_nair, sam_okafor)")
@click.option("--model", "-m", default=None, help="Model name for extraction (default: settings.extractor_model / qwen2.5:3b)")
@click.option("--auto-confirm", is_flag=True, default=False, help="Automatically confirm all extracted claims without prompt")
@click.option("--fake-llm", is_flag=True, default=False, help="Use offline FakeLLMClient")
def extract_cmd(resume: str, user: str, persona: str | None, model: str | None, auto_confirm: bool, fake_llm: bool):
    """Extract and review claims and profile facts from a resume."""
    click.echo(f"==> Parsing resume: {resume}")
    parsed = parse_resume(resume)
    click.echo(f"==> Extracted {len(parsed.lines)} lines using {parsed.extractor_used}")

    if fake_llm or settings.use_fake_llm:
        client = FakeLLMClient()
    else:
        client = OpenAICompatibleClient(base_url=settings.ollama_base_url)

    ext_model = model or settings.extractor_model
    click.echo(f"==> Running structured fact and claim extraction with model: {ext_model}...")
    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id=persona,
        user_id=user,
        model=ext_model
    )
    click.echo(f"==> Extracted {len(facts)} profile facts and {len(claims)} mastery claims.\n")

    db = Database(settings.db_path)
    try:
        store = ClaimStore(db)
        session = ReviewSession(store)

        if auto_confirm:
            click.echo("==> Auto-confirming all facts and claims...")
            rev_facts, rev_claims = session.run_batch_review(facts, claims, auto_confirm_all=True)
            click.echo(f"==> Successfully stored {len(rev_facts)} profile facts and {len(rev_claims)} claims in SQLite.")
            return

        # Interactive Review Step
        click.echo("=" * 60)
        click.echo("STAGE 1: PROFILE FACTS REVIEW (Not interview-gated)")
        click.echo("=" * 60)

        for idx, fact in enumerate(facts, start=1):
            click.echo(f"\n[{idx}/{len(facts)}] {fact.fact_type.value.upper()}: {fact.value} (Line: {fact.source_line})")
            choice = click.prompt("Action: [c]onfirm, [e]dit, [r]eject", type=click.Choice(["c", "e", "r"]), default="c")
            if choice == "c":
                session.review_profile_fact(fact, confirm=True)
                click.secho("Confirmed.", fg="green")
            elif choice == "e":
                new_val = click.prompt("Enter corrected value", default=fact.value)
                session.review_profile_fact(fact, confirm=True, correction=new_val)
                click.secho(f"Edited and confirmed: {new_val}", fg="yellow")
            else:
                session.review_profile_fact(fact, confirm=False)
                click.secho("Rejected.", fg="red")

        click.echo("\n" + "=" * 60)
        click.echo("STAGE 2: MASTERY CLAIMS REVIEW (Interview-gated)")
        click.echo("=" * 60)

        for idx, claim in enumerate(claims, start=1):
            click.echo(f"\n[{idx}/{len(claims)}] {claim.claim_type.value.upper()} (Line: {claim.source_line})")
            click.echo(f"  Statement: {claim.text}")
            if claim.tools:
                click.echo(f"  Tools: {', '.join(claim.tools)}")
            if claim.metric_values:
                click.echo(f"  Metrics: {', '.join(claim.metric_values)}")

            choice = click.prompt("Action: [c]onfirm, [e]dit, [r]eject", type=click.Choice(["c", "e", "r"]), default="c")
            if choice == "c":
                session.review_claim(claim, ReviewDecision(action=ClaimStatus.CONFIRMED))
                click.secho("Confirmed into claim store.", fg="green")
            elif choice == "e":
                new_text = click.prompt("Corrected text", default=claim.text)
                tools_str = click.prompt("Tools (comma-separated)", default=", ".join(claim.tools))
                metrics_str = click.prompt("Metrics (comma-separated)", default=", ".join(claim.metric_values))
                new_tools = [t.strip() for t in tools_str.split(",") if t.strip()]
                new_metrics = [m.strip() for m in metrics_str.split(",") if m.strip()]
                session.review_claim(claim, ReviewDecision(
                    action=ClaimStatus.EDITED,
                    user_text=new_text,
                    user_tools=new_tools,
                    user_metrics=new_metrics
                ))
                click.secho("Edited claim saved with full audit trail.", fg="yellow")
            else:
                session.review_claim(claim, ReviewDecision(action=ClaimStatus.REJECTED))
                click.secho("Rejected claim (excluded from interview pool).", fg="red")

        click.echo("\n==> Review complete! All facts and claims saved to SQLite.")
    finally:
        db.close()

@cli.command("interview")
@click.option("--persona", "-p", required=True, help="Demo Persona ID (sam_okafor, alex_rivera, priya_nair)")
@click.option("--user", "-u", default="default_user", help="User ID")
@click.option("--budget", "-b", default=5, type=int, help="Question budget")
@click.option("--auto-answer", is_flag=True, default=False, help="Automatically answer using persona sample answers (scripted test run)")
@click.option("--interactive", "-i", is_flag=True, default=False, help="Interactive mode where user types answers at CLI prompt")
@click.option("--async-grade", is_flag=True, default=False, help="Grade in background asynchronously without blocking per turn")
@click.option("--fake-llm", is_flag=True, default=False, help="Use offline FakeLLMClient")
def interview_cmd(persona: str, user: str, budget: int, auto_answer: bool, interactive: bool, async_grade: bool, fake_llm: bool):
    """Run an adaptive mock interview session for a persona with 3B generator and 7B grader."""
    is_scripted = auto_answer and not interactive
    mode_label = "[SCRIPTED TEST RUN]" if is_scripted else "[INTERACTIVE CANDIDATE SESSION]"

    click.echo(f"\n============================================================")
    click.echo(f"  RESUMEPROOF ADAPTIVE INTERVIEW SESSION: {persona}")
    click.echo(f"  Execution Mode: {mode_label}")
    click.echo(f"============================================================")

    db = Database(settings.db_path)
    try:
        claim_store = ClaimStore(db)
        rubric_store = RubricStore(db)
        session_store = SessionStore(db, claim_store)

        # Step 1: Ensure rubrics are built and saved for this persona
        click.echo(f"==> Loading project rubrics for persona: {persona}...")
        rubrics = build_rubrics_for_persona(persona)
        rubric_store.save_rubrics(rubrics)
        click.echo(f"==> Registered {len(rubrics)} grounded rubric items in SQLite.")

        # Step 2: Ensure claims exist for persona in claim store
        claims = claim_store.get_claims(user_id=user, persona_id=persona)
        if not claims:
            click.echo(f"==> No claims found in store for {persona}. Auto-extracting from resume...")
            resume_path = Path("data/demo") / persona / "resume.pdf"
            if not resume_path.exists():
                resume_path = Path("data/demo") / persona / "resume.md"
            parsed = parse_resume(resume_path)
            temp_client = FakeLLMClient() if (fake_llm or settings.use_fake_llm) else OpenAICompatibleClient(base_url=settings.ollama_base_url)
            facts, raw_claims = extract_from_resume(
                parse_result=parsed,
                llm_client=temp_client,
                persona_id=persona,
                user_id=user,
                model=settings.extractor_model
            )
            rev_session = ReviewSession(claim_store)
            rev_session.run_batch_review(facts, raw_claims, auto_confirm_all=True)
            claims = claim_store.get_claims(user_id=user, persona_id=persona)
            click.echo(f"==> Initialized and confirmed {len(claims)} claims into pool.")

        # Step 3: Setup LLM Clients
        if fake_llm or settings.use_fake_llm:
            click.echo("==> Running in offline mode with FakeLLMClient (recorded fixtures).")
            gen_client = FakeLLMClient()
            grader_client = FakeLLMClient()
        else:
            click.echo(f"==> Connecting to Ollama: Interviewer={settings.interviewer_model}, Grader={settings.grader_model}")
            gen_client = OpenAICompatibleClient(base_url=settings.ollama_base_url)
            grader_client = OpenAICompatibleClient(base_url=settings.ollama_base_url)

        generator = QuestionGenerator(llm_client=gen_client, model=settings.interviewer_model)
        grader = AnswerGrader(llm_client=grader_client, model=settings.grader_model)
        cache = GradingCache(db)
        bg_grader = BackgroundGrader(grader=grader, cache=cache)

        engine = InterviewEngine(
            claim_store=claim_store,
            rubric_store=rubric_store,
            session_store=session_store,
            generator=generator,
            background_grader=bg_grader,
        )

        # Load sample answers if in auto_answer mode
        sample_answers = {}
        sample_path = Path("data/demo") / persona / "sample_answers.json"
        if sample_path.exists():
            import json
            sample_answers = json.loads(sample_path.read_text(encoding="utf-8"))

        session = engine.start_session(persona_id=persona, user_id=user, budget=budget)
        click.echo(f"==> Interview session created [ID: {session.id[:8]}...], budget: {budget} questions.\n")

        turn_idx = 1
        sample_cycle = ["strong", "terse", "strong", "generic", "confidently_wrong"]
        wait_for_grade = not async_grade

        while True:
            t0_q = time.perf_counter()
            question = engine.get_next_question(session.id)
            dt_q_ms = (time.perf_counter() - t0_q) * 1000.0

            if question is None:
                click.echo(f"\n[Interview Session Concluded: Budget of {budget} questions reached]")
                break

            click.echo("-" * 65)
            followup_tag = " [FOLLOW-UP]" if question.is_followup else ""
            click.secho(
                f"TURN {turn_idx}/{budget} | Priority: {question.priority.value.upper()}{followup_tag} | Latency: {dt_q_ms:.1f}ms",
                fg="cyan",
                bold=True
            )
            click.echo(f"Interviewer (3B): {question.question_text}")

            if is_scripted:
                ans_key = sample_cycle[(turn_idx - 1) % len(sample_cycle)]
                answer_text = None
                # Check for project-specific sample answer
                if question.project_id and question.project_id in sample_answers and isinstance(sample_answers[question.project_id], dict):
                    answer_text = sample_answers[question.project_id].get(ans_key)

                if not answer_text:
                    q_lower = (question.question_text + " " + (question.claim_id or "")).lower()
                    for proj_key, proj_dict in sample_answers.items():
                        if isinstance(proj_dict, dict) and any(kw in q_lower for kw in proj_key.split("-")):
                            answer_text = proj_dict.get(ans_key)
                            if answer_text:
                                break

                if not answer_text:
                    answer_text = sample_answers.get(ans_key, "I implemented this using verified project architecture.")

                click.secho(f"Candidate: {answer_text}", fg="white")
            else:
                answer_text = click.prompt("Your Answer", type=str)

            if wait_for_grade:
                click.echo("\n[Grading answer with 7B model at Temp 0...]")
            else:
                click.secho("\n[Background grading queued: Next question unblocked on CPU]", fg="magenta")

            turn = engine.submit_answer(session.id, question, answer_text, wait_for_grade=wait_for_grade)

            if turn.grade:
                score_color = "green" if turn.grade.raw_score >= 3 else ("yellow" if turn.grade.raw_score == 2 else "red")
                cached_tag = " (CACHED)" if turn.grade.cached else ""
                click.secho(
                    f"Grade: {turn.grade.raw_score} / 4{cached_tag} | Latency: {turn.turn_latency_ms:.1f}ms",
                    fg=score_color,
                    bold=True
                )
                click.echo(f"Evidence Quote: \"{turn.grade.evidence_quote}\"")
                click.echo(f"Rubric Anchor:  \"{turn.grade.rubric_reference}\"")
                if turn.grade.matched_rubric_ids:
                    click.echo(f"Matched Rubrics: {', '.join(turn.grade.matched_rubric_ids)}")
                if turn.grade.missed_rubric_ids:
                    click.echo(f"Missed Rubrics:  {', '.join(turn.grade.missed_rubric_ids)}")
                click.echo(f"Critique:       {turn.grade.feedback}")

            # Check updated mastery state for the claim
            m = claim_store.get_mastery_state(question.claim_id)
            if m:
                status_str = "ELIGIBLE" if m.is_eligible else "LOCKED"
                status_color = "green" if m.is_eligible else "red"
                click.secho(
                    f"Mastery Status: {status_str} | Score: {m.effective_score:.2f}/4.0 (Agg: {m.aggregate_score:.2f}) | Conf: {m.confidence:.3f} (n_eff: {m.n_effective:.2f})",
                    fg=status_color
                )
            click.echo()
            turn_idx += 1

        finalized = engine.finalize_session(session.id)
        click.echo("=" * 65)
        click.secho(f"INTERVIEW TRANSCRIPT SUMMARY {mode_label}", bold=True)
        click.echo(f"Session ID: {finalized.id}")
        click.echo(f"Total Turns Completed: {finalized.total_questions}")
        click.echo(f"Status: {finalized.status}")

        total_latency = sum(t.turn_latency_ms for t in finalized.turns)
        click.echo(f"Total Session Turn Latency: {total_latency:.1f}ms (Avg: {total_latency/max(1, len(finalized.turns)):.1f}ms/turn)")
        click.echo("=" * 65)

    finally:
        db.close()

if __name__ == "__main__":
    cli()
