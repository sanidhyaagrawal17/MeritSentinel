"""CLI interface for ResumeProof: parse resume, review claims, and store audit logs."""

from pathlib import Path
import sys
import click

from resumeproof.core.config import settings
from resumeproof.core.db import Database
from resumeproof.core.llm import OpenAICompatibleClient, FakeLLMClient
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.schemas.claims import ClaimStatus
from resumeproof.store.claim_store import ClaimStore
from resumeproof.review.flow import ReviewSession, ReviewDecision

@click.group()
def cli():
    """ResumeProof: Truth-First Agentic Resume & Mock Interview Engine."""
    pass

@cli.command("extract")
@click.option("--resume", "-r", required=True, type=click.Path(exists=True), help="Path to resume PDF or Markdown file")
@click.option("--user", "-u", default="default_user", help="User ID")
@click.option("--persona", "-p", default=None, help="Demo Persona ID (alex_rivera, priya_nair, sam_okafor)")
@click.option("--auto-confirm", is_flag=True, default=False, help="Automatically confirm all extracted claims without prompt")
@click.option("--fake-llm", is_flag=True, default=False, help="Use offline FakeLLMClient")
def extract_cmd(resume: str, user: str, persona: str | None, auto_confirm: bool, fake_llm: bool):
    """Extract and review claims and profile facts from a resume."""
    click.echo(f"==> Parsing resume: {resume}")
    parsed = parse_resume(resume)
    click.echo(f"==> Extracted {len(parsed.lines)} lines using {parsed.extractor_used}")

    if fake_llm or settings.use_fake_llm:
        client = FakeLLMClient()
    else:
        client = OpenAICompatibleClient(base_url=settings.ollama_base_url)

    click.echo("==> Running structured fact and claim extraction...")
    facts, claims = extract_from_resume(
        parse_result=parsed,
        llm_client=client,
        persona_id=persona,
        user_id=user,
        model=settings.grader_model
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

if __name__ == "__main__":
    cli()
