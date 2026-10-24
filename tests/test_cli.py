"""Tests for CLI extract command and interactive review prompts using Click CliRunner."""

from click.testing import CliRunner
from resumeproof.cli.main import cli
from pathlib import Path

DEMO_DIR = Path("data/demo")

def test_cli_extract_auto_confirm(tmp_path, monkeypatch):
    test_db = tmp_path / "cli_test.db"
    monkeypatch.setenv("RESUMEPROOF_DB_PATH", str(test_db))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "extract",
        "--resume", str(DEMO_DIR / "alex_rivera" / "resume.pdf"),
        "--persona", "alex_rivera",
        "--auto-confirm",
        "--fake-llm"
    ])

    assert result.exit_code == 0
    assert "Parsing resume:" in result.output
    assert "Extracted 10 profile facts and 7 mastery claims." in result.output
    assert "Successfully stored 10 profile facts and 7 claims in SQLite." in result.output

def test_cli_extract_interactive_confirm_and_edit(tmp_path, monkeypatch):
    test_db = tmp_path / "cli_interactive.db"
    monkeypatch.setenv("RESUMEPROOF_DB_PATH", str(test_db))

    runner = CliRunner()
    # Provide inputs for facts and claims:
    # 10 facts (c for all)
    # 7 claims (first is c, second is e with text, tools, metrics, then c for rest)
    user_inputs = (
        ["c"] * 10 +
        ["c"] +
        ["e", "Corrected write throughput benchmark", "Go, Raft", "45k req/sec"] +
        ["c"] * 5
    )
    stdin_data = "\n".join(user_inputs) + "\n"

    result = runner.invoke(cli, [
        "extract",
        "--resume", str(DEMO_DIR / "alex_rivera" / "resume.pdf"),
        "--persona", "alex_rivera",
        "--fake-llm"
    ], input=stdin_data)

    assert result.exit_code == 0
    assert "STAGE 1: PROFILE FACTS REVIEW" in result.output
    assert "STAGE 2: MASTERY CLAIMS REVIEW" in result.output
    assert "Review complete! All facts and claims saved to SQLite." in result.output

def test_cli_interview_auto_answer(tmp_path, monkeypatch):
    test_db = tmp_path / "cli_interview.db"
    monkeypatch.setenv("RESUMEPROOF_DB_PATH", str(test_db))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "interview",
        "--persona", "sam_okafor",
        "--budget", "3",
        "--auto-answer",
        "--fake-llm",
    ])

    assert result.exit_code == 0
    assert "RESUMEPROOF ADAPTIVE INTERVIEW SESSION: sam_okafor" in result.output
    assert "Registered" in result.output and "grounded rubric items" in result.output
    assert "TURN 1/3" in result.output
    assert "TURN 2/3" in result.output
    assert "TURN 3/3" in result.output
    assert "Grade:" in result.output
    assert "INTERVIEW TRANSCRIPT SUMMARY" in result.output
    assert "Total Turns Completed: 3" in result.output
