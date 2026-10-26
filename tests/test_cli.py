"""Tests for CLI extract command and interactive review prompts using Click CliRunner."""

from click.testing import CliRunner
from resumeproof.cli.main import cli
from resumeproof.core.llm import FakeLLMClient
from tests.fixtures.demo_fixtures import DEMO_FIXTURES
from pathlib import Path

DEMO_DIR = Path("data/demo")

def test_cli_extract_auto_confirm(tmp_path, monkeypatch):
    test_db = tmp_path / "cli_test.db"
    monkeypatch.setenv("RESUMEPROOF_DB_PATH", str(test_db))
    monkeypatch.setattr("resumeproof.cli.main.FakeLLMClient", lambda: FakeLLMClient(recorded_fixtures=DEMO_FIXTURES))

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
    assert "Extracted 10 profile facts and 25 mastery claims." in result.output
    assert "Successfully stored 10 profile facts and 25 claims in SQLite." in result.output

def test_cli_extract_interactive_confirm_and_edit(tmp_path, monkeypatch):
    test_db = tmp_path / "cli_interactive.db"
    monkeypatch.setenv("RESUMEPROOF_DB_PATH", str(test_db))
    monkeypatch.setattr("resumeproof.cli.main.FakeLLMClient", lambda: FakeLLMClient(recorded_fixtures=DEMO_FIXTURES))

    runner = CliRunner()
    # Provide inputs for facts and claims:
    # 10 facts (c for all)
    # 25 claims (first is c, second is e with text, tools, metrics, then c for remaining 23)
    user_inputs = (
        ["c"] * 10 +
        ["c"] +
        ["e", "Corrected write throughput benchmark", "Go, Raft", "45k req/sec"] +
        ["c"] * 23
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
    assert "Execution Mode: [SCRIPTED ANSWERS]" in result.output
    assert "Candidate [SCRIPTED ANSWERS]:" in result.output
    assert "Registered" in result.output and "grounded rubric items" in result.output
    assert "TURN 1/3" in result.output
    assert "TURN 2/3" in result.output
    assert "TURN 3/3" in result.output
    assert "Claim ID:" in result.output
    assert "Rubric Item ID:" in result.output
    assert "Grade:" in result.output
    assert "INTERVIEW TRANSCRIPT SUMMARY [SCRIPTED ANSWERS]" in result.output
    assert "Total Turns Completed: 3" in result.output

def test_cli_interview_interactive_mode(tmp_path, monkeypatch):
    test_db = tmp_path / "cli_interactive_interview.db"
    monkeypatch.setenv("RESUMEPROOF_DB_PATH", str(test_db))

    runner = CliRunner()
    user_inputs = [
        "In Task-Tracker API, SQLite was configured with WAL mode and SQLAlchemy connection pool of 10 connections.",
        "We implemented recursive UDP DNS resolution in Python sockets with struct parsing.",
    ]
    stdin_data = "\n".join(user_inputs) + "\n"

    result = runner.invoke(cli, [
        "interview",
        "--persona", "sam_okafor",
        "--budget", "2",
        "--interactive",
        "--fake-llm",
    ], input=stdin_data)

    assert result.exit_code == 0
    assert "Execution Mode: [INTERACTIVE CANDIDATE SESSION]" in result.output
    assert "Your Answer:" in result.output
    assert "INTERVIEW TRANSCRIPT SUMMARY [INTERACTIVE CANDIDATE SESSION]" in result.output
    assert "Total Turns Completed: 2" in result.output

def test_cli_extract_and_interview_data_dir(tmp_path, monkeypatch):
    """Requirement 6: Test --data-dir option for extract and interview commands."""
    test_db = tmp_path / "cli_data_dir.db"
    monkeypatch.setenv("RESUMEPROOF_DB_PATH", str(test_db))

    user_data_dir = tmp_path / "local_candidate"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    proj_dir = user_data_dir / "projects" / "custom_service"
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Create local resume
    resume_file = user_data_dir / "resume.md"
    resume_file.write_text(
        "# Local User\n"
        "local.user@example.com | (555) 123-4567\n"
        "## Education\n"
        "B.S. in Computer Science\n"
        "## Projects\n"
        "Custom Service:\n"
        "- Sustained 500 req/sec write throughput with local cache.\n"
        "## Skills\n"
        "Languages: Python, Go\n",
        encoding="utf-8"
    )

    # Create local project README
    readme_file = proj_dir / "README.md"
    readme_file.write_text(
        "# Custom Service\n"
        "High performance local service.\n"
        "## Benchmarks\n"
        "- Sustained 500 req/sec write throughput with local cache.\n",
        encoding="utf-8"
    )

    runner = CliRunner()
    # Test extract with --data-dir
    ext_result = runner.invoke(cli, [
        "extract",
        "--data-dir", str(user_data_dir),
        "--auto-confirm",
        "--fake-llm",
    ])
    assert ext_result.exit_code == 0
    assert "Parsing resume:" in ext_result.output
    assert "Auto-confirming all facts and claims" in ext_result.output

    # Test interview with --data-dir
    interview_result = runner.invoke(cli, [
        "interview",
        "--data-dir", str(user_data_dir),
        "--budget", "1",
        "--auto-answer",
        "--fake-llm",
    ])
    assert interview_result.exit_code == 0
    assert "RESUMEPROOF ADAPTIVE INTERVIEW SESSION: local_candidate" in interview_result.output
    assert "TURN 1/1" in interview_result.output
    assert "Claim ID:" in interview_result.output
    assert "Rubric Item ID:" in interview_result.output
    assert "INTERVIEW TRANSCRIPT SUMMARY [SCRIPTED ANSWERS]" in interview_result.output


