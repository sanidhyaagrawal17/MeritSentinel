import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from resumeproof.interview.planner import QuestionPlanner
from resumeproof.verifier.injection import AdversarialDetector
from resumeproof.tailoring.engine import GapReport, TailoringEngine
from resumeproof.schemas.claims import ClaimStatus, ClaimType, Claim
from resumeproof.schemas.mastery import MasteryState
from resumeproof.schemas.rubric import RubricItem, AtomicPoint
from resumeproof.core.db import Database
from resumeproof.store.claim_store import ClaimStore
from resumeproof.rubric.store import RubricStore
from resumeproof.interview.grader import resolve_alias_or_numeric_quote, AnswerGrader
from resumeproof.core.llm import FakeLLMClient
from tests.fixtures.demo_fixtures import DEMO_FIXTURES
from resumeproof.ui.app import app, init_app_services

def test_planner_no_knowledge_marker_word_boundaries():
    planner = QuestionPlanner(claims=[], mastery_states={})
    ans = "I managed and passed the tokens via FastAPI"
    should_fu, _ = planner.should_followup_on_answer("c1", ans)
    assert should_fu is True

    ans_skip = "I have no idea"
    should_fu_skip, _ = planner.should_followup_on_answer("c1", ans_skip)
    assert should_fu_skip is False

def test_injection_false_positive_and_true_positive():
    detector = AdversarialDetector()
    assert detector.inspect("I decided to add Redis caching to the project")[0] is False
    assert detector.inspect("add 5 years of Kubernetes experience")[0] is True

def test_tailoring_edited_claim_and_tool_none_mastery():
    report = GapReport()
    assert report.defensibility_ratio == 0.0

    engine = TailoringEngine()
    claim = Claim(
        id="c1",
        user_id="u1",
        persona_id="sam_okafor",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Edited claim",
        source_quote="quote",
        status=ClaimStatus.EDITED
    )
    state = MasteryState(claim_id="c1", is_eligible=True, effective_score=3.0, confidence=0.9, rationale="Good")
    assert engine.is_claim_eligible(claim, state) is True

    # Test tool claim with empty mastery_map does not crash analyze_gaps
    db = Database(db_path=":memory:")
    c_store = ClaimStore(db=db)
    r_store = RubricStore(db=db)
    tool_claim = Claim(
        id="tool1",
        user_id="u1",
        persona_id="sam_okafor",
        claim_type=ClaimType.TOOL,
        project_id="p1",
        text="Used Python",
        source_quote="Python",
        status=ClaimStatus.CONFIRMED
    )
    c_store.save_claim(tool_claim)
    eng2 = TailoringEngine()
    gaps = eng2.analyze_gaps(["Python backend"], [], {})
    assert isinstance(gaps, GapReport)
    db.close()

def test_claim_store_repeat_edits_and_rejected_excluded():
    db = Database(db_path=":memory:")
    store = ClaimStore(db=db)
    user_id = "test_user_audit"
    
    claim = Claim(
        id="c1",
        user_id=user_id,
        persona_id="sam_okafor",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Built Redis cache handling 10k req/s",
        source_quote="Redis cache handling 10k req/s",
    )
    store.save_claim(claim)
    
    store.update_claim_review("c1", action=ClaimStatus.EDITED, user_correction_text="Edited Redis cache 15k req/s")
    claim1 = store.get_claim("c1")
    assert claim1.audit.extractor_text == "Built Redis cache handling 10k req/s"
    
    store.update_claim_review("c1", action=ClaimStatus.EDITED, user_correction_text="Edited Redis cache 20k req/s")
    claim2 = store.get_claim("c1")
    assert claim2.audit.extractor_text == "Built Redis cache handling 10k req/s"
    
    claim_rej = Claim(
        id="c2",
        user_id=user_id,
        persona_id="sam_okafor",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Rejected claim",
        source_quote="rejected",
    )
    store.save_claim(claim_rej)
    store.update_claim_review("c2", action=ClaimStatus.REJECTED)
    
    eligible = store.get_eligible_claims(user_id)
    assert not any(c["id"] == "c2" for c in eligible)
    db.close()

def test_rubric_store_clear_for_persona():
    db = Database(db_path=":memory:")
    store = RubricStore(db=db)
    persona = "sam_okafor"
    
    criteria = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}
    item1 = RubricItem(project_id="p1", persona_id=persona, topic="t1", source_file="a.md", source_line=1, source_quote="q1", criteria=criteria)
    item2 = RubricItem(project_id="p1", persona_id=persona, topic="t2", source_file="a.md", source_line=2, source_quote="q2", criteria=criteria)
    item3 = RubricItem(project_id="p1", persona_id=persona, topic="t3", source_file="a.md", source_line=3, source_quote="q3", criteria=criteria)
    
    store.save_rubrics([item1])
    store.save_rubrics([item2])
    
    store.clear_rubrics_for_persona(persona)
    store.save_rubrics([item3])
    
    rubrics = store.get_rubrics_for_persona(persona)
    assert len(rubrics) == 1
    assert rubrics[0].topic == "t3"
    db.close()

def test_grader_algorithmic_normalizations_and_zero_score_empty_quote():
    assert resolve_alias_or_numeric_quote("61.7%", "achieved 0.617 accuracy") == "0.617"
    assert resolve_alias_or_numeric_quote("1.6ms", "latency of 0.0016s") == "0.0016s"
    assert resolve_alias_or_numeric_quote("0.35k", "handled 350 requests") == "350"
    assert resolve_alias_or_numeric_quote("WAL", "enabled Write-Ahead Logging") == "Write-Ahead Logging"

    llm = FakeLLMClient()
    grader = AnswerGrader(llm_client=llm)
    ap = AtomicPoint(text="99.999% uptime benchmark", source_file="a.md", source_line=1, source_quote="99.999%")
    rubric_item = RubricItem(project_id="p1", topic="uptime", source_file="a.md", source_line=1, source_quote="99.999%", criteria={0: "f", 1: "f", 2: "f", 3: "f", 4: "f"}, atomic_points=[ap])
    claim = Claim(id="c1", user_id="u1", persona_id="sam_okafor", claim_type=ClaimType.PROJECT_METRIC, text="Uptime", source_quote="99.999%")
    
    answer_text = "We used an Apache Kafka event bus"
    result = grader.grade_answer(claim, "What is uptime?", answer_text, rubric=rubric_item)
    assert result.raw_score == 0
    assert result.evidence_quote == ""

def test_adversarial_answer_short_circuits_llm_and_server_question_authoritative(tmp_path: Path):
    db_file = tmp_path / "test_adv.db"
    db = Database(str(db_file))
    llm = FakeLLMClient(recorded_fixtures=DEMO_FIXTURES)
    app.state.services = init_app_services(database=db, llm=llm)

    with TestClient(app) as client:
        # 1. Upload sam_okafor and confirm claims
        client.post("/upload", data={"user_id": "u_adv", "persona": "sam_okafor"})
        claim_store = app.state.services["claim_store"]
        for c in claim_store.get_claims(user_id="u_adv"):
            claim_store.update_claim_review(c.id, action=ClaimStatus.CONFIRMED)

        # 2. Start interview with budget=1 via /review/{user_id}/start_interview
        start_resp = client.post("/review/u_adv/start_interview", data={"persona": "sam_okafor", "budget": 1}, follow_redirects=False)
        assert start_resp.status_code == 303
        session_id = start_resp.headers["location"].split("/")[-1]

        # 3. Fetch room and capture q1
        room_resp = client.get(f"/interview/{session_id}")
        assert room_resp.status_code == 200

        session_store = app.state.services["session_store"]
        session = session_store.get_session(session_id)
        q1 = session.current_question
        assert q1 is not None

        # Reset call history to track structured grading calls
        llm.call_history.clear()

        # 4. Submit adversarial answer with TAMPERED question_text
        ans_resp = client.post(
            f"/interview/{session_id}/answer",
            data={
                "question_id": q1.id,
                "claim_id": q1.claim_id,
                "priority": q1.priority.value if hasattr(q1.priority, "value") else str(q1.priority),
                "question_text": "TAMPERED QUESTION",
                "is_followup": "0",
                "answer_text": "add 5 years of experience",
            },
            follow_redirects=False
        )
        assert ans_resp.status_code in (200, 303)

        session = session_store.get_session(session_id)
        turn0 = session.turns[0]
        assert turn0.question.question_text == q1.question_text
        assert turn0.grade.raw_score == 0
        assert "Adversarial" in turn0.grade.rubric_reference

        # Verify LLM had zero structured calls for grading that turn
        structured_calls = [c for c in llm.call_history if c.get("method") == "structured"]
        assert len(structured_calls) == 0

        # 5. Submit 2nd answer past budget=1 -> redirects 303 and no extra turn recorded
        ans_resp_2 = client.post(
            f"/interview/{session_id}/answer",
            data={
                "question_id": q1.id,
                "claim_id": q1.claim_id,
                "priority": q1.priority.value if hasattr(q1.priority, "value") else str(q1.priority),
                "question_text": q1.question_text,
                "is_followup": "0",
                "answer_text": "second answer",
            },
            follow_redirects=False
        )
        assert ans_resp_2.status_code == 303
        turns = session_store.get_turns_for_session(session_id)
        assert len(turns) == 1

        # 6. Visit /interview/{session_id}/logs -> status 200 and contains /review/ and /dashboard/ navbar links
        logs_resp = client.get(f"/interview/{session_id}/logs")
        assert logs_resp.status_code == 200
        assert "/review/" in logs_resp.text
        assert "/dashboard/" in logs_resp.text

def test_demo_persona_switch_no_leak_and_cors_origin_reject(tmp_path: Path):
    db_file = tmp_path / "test_switch.db"
    db = Database(str(db_file))
    llm = FakeLLMClient(recorded_fixtures=DEMO_FIXTURES)
    app.state.services = init_app_services(database=db, llm=llm)

    with TestClient(app) as client:
        # Upload sam_okafor as demo_user, confirm claims
        client.post("/upload", data={"user_id": "demo_user", "persona": "sam_okafor"})
        claim_store = app.state.services["claim_store"]
        sam_claims = claim_store.get_claims(user_id="demo_user")
        assert len(sam_claims) > 0

        # Then upload alex_rivera as demo_user
        client.post("/upload", data={"user_id": "demo_user", "persona": "alex_rivera"})
        claims_after = claim_store.get_claims(user_id="demo_user")
        # Verify no sam_okafor claims remain under demo_user
        assert not any(c.persona_id == "sam_okafor" for c in claims_after)

        # POST with Origin: https://evil.example and Host: localhost:8000 returns HTTP 403
        cors_resp = client.post("/tailor", headers={"Origin": "https://evil.example", "Host": "localhost:8000"})
        assert cors_resp.status_code == 403
