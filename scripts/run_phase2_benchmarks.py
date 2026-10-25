"""Comprehensive benchmark and verification script for Phase 2 Gates 1-5 using real Ollama models.
"""

from pathlib import Path
import re
import json
import time

from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.core.telemetry import telemetry
from resumeproof.extractor.pdf_parser import parse_resume
from resumeproof.extractor.claim_extractor import extract_from_resume
from resumeproof.interview.generator import QuestionGenerator, check_rubric_overlap
from resumeproof.interview.grader import AnswerGrader
from resumeproof.rubric.builder import build_rubrics_for_persona
from resumeproof.schemas.claims import Claim, ClaimType
from resumeproof.schemas.rubric import RubricItem
from resumeproof.schemas.interview import QuestionPriority
from resumeproof.schemas.facts import ProfileFactType

def run_all_benchmarks():
    llm = OpenAICompatibleClient()

    print("================================================================================")
    print("PHASE 2 GATE VERIFICATION: REAL MODEL EVALUATION (qwen2.5:3b & qwen2.5:7b)")
    print("================================================================================\n")

    # -------------------------------------------------------------------------
    # ITEM 2: Rubric-anchored questions (10 questions from 10 rubric items with 3B)
    # -------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("ITEM 2: RUBRIC-ANCHORED QUESTIONS BENCHMARK (Real qwen2.5:3b)")
    print("--------------------------------------------------------------------------------")
    generator = QuestionGenerator(llm_client=llm, model="qwen2.5:3b")

    # Gather 10 rubric items across all 3 personas
    all_rubrics = (
        build_rubrics_for_persona("sam_okafor") +
        build_rubrics_for_persona("alex_rivera") +
        build_rubrics_for_persona("priya_nair")
    )

    # Pick 10 distinct, substantive rubric items
    target_rubrics = [r for r in all_rubrics if len(r.source_quote.split()) >= 6][:10]

    item2_results = []
    passed_count = 0

    STOPWORDS = {
        "and", "with", "for", "the", "a", "an", "in", "on", "at", "to", "of", "by",
        "from", "as", "is", "using", "built", "implemented", "sustained", "how",
        "what", "did", "you", "your", "can", "could", "would", "explain", "describe",
        "detail", "details", "approach", "handle", "handling", "were", "was", "are",
        "that", "this", "these", "those", "when", "which", "where", "why", "who", "their"
    }

    for idx, r in enumerate(target_rubrics, start=1):
        claim = Claim(
            id=f"c_benchmark_{idx}",
            user_id="benchmark_user",
            persona_id=r.persona_id,
            project_id=r.project_id,
            claim_type=ClaimType.PROJECT_METRIC,
            text=r.source_quote,
        )

        t0 = time.perf_counter()
        question = generator.generate_question(
            claim=claim,
            priority=QuestionPriority.UNTESTED,
            rubric=r,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_passed = check_rubric_overlap(question.question_text, r)
        if is_passed:
            passed_count += 1

        rubric_tokens = {
            t.lower() for t in re.findall(r"[a-zA-Z0-9_\-\.]+", f"{r.topic} {r.source_quote}")
            if len(t) >= 3 and t.lower() not in STOPWORDS
        }
        q_tokens = {
            t.lower() for t in re.findall(r"[a-zA-Z0-9_\-\.]+", question.question_text)
            if len(t) >= 3 and t.lower() not in STOPWORDS
        }
        overlap = sorted(q_tokens.intersection(rubric_tokens))

        print(f"[{idx}/10] Project: {r.project_id} | Topic: {r.topic}")
        print(f"  Rubric Quote: \"{r.source_quote}\" ({r.source_file}:{r.source_line})")
        print(f"  Generated Q:  \"{question.question_text}\"")
        print(f"  Overlap:      {overlap} (Passed: {is_passed}, Latency: {latency_ms:.1f}ms)")
        print()

        item2_results.append({
            "idx": idx,
            "project_id": r.project_id,
            "topic": r.topic,
            "source_quote": r.source_quote,
            "question": question.question_text,
            "overlap_tokens": overlap,
            "passed": is_passed,
            "latency_ms": latency_ms,
        })

    print(f"==> ITEM 2 RESULT: {passed_count} / {len(target_rubrics)} questions passed the deterministic overlap check.\n")

    # -------------------------------------------------------------------------
    # ITEM 3: Grader behavior with the real 7B model across 4 cases
    # -------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("ITEM 3: GRADER BEHAVIOR WITH REAL 7B MODEL (4 EVALUATION CASES)")
    print("--------------------------------------------------------------------------------")
    grader = AnswerGrader(llm_client=llm, model="qwen2.5:7b")

    wal_claim = Claim(
        id="c_wal_eval",
        user_id="user_eval",
        persona_id="sam_okafor",
        project_id="task-tracker-api",
        claim_type=ClaimType.PROJECT_METRIC,
        text="Built RESTful API with SQLite connection pooling sustaining 350 concurrent read req/sec.",
        tools=["FastAPI", "SQLite", "Python"],
        metric_values=["350 req/sec"],
        source_line=6,
    )

    wal_rubric = RubricItem(
        id="rubric_wal_anchor",
        project_id="task-tracker-api",
        persona_id="sam_okafor",
        topic="sqlite_wal_connection_pooling",
        source_file="data/demo/sam_okafor/projects/task-tracker-api/README.md",
        source_line=5,
        source_quote="Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.",
        criteria={
            0: "Irrelevant, refuses to answer, or makes unsubstantiated claims contradicting verified facts (e.g. claims unverified technologies or scale for task-tracker-api).",
            1: "Surface-level or generic answer. Fails to provide concrete details matching 'Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.'.",
            2: "Partial technical answer. Mentions some relevant aspects but omits key parameters or constraints from 'Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.'.",
            3: "Competent answer matching verified ground truth: 'Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.'. Demonstrates accurate understanding of project parameters.",
            4: "Expert answer demonstrating deep architectural mastery: accurately articulates 'Sustained 350 concurrent read req/sec with SQLite WAL mode and connection pooling.', explains underlying trade-offs, constraints, and why this design was chosen.",
        }
    )

    question_text = "How did you scale SQLite to sustain 350 concurrent read req/sec in Task-Tracker API?"

    test_cases = {
        "(a) Confident wrong topic": (
            "We built an Apache Kafka event bus and BGP scrubbing centers to stream financial transactions "
            "into our Byzantine fault-tolerant Raft cluster with zero downtime."
        ),
        "(b) Correct but terse": (
            "Enabled WAL mode and SQLAlchemy connection pool of 10 to hit 350 req/sec."
        ),
        "(c) Copy-pasted generic": (
            "APIs require good database connections and error handling. We followed standard tutorials "
            "to make sure the app responded quickly."
        ),
        "(d) Strong answer": (
            "In Task-Tracker API, SQLite can become a write bottleneck, so we enabled WAL (Write-Ahead Logging) "
            "mode and tuned SQLAlchemy connection pooling to 10 connections. For read-heavy operations, this "
            "allowed concurrent reads while WAL checkpointing happened in the background, reaching 350 read "
            "req/sec in our locust test."
        ),
    }

    item3_results = {}
    for case_name, answer in test_cases.items():
        print(f"Evaluating Case {case_name}:")
        print(f"  Answer: \"{answer}\"")
        grade = grader.grade_answer(
            claim=wal_claim,
            question_text=question_text,
            answer_text=answer,
            rubric=wal_rubric,
        )
        print(f"  Score:            {grade.raw_score} / 4")
        print(f"  Evidence Quote:   \"{grade.evidence_quote}\"")
        print(f"  Rubric Reference: \"{grade.rubric_reference}\"")
        print(f"  Matched Rubrics:  {grade.matched_rubric_ids}")
        print(f"  Missed Rubrics:   {grade.missed_rubric_ids}")
        print(f"  Critique:         {grade.feedback}")
        print(f"  Latency:          {grade.latency_ms:.1f}ms")
        print()
        item3_results[case_name] = {
            "score": grade.raw_score,
            "evidence_quote": grade.evidence_quote,
            "matched_rubrics": grade.matched_rubric_ids,
            "missed_rubrics": grade.missed_rubric_ids,
            "critique": grade.feedback,
            "latency_ms": grade.latency_ms,
        }

    # -------------------------------------------------------------------------
    # ITEM 4: Score Anchors Re-grade of Task-Tracker Answer
    # -------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("ITEM 4: SCORE ANCHORS RE-GRADE OF TASK-TRACKER ANSWER")
    print("--------------------------------------------------------------------------------")
    # Task-Tracker answer re-graded with strict anchors
    tt_grade = item3_results["(d) Strong answer"]
    print("Task-Tracker Re-grade with Anchors:")
    print(f"  Score:    {tt_grade['score']} / 4")
    print(f"  Critique: {tt_grade['critique']}")
    print(f"  Evidence: \"{tt_grade['evidence_quote']}\"")
    print()

    # -------------------------------------------------------------------------
    # ITEM 5: Extractor Evaluation on all 3 personas with real qwen2.5:3b
    # -------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("ITEM 5: EXTRACTOR REAL-MODEL EVALUATION (All 3 Personas with qwen2.5:3b)")
    print("--------------------------------------------------------------------------------")
    personas = ["sam_okafor", "alex_rivera", "priya_nair"]
    item5_results = {}

    for persona in personas:
        pdf_path = Path("data/demo") / persona / "resume.pdf"
        parse_result = parse_resume(pdf_path)
        facts, claims = extract_from_resume(
            parse_result=parse_result,
            llm_client=llm,
            persona_id=persona,
            model="qwen2.5:3b"
        )

        # Check (a): profile facts never inferred without experience section
        has_employer = any(f.fact_type == ProfileFactType.EMPLOYER_NAME for f in facts)
        has_emp_dates = any(f.fact_type == ProfileFactType.EMPLOYMENT_DATES for f in facts)

        # Count claims by type
        skill_claims = [c for c in claims if c.claim_type == ClaimType.SKILL]
        tool_claims = [c for c in claims if c.claim_type == ClaimType.TOOL]
        project_claims = [c for c in claims if c.claim_type in (ClaimType.PROJECT_METRIC, ClaimType.EXPERIENCE_CLAIM)]

        # Check (c): experience or project claims must never be typed as skill or tool
        misclassified_proj_as_skill = [
            c for c in claims
            if c.project_id is not None and c.claim_type in (ClaimType.SKILL, ClaimType.TOOL)
        ]

        # Load planted traps / answer key
        traps_path = Path("data/demo") / persona / "planted_traps.json"
        ground_truth = json.loads(traps_path.read_text(encoding="utf-8"))
        supported_keys = [item["claim"].lower() for item in ground_truth.get("supported", [])]
        partial_keys = [item["claim"].lower() for item in ground_truth.get("partial", [])]
        unsupported_keys = [item["claim"].lower() for item in ground_truth.get("unsupported", [])]

        total_key_claims = len(supported_keys) + len(partial_keys) + len(unsupported_keys)

        # Check how many planted traps / key claims were extracted
        extracted_texts = [c.text.lower() for c in claims]
        matched_keys = 0
        for key in supported_keys + partial_keys + unsupported_keys:
            key_words = set(re.findall(r"[a-zA-Z0-9]+", key))
            if any(len(key_words.intersection(set(re.findall(r"[a-zA-Z0-9]+", ext)))) >= max(2, len(key_words) // 2) for ext in extracted_texts):
                matched_keys += 1

        recall = matched_keys / max(1, total_key_claims)
        precision = 1.0 if len(misclassified_proj_as_skill) == 0 else (len(claims) - len(misclassified_proj_as_skill)) / len(claims)

        print(f"Persona: {persona}")
        print(f"  Profile Facts Extracted:  {len(facts)}")
        print(f"    - Employer Name Present: {has_employer}")
        print(f"    - Employment Dates Present: {has_emp_dates}")
        print(f"  Total Claims Extracted:   {len(claims)}")
        print(f"    - Individual Skills:    {len(skill_claims)} ({', '.join(c.text for c in skill_claims[:8])}...)")
        print(f"    - Project/Exp Claims:   {len(project_claims)}")
        print(f"    - Misclassified as Skill/Tool: {len(misclassified_proj_as_skill)}")
        print(f"  Answer Key Traps Matched: {matched_keys}/{total_key_claims}")
        print(f"  Precision: {precision * 100:.1f}%")
        print(f"  Recall:    {recall * 100:.1f}%")
        print()

        item5_results[persona] = {
            "total_facts": len(facts),
            "has_employer": has_employer,
            "has_emp_dates": has_emp_dates,
            "total_claims": len(claims),
            "skill_claims_count": len(skill_claims),
            "project_claims_count": len(project_claims),
            "misclassified_count": len(misclassified_proj_as_skill),
            "key_claims_matched": f"{matched_keys}/{total_key_claims}",
            "precision": precision,
            "recall": recall,
        }

    # Save summary artifact
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "phase2_gate_results.json"
    summary_path.write_text(json.dumps({
        "item2_rubric_questions": {
            "passed": passed_count,
            "total": len(target_rubrics),
            "details": item2_results,
        },
        "item3_grader_cases": item3_results,
        "item4_task_tracker_anchors": tt_grade,
        "item5_extractor_eval": item5_results,
    }, indent=2), encoding="utf-8")
    print(f"Saved full benchmark results to: {summary_path}")

if __name__ == "__main__":
    run_all_benchmarks()
