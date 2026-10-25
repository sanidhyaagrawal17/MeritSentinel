"""Test for Requirement 2: Rubric-anchored questions using real qwen2.5:3b model.

Generates 10 questions from 10 rubric items, runs deterministic overlap check,
and reports the exact number that pass.
"""

import pytest
import re
from resumeproof.core.llm import OpenAICompatibleClient
from resumeproof.interview.generator import (
    QuestionGenerator,
    check_rubric_overlap,
    check_question_outside_points,
)
from resumeproof.rubric.builder import build_rubrics_for_persona
from resumeproof.schemas.claims import Claim, ClaimType
from resumeproof.schemas.interview import QuestionPriority

@pytest.mark.real_model
def test_ten_rubric_anchored_questions_real_3b():
    """Requirement 2: Generate 10 questions from 10 rubric items with real 3B model and report pass count."""
    llm = OpenAICompatibleClient()
    generator = QuestionGenerator(llm_client=llm, model="qwen2.5:3b")

    all_rubrics = (
        build_rubrics_for_persona("sam_okafor") +
        build_rubrics_for_persona("alex_rivera") +
        build_rubrics_for_persona("priya_nair")
    )

    # Select 10 diverse rubric items with substantive source quotes
    selected_rubrics = [r for r in all_rubrics if len(r.source_quote.split()) >= 6][:10]
    assert len(selected_rubrics) == 10, f"Expected 10 rubric items, found {len(selected_rubrics)}"

    passed = 0
    outside_count = 0
    results = []

    for idx, r in enumerate(selected_rubrics, start=1):
        claim = Claim(
            id=f"c_test_{idx}",
            user_id="test_user",
            persona_id=r.persona_id,
            project_id=r.project_id,
            claim_type=ClaimType.PROJECT_METRIC,
            text=r.source_quote,
        )

        question = generator.generate_question(
            claim=claim,
            priority=QuestionPriority.UNTESTED,
            rubric=r,
        )

        is_overlap = check_rubric_overlap(question.question_text, r)
        is_outside = check_question_outside_points(question.question_text, r)
        if is_overlap:
            passed += 1
        if is_outside:
            outside_count += 1

        results.append({
            "idx": idx,
            "project_id": r.project_id,
            "topic": r.topic,
            "quote": r.source_quote,
            "question": question.question_text,
            "passed": is_overlap,
            "is_outside": is_outside,
        })

    print(f"\n[REAL 3B MODEL OVERLAP REPORT]: {passed} / 10 questions passed the deterministic overlap check.")
    print(f"[OUTSIDE POINTS REPORT]: {outside_count} / 10 questions asked for content outside their atomic points.")
    for res in results:
        status = "PASSED" if res["passed"] else "FAILED"
        outside_flag = " [OUTSIDE!]" if res["is_outside"] else ""
        print(f"  [{res['idx']}/10] {status}{outside_flag} | Proj: {res['project_id']} | Q: {res['question']}")

    # Requirement 2 & 3: High overlap pass count and zero questions asking outside atomic points
    assert passed >= 8, f"Expected at least 8/10 to pass overlap check, but got {passed}/10"
    assert outside_count == 0, f"Expected 0 questions asking for content outside atomic points, but got {outside_count}"

