"""Test suite evaluating StrictVerifier on 30 NEW Supported and 30 NEW Adversarial sentences."""

import json
from pathlib import Path
import pytest

from tests.test_verifier_100 import test_candidate_profile
from resumeproof.schemas.claims import Claim, ClaimType, ClaimStatus
from resumeproof.verifier.strict_verifier import StrictVerifier


@pytest.fixture
def full_candidate_profile(test_candidate_profile):
    facts, claims = test_candidate_profile
    extra_claims = [
        # Technical skills (24 skills)
        Claim(id="c_sk_py", user_id="u1", claim_type=ClaimType.SKILL, text="Python", tools=["Python"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_sql", user_id="u1", claim_type=ClaimType.SKILL, text="SQL", tools=["SQL"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_js", user_id="u1", claim_type=ClaimType.SKILL, text="JavaScript", tools=["JavaScript"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_ts", user_id="u1", claim_type=ClaimType.SKILL, text="TypeScript", tools=["TypeScript"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_mcp", user_id="u1", claim_type=ClaimType.SKILL, text="Model Context Protocol (MCP)", tools=["Model Context Protocol", "MCP"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_chroma", user_id="u1", claim_type=ClaimType.SKILL, text="ChromaDB", tools=["ChromaDB"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_embed", user_id="u1", claim_type=ClaimType.SKILL, text="Vector Embeddings", tools=["Vector Embeddings"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_sem_search", user_id="u1", claim_type=ClaimType.SKILL, text="Semantic Search", tools=["Semantic Search"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_rag", user_id="u1", claim_type=ClaimType.SKILL, text="Retrieval-Augmented Generation (RAG)", tools=["RAG", "Retrieval-Augmented Generation"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_tools", user_id="u1", claim_type=ClaimType.SKILL, text="Tool and Function Calling", tools=["Tool and Function Calling"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_guard", user_id="u1", claim_type=ClaimType.SKILL, text="Output Guardrails", tools=["Output Guardrails"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_fastapi", user_id="u1", claim_type=ClaimType.SKILL, text="FastAPI", tools=["FastAPI"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_rest", user_id="u1", claim_type=ClaimType.SKILL, text="RESTful APIs", tools=["RESTful APIs", "REST"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_pydantic", user_id="u1", claim_type=ClaimType.SKILL, text="Pydantic", tools=["Pydantic"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_sqlite", user_id="u1", claim_type=ClaimType.SKILL, text="SQLite", tools=["SQLite"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_nextjs", user_id="u1", claim_type=ClaimType.SKILL, text="Next.js", tools=["Next.js", "Next.js 14"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_react", user_id="u1", claim_type=ClaimType.SKILL, text="React", tools=["React"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_git", user_id="u1", claim_type=ClaimType.SKILL, text="Git", tools=["Git"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_github", user_id="u1", claim_type=ClaimType.SKILL, text="GitHub", tools=["GitHub"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_pytest", user_id="u1", claim_type=ClaimType.SKILL, text="pytest", tools=["pytest"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_postman", user_id="u1", claim_type=ClaimType.SKILL, text="Postman", tools=["Postman"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_docker", user_id="u1", claim_type=ClaimType.SKILL, text="Docker", tools=["Docker"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_ec2", user_id="u1", claim_type=ClaimType.SKILL, text="AWS EC2", tools=["AWS EC2"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_sk_gemini", user_id="u1", claim_type=ClaimType.SKILL, text="Google Antigravity (Gemini)", tools=["Google Antigravity (Gemini)", "Google Antigravity", "Gemini"], status=ClaimStatus.CONFIRMED),
        # Competitions and achievements
        Claim(id="c_ach_1", user_id="u1", claim_type=ClaimType.ACHIEVEMENT, text="Winner of the Medi-Caps University Internal Hackathon 2025 across 120 participating teams.", tools=[], metric_values=["120", "2025"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_ach_2", user_id="u1", claim_type=ClaimType.ACHIEVEMENT, text="Top 50 Finalist at the college level in Smart India Hackathon 2025 out of 300+ submissions.", tools=[], metric_values=["50", "2025", "300"], status=ClaimStatus.CONFIRMED),
        Claim(id="c_ach_3", user_id="u1", claim_type=ClaimType.ACHIEVEMENT, text="Participant in TATA AIG InnoVent 2026 national competition.", tools=[], metric_values=["2026"], status=ClaimStatus.CONFIRMED),
        # Experience claims
        Claim(id="c_exp_2", user_id="u1", claim_type=ClaimType.EXPERIENCE_CLAIM, text="Instructed secondary school students on foundational digital literacy, ethical AI awareness, and cyber hygiene.", tools=["Digital Literacy"], metric_values=[], status=ClaimStatus.CONFIRMED),
        Claim(id="c_proj_rag", user_id="u1", project_id="paperpulse-ai", claim_type=ClaimType.PROJECT_METRIC, text="Experienced in implementing vector retrieval with ChromaDB and deterministic token-overlap output guardrails.", tools=["ChromaDB", "Output Guardrails"], metric_values=[], status=ClaimStatus.CONFIRMED),
        Claim(id="c_proj_backend", user_id="u1", claim_type=ClaimType.EXPERIENCE_CLAIM, text="Built Python backend services using FastAPI and SQLite.", tools=["Python", "FastAPI", "SQLite"], metric_values=[], status=ClaimStatus.CONFIRMED),
        Claim(id="c_proj_meraki_v1", user_id="u1", project_id="netpass-gateway", claim_type=ClaimType.EXPERIENCE_CLAIM, text="Engineered a captive portal authentication engine integrating Cisco Meraki Dashboard API v1 with FastAPI, automating guest client provisioning.", tools=["Cisco Meraki API", "FastAPI", "Python"], metric_values=[], status=ClaimStatus.CONFIRMED),
        Claim(id="c_proj_meraki_jwt", user_id="u1", project_id="netpass-gateway", claim_type=ClaimType.EXPERIENCE_CLAIM, text="Secured administrative endpoints with JSON Web Tokens (JWT) and persistent SQLite session storage.", tools=["JWT", "SQLite"], metric_values=[], status=ClaimStatus.CONFIRMED),
        Claim(id="c_proj_pytest_mcp", user_id="u1", project_id="gamedata-mcp", claim_type=ClaimType.PROJECT_METRIC, text="Configured automated testing with pytest to validate SQLite queries and MCP stdio handlers.", tools=["pytest", "SQLite", "Model Context Protocol", "MCP"], metric_values=[], status=ClaimStatus.CONFIRMED),
    ]
    all_claims = list(claims) + extra_claims
    return facts, all_claims


def test_30_new_supported_sentences_evaluation(full_candidate_profile):
    """Evaluates 30 NEW supported sentences on the strict verifier."""
    facts, claims = full_candidate_profile
    verifier = StrictVerifier()

    fixture_path = Path("tests/fixtures/eval_sentences_60_new.json")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    supported_sentences = data["supported_sentences_30"]

    tp = 0  # correctly verified supported claim
    fn = 0  # supported claim falsely blocked
    errors = []

    for idx, sentence in enumerate(supported_sentences, start=1):
        res = verifier.verify_sentence(
            sentence=sentence,
            eligible_claims=claims,
            confirmed_facts=facts,
            sentence_idx=idx,
        )
        if res.status == "VERIFIED":
            tp += 1
        else:
            fn += 1
            errors.append({
                "sentence_idx": idx,
                "sentence": sentence,
                "blocking_reasons": res.blocking_reasons,
            })

    total = len(supported_sentences)
    recall = tp / total if total > 0 else 0.0

    print(f"\n[30 NEW Supported Sentences] Total: {total}, Verified (TP): {tp}, Blocked (FN): {fn}, Recall: {recall:.4f}")
    if errors:
        print("  Errors (False Negatives):")
        for err in errors:
            print(f"    - Sentence {err['sentence_idx']}: {err['sentence']}")
            print(f"      Reasons: {err['blocking_reasons']}")

    assert fn == 0, f"{fn} valid sentences were blocked: {errors}"
    assert tp == total, f"Expected {total} verified sentences, got {tp}"


def test_30_new_adversarial_sentences_evaluation(full_candidate_profile):
    """Evaluates 30 NEW adversarial sentences on the strict verifier."""
    facts, claims = full_candidate_profile
    verifier = StrictVerifier()

    fixture_path = Path("tests/fixtures/eval_sentences_60_new.json")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    adversarial_sentences = data["adversarial_sentences_30"]

    tn = 0  # correctly blocked adversarial sentence
    fp = 0  # adversarial sentence falsely passed
    errors = []

    for idx, sentence in enumerate(adversarial_sentences, start=1):
        res = verifier.verify_sentence(
            sentence=sentence,
            eligible_claims=claims,
            confirmed_facts=facts,
            sentence_idx=idx,
        )
        if res.status != "VERIFIED":
            tn += 1
        else:
            fp += 1
            errors.append({
                "sentence_idx": idx,
                "sentence": sentence,
                "matching_claim_id": res.supporting_claim_id,
            })

    total = len(adversarial_sentences)
    fpr = fp / total if total > 0 else 0.0

    print(f"\n[30 NEW Adversarial Sentences] Total: {total}, Blocked (TN): {tn}, Leaked (FP): {fp}, FPR: {fpr:.4f}")
    if errors:
        print("  CRITICAL LEAKAGE DETECTED (False Positives):")
        for err in errors:
            safe_s = err['sentence'].encode('ascii', 'replace').decode('ascii')
            print(f"    - Sentence {err['sentence_idx']}: {safe_s}")
            print(f"      Falsely matched: {err['matching_claim_id']}")

    assert fp == 0, f"CRITICAL LEAKAGE DETECTED! {fp} adversarial sentences passed verifier: {errors}"
    assert tn == total, f"Expected {total} blocked sentences, got {tn}"
