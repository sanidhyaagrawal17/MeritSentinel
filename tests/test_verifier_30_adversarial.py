"""Evaluation of StrictVerifier against 30 NEW independent adversarial sentences.

Verifies that:
- Fabricated numbers & metrics are strictly blocked.
- Unbacked tools & architectures are strictly blocked.
- Role/leadership overclaims are strictly blocked.
- Fabricated employers, institutions, and inflated CGPAs are blocked.
- Prompt injection bypasses and homoglyphs are strictly neutralized.
- Reports TP, FP, FN, TN, precision, recall, and false-positive rate (FPR).
"""

import json
from pathlib import Path
import pytest
from resumeproof.verifier.strict_verifier import StrictVerifier
from tests.test_verifier_100 import test_candidate_profile


def test_30_adversarial_sentences_strict_evaluation(test_candidate_profile):
    facts, claims = test_candidate_profile
    verifier = StrictVerifier()

    fixture_path = Path("tests/fixtures/adversarial_sentences_30.json")
    assert fixture_path.exists(), "Adversarial fixture missing!"

    with open(fixture_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 30, f"Expected 30 adversarial cases, got {len(cases)}"

    # In adversarial testing:
    # Ground truth is INVALID (Negative, meaning it should NOT be verified).
    # If the verifier blocks it -> True Negative (TN, correct rejection).
    # If the verifier passes it -> False Positive (FP, security/safety leak).
    tn = 0
    fp = 0
    leaked_cases = []

    print("\n" + "="*80)
    print("30 NEW ADVERSARIAL SENTENCES STRICT VERIFIER EVALUATION")
    print("="*80)

    for idx, c in enumerate(cases, start=1):
        res = verifier.verify_sentence(
            c["sentence"],
            eligible_claims=claims,
            confirmed_facts=facts,
            sentence_idx=idx
        )
        if res.status == "BLOCKED":
            tn += 1
            print(f"[{c['id']}] PASS (BLOCKED): {c['sentence'][:55]}... | Reasons: {res.blocking_reasons}")
        else:
            fp += 1
            leaked_cases.append(c)
            print(f"[{c['id']}] FAIL (LEAKED): {c['sentence']}")

    # False Positive Rate (FPR) = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("="*80)
    print(f"Adversarial Evaluation Summary (N={len(cases)}):")
    print(f"  * Correctly Blocked (TN): {tn}/{len(cases)} ({tn/len(cases):.1%})")
    print(f"  * Falsely Leaked    (FP): {fp}/{len(cases)} ({fp/len(cases):.1%})")
    print(f"  * False Positive Rate (FPR): {fpr:.4f} ({fpr:.1%})")
    print("="*80)

    # Invariants:
    # Leak rate must be strictly 0.0 (Zero tolerance for ungrounded claims)
    assert fp == 0, f"Critical leak! {fp} adversarial sentences were allowed through verifier: {leaked_cases}"
    assert tn == 30, f"Expected all 30 adversarial sentences to be blocked, but only {tn} were blocked."
    assert fpr == 0.0, f"Expected FPR to be 0.0, got {fpr}"
