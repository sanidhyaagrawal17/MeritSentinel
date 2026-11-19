"""Quadratic-Weighted Cohen's Kappa (QWK) evaluation script for Phase 5 human-in-the-loop validation.

This script implements:
1. Unweighted Cohen's Kappa
2. Linear-Weighted Cohen's Kappa
3. Quadratic-Weighted Cohen's Kappa (QWK, standard for ordinal grading 0-4)

By default, this script runs strictly on a synthetic test fixture to prove mathematical correctness.
It will NOT claim real human evaluation until human labels are explicitly supplied via --human-labels.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


def compute_cohens_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    num_categories: int = 5,
    weighting: str = "quadratic",
) -> float:
    """Computes Cohen's Kappa with optional linear or quadratic weighting.
    
    Args:
        rater_a: Sequence of integer scores from rater A (e.g. Model, 0-4).
        rater_b: Sequence of integer scores from rater B (e.g. Human, 0-4).
        num_categories: Number of ordinal categories (k = 5 for 0, 1, 2, 3, 4).
        weighting: 'unweighted', 'linear', or 'quadratic'.
        
    Returns:
        Kappa coefficient in [-1.0, 1.0].
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"Length mismatch: rater_a has {len(rater_a)} items, rater_b has {len(rater_b)}.")
    n = len(rater_a)
    if n == 0:
        return 0.0

    # Build observed confusion matrix O[i][j]
    o_matrix = [[0 for _ in range(num_categories)] for _ in range(num_categories)]
    for a, b in zip(rater_a, rater_b):
        if not (0 <= a < num_categories and 0 <= b < num_categories):
            raise ValueError(f"Scores must be within [0, {num_categories - 1}], got ({a}, {b})")
        o_matrix[a][b] += 1

    # Marginals
    hist_a = [sum(o_matrix[i][j] for j in range(num_categories)) for i in range(num_categories)]
    hist_b = [sum(o_matrix[i][j] for i in range(num_categories)) for j in range(num_categories)]

    # Expected matrix E[i][j] under independence
    e_matrix = [
        [(hist_a[i] * hist_b[j]) / float(n) for j in range(num_categories)]
        for i in range(num_categories)
    ]

    # Distance/weight matrix W[i][j]
    w_matrix = [[0.0 for _ in range(num_categories)] for _ in range(num_categories)]
    for i in range(num_categories):
        for j in range(num_categories):
            diff = abs(i - j)
            max_diff = num_categories - 1
            if weighting == "unweighted":
                w_matrix[i][j] = 0.0 if i == j else 1.0
            elif weighting == "linear":
                w_matrix[i][j] = diff / float(max_diff)
            elif weighting == "quadratic":
                w_matrix[i][j] = (diff ** 2) / float(max_diff ** 2)
            else:
                raise ValueError(f"Unknown weighting scheme: {weighting}")

    # Numerator: sum(W * O), Denominator: sum(W * E)
    numerator = sum(w_matrix[i][j] * o_matrix[i][j] for i in range(num_categories) for j in range(num_categories))
    denominator = sum(w_matrix[i][j] * e_matrix[i][j] for i in range(num_categories) for j in range(num_categories))

    if denominator == 0.0:
        return 1.0  # Perfect agreement

    kappa = 1.0 - (numerator / denominator)
    return round(kappa, 4)


# Canonical 20-sample synthetic test fixture for verification of formula
SYNTHETIC_MODEL_SCORES = [4, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 4, 3, 2, 1, 0, 3, 2, 1]
SYNTHETIC_HUMAN_SCORES = [4, 3, 3, 2, 1, 0, 4, 4, 2, 0, 0, 4, 4, 2, 2, 1, 0, 3, 2, 1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quadratic-Weighted Cohen's Kappa for Phase 5")
    parser.add_argument("--human-labels", type=str, default=None, help="Path to JSON file containing real human evaluations")
    parser.add_argument("--categories", type=int, default=5, help="Number of score categories (default 5 for 0..4)")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 5 INTER-RATER RELIABILITY: COHEN'S KAPPA EVALUATOR")
    print("=" * 80)

    if args.human_labels is None:
        print("[NOTICE] No --human-labels provided.")
        print("[NOTICE] Running on SYNTHETIC validation benchmark to demonstrate mathematical correctness.")
        print("[NOTICE] Real human evaluation is strictly gated pending user-provided human annotations.")
        print("-" * 80)
        
        m_scores = SYNTHETIC_MODEL_SCORES
        h_scores = SYNTHETIC_HUMAN_SCORES
        source_name = f"Synthetic Benchmark (N={len(m_scores)})"
    else:
        label_file = Path(args.human_labels)
        if not label_file.exists():
            print(f"[ERROR] Specified human label file '{label_file}' does not exist.")
            sys.exit(1)
        data = json.loads(label_file.read_text(encoding="utf-8"))
        m_scores = [d["model_score"] for d in data]
        h_scores = [d["human_score"] for d in data]
        source_name = f"Human Annotation File: {label_file.name} (N={len(m_scores)})"

    k_unweighted = compute_cohens_kappa(m_scores, h_scores, num_categories=args.categories, weighting="unweighted")
    k_linear = compute_cohens_kappa(m_scores, h_scores, num_categories=args.categories, weighting="linear")
    k_quadratic = compute_cohens_kappa(m_scores, h_scores, num_categories=args.categories, weighting="quadratic")

    # Exact agreement percentage
    exact_agree = sum(1 for a, b in zip(m_scores, h_scores) if a == b) / float(len(m_scores))
    adjacent_agree = sum(1 for a, b in zip(m_scores, h_scores) if abs(a - b) <= 1) / float(len(m_scores))

    print(f"Data Source:               {source_name}")
    print(f"Total Evaluations:         {len(m_scores)}")
    print(f"Exact Agreement:           {exact_agree * 100:.2f}% ({sum(1 for a, b in zip(m_scores, h_scores) if a == b)}/{len(m_scores)})")
    print(f"Adjacent Agreement (±1):   {adjacent_agree * 100:.2f}% ({sum(1 for a, b in zip(m_scores, h_scores) if abs(a - b) <= 1)}/{len(m_scores)})")
    print(f"Unweighted Cohen's Kappa:  {k_unweighted:.4f}")
    print(f"Linear-Weighted Kappa:     {k_linear:.4f}")
    print(f"Quadratic-Weighted Kappa:  {k_quadratic:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
