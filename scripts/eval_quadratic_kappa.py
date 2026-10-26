"""Quadratic-Weighted Cohen's Kappa evaluation script.

Implements quadratic-weighted Cohen's kappa (kappa_w) for ordinal rating scales (e.g. 0 to 4).
Includes verification against synthetic test cases:
1. Perfect agreement -> kappa_w = 1.0
2. Chance agreement -> kappa_w ~ 0.0
3. Inverse / opposite agreement -> kappa_w < 0.0

NOTE: Phase 5 is NOT approved. Do NOT run this script on human labels until explicitly instructed.
"""

from typing import Sequence
import numpy as np


def quadratic_weighted_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    min_rating: int = 0,
    max_rating: int = 4,
) -> float:
    """Computes quadratic-weighted Cohen's kappa on two rater sequences."""
    assert len(rater_a) == len(rater_b), "Rater sequences must be equal length"
    n = len(rater_a)
    if n == 0:
        return 1.0

    k = max_rating - min_rating + 1
    # Build confusion matrix O
    O = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(rater_a, rater_b):
        ai = int(np.clip(a - min_rating, 0, k - 1))
        bi = int(np.clip(b - min_rating, 0, k - 1))
        O[ai, bi] += 1.0

    # Normalize observed matrix
    O /= n

    # Expected marginals
    hist_a = O.sum(axis=1)
    hist_b = O.sum(axis=0)
    E = np.outer(hist_a, hist_b)

    # Quadratic weight matrix: w_ij = (i - j)^2 / (k - 1)^2
    denom_weight = float((k - 1) ** 2) if k > 1 else 1.0
    W = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            W[i, j] = ((i - j) ** 2) / denom_weight

    observed_disagreement = np.sum(W * O)
    expected_disagreement = np.sum(W * E)

    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0

    kappa = 1.0 - (observed_disagreement / expected_disagreement)
    return float(kappa)


def run_synthetic_tests():
    print("=" * 70)
    print("TESTING QUADRATIC-WEIGHTED KAPPA ON SYNTHETIC BENCHMARKS")
    print("=" * 70)

    # 1. Perfect agreement
    labels_a = [0, 1, 2, 3, 4, 2, 3, 1, 0, 4] * 10
    labels_b = list(labels_a)
    k_perf = quadratic_weighted_kappa(labels_a, labels_b, 0, 4)
    print(f"1. Perfect Agreement Test:  kappa = {k_perf:.4f} (Expected: 1.0000)")
    assert abs(k_perf - 1.0) < 1e-6, f"Expected 1.0, got {k_perf}"

    # 2. Chance agreement (uncorrelated random ratings)
    np.random.seed(42)
    chance_a = np.random.randint(0, 5, size=5000)
    chance_b = np.random.randint(0, 5, size=5000)
    k_chance = quadratic_weighted_kappa(chance_a, chance_b, 0, 4)
    print(f"2. Chance Agreement Test:   kappa = {k_chance:.4f} (Expected: ~0.0000)")
    assert abs(k_chance) < 0.05, f"Expected near 0.0, got {k_chance}"

    # 3. Inverse / opposite agreement (maximal disagreement)
    inv_a = [0, 1, 2, 3, 4] * 20
    inv_b = [4, 3, 2, 1, 0] * 20
    k_inv = quadratic_weighted_kappa(inv_a, inv_b, 0, 4)
    print(f"3. Opposite Agreement Test: kappa = {k_inv:.4f} (Expected: < 0.0000)")
    assert k_inv < -0.5, f"Expected negative, got {k_inv}"

    print("=" * 70)
    print("All synthetic tests PASSED cleanly.")
    print("GATE NOTICE: Phase 5 is NOT approved. Human label evaluation is deferred.")
    print("=" * 70)


if __name__ == "__main__":
    run_synthetic_tests()
