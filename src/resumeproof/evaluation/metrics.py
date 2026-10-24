"""Evaluation metrics: Quadratic-weighted Cohen's kappa, exact agreement rate, and MAE."""

from typing import Sequence

def quadratic_weighted_kappa(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    min_rating: int = 0,
    max_rating: int = 4
) -> float:
    """Calculates Quadratic-Weighted Cohen's Kappa for ordinal scores.

    Formula:
        w_ij = (i - j)^2 / (max_rating - min_rating)^2
        kappa = 1 - (sum(w_ij * O_ij) / sum(w_ij * E_ij))
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater sequences must have identical lengths.")
    n = len(rater_a)
    if n == 0:
        return 1.0

    k = max_rating - min_rating + 1
    # Build confusion matrix O
    observed = [[0.0] * k for _ in range(k)]
    for a, b in zip(rater_a, rater_b):
        idx_a = max(0, min(k - 1, a - min_rating))
        idx_b = max(0, min(k - 1, b - min_rating))
        observed[idx_a][idx_b] += 1.0

    # Marginals
    sum_a = [sum(observed[i][j] for j in range(k)) for i in range(k)]
    sum_b = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    # Expected matrix E
    expected = [[(sum_a[i] * sum_b[j]) / n for j in range(k)] for i in range(k)]

    # Weight matrix (distance squared)
    denominator_norm = float((k - 1) ** 2) if k > 1 else 1.0

    observed_disagreement = 0.0
    expected_disagreement = 0.0

    for i in range(k):
        for j in range(k):
            weight = ((i - j) ** 2) / denominator_norm
            observed_disagreement += weight * observed[i][j]
            expected_disagreement += weight * expected[i][j]

    if expected_disagreement == 0.0:
        return 1.0

    kappa = 1.0 - (observed_disagreement / expected_disagreement)
    return round(kappa, 4)

def exact_agreement_rate(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Calculates the proportion of exact identical ratings."""
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater sequences must have identical lengths.")
    if not rater_a:
        return 1.0
    matches = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    return round(matches / len(rater_a), 4)

def mean_absolute_error(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Calculates Mean Absolute Error between two sets of ratings."""
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater sequences must have identical lengths.")
    if not rater_a:
        return 0.0
    mae = sum(abs(a - b) for a, b in zip(rater_a, rater_b)) / len(rater_a)
    return round(mae, 4)

def evaluate_ratings(rater_a: Sequence[int], rater_b: Sequence[int]) -> dict[str, float]:
    """Returns complete evaluation dictionary containing quadratic kappa, exact agreement, and MAE."""
    return {
        "quadratic_weighted_kappa": quadratic_weighted_kappa(rater_a, rater_b),
        "exact_agreement_rate": exact_agreement_rate(rater_a, rater_b),
        "mean_absolute_error": mean_absolute_error(rater_a, rater_b),
        "sample_size": len(rater_a),
    }
