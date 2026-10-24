"""Tests for quadratic-weighted Cohen's kappa, exact agreement, and MAE."""

import pytest
from resumeproof.evaluation.metrics import (
    quadratic_weighted_kappa,
    exact_agreement_rate,
    mean_absolute_error,
    evaluate_ratings,
)

def test_perfect_agreement():
    rater_a = [0, 1, 2, 3, 4, 3, 2, 1]
    rater_b = [0, 1, 2, 3, 4, 3, 2, 1]

    metrics = evaluate_ratings(rater_a, rater_b)
    assert metrics["quadratic_weighted_kappa"] == 1.0
    assert metrics["exact_agreement_rate"] == 1.0
    assert metrics["mean_absolute_error"] == 0.0

def test_quadratic_weighting_penalizes_large_discrepancies():
    # Rater A vs Rater B: off by 1 on every sample
    r_a = [1, 2, 3, 4]
    r_b_off_1 = [2, 3, 4, 3]

    # Rater A vs Rater C: off by 3 on every sample
    r_c_off_3 = [4, 0, 0, 1]

    kappa_off_1 = quadratic_weighted_kappa(r_a, r_b_off_1)
    kappa_off_3 = quadratic_weighted_kappa(r_a, r_c_off_3)

    mae_1 = mean_absolute_error(r_a, r_b_off_1)
    mae_3 = mean_absolute_error(r_a, r_c_off_3)

    assert kappa_off_1 > kappa_off_3
    assert mae_1 < mae_3

def test_exact_agreement_and_mae():
    r1 = [0, 2, 4, 3]
    r2 = [0, 2, 2, 3]  # 3 out of 4 exact matches, one off by 2

    assert exact_agreement_rate(r1, r2) == 0.75
    assert mean_absolute_error(r1, r2) == 0.5

def test_input_validation():
    with pytest.raises(ValueError, match="Rater sequences must have identical lengths"):
        quadratic_weighted_kappa([1, 2], [1])

    with pytest.raises(ValueError, match="Ratings must be between"):
        quadratic_weighted_kappa([5, 2], [1, 2])

    with pytest.raises(ValueError, match="Ratings must be between"):
        quadratic_weighted_kappa([-1, 2], [1, 2])

    with pytest.raises(ValueError):
        exact_agreement_rate([1], [1, 2])

    with pytest.raises(ValueError):
        mean_absolute_error([1], [1, 2])

def test_empty_sequences():
    assert quadratic_weighted_kappa([], []) == 1.0
    assert exact_agreement_rate([], []) == 1.0
    assert mean_absolute_error([], []) == 0.0
