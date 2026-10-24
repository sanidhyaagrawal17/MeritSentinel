"""Mastery math calculator: Half-life decay, aggregate jump cap (1.0), and configurable eligibility."""

from datetime import datetime, timezone
from resumeproof.core.config import settings
from resumeproof.schemas.mastery import AnswerRecord, MasteryState

def compute_mastery_state(
    claim_id: str,
    answers: list[AnswerRecord],
    previous_state: MasteryState | None = None,
    evaluation_time: datetime | None = None,
    effective_threshold: float | None = None,
    confidence_threshold: float | None = None,
    max_jump: float | None = None,
    half_life_days: float | None = None,
    prior_score: float | None = None,
    prior_weight: float | None = None,
) -> MasteryState:
    """Computes the updated mastery state given all answers for a claim.

    Invariant Rules:
    1. Raw answer scores are immutable and uncapped (0-4).
    2. Aggregate score change is capped at max_jump (default 1.0) relative to previous_state.aggregate_score.
    3. Half-life decay weight: 0.5 ** (age_in_days / half_life_days).
    4. Confidence: n_eff / (n_eff + prior_weight).
    5. Effective score: prior + (aggregate_score - prior) * confidence.
    6. Eligibility: effective_score >= effective_threshold AND confidence >= confidence_threshold.
    """
    if effective_threshold is None:
        effective_threshold = settings.effective_score_threshold
    if confidence_threshold is None:
        confidence_threshold = settings.confidence_threshold
    if max_jump is None:
        max_jump = settings.max_aggregate_jump
    if half_life_days is None:
        half_life_days = settings.half_life_days
    if prior_score is None:
        prior_score = settings.prior_score
    if prior_weight is None:
        prior_weight = settings.prior_weight

    now = evaluation_time or datetime.now(timezone.utc)

    if not answers:
        return MasteryState(
            claim_id=claim_id,
            aggregate_score=prior_score,
            effective_score=prior_score,
            confidence=0.0,
            n_effective=0.0,
            total_answers=0,
            is_eligible=False,
            last_updated=now,
        )

    # Sort answers chronologically
    sorted_answers = sorted(answers, key=lambda a: a.timestamp)

    # Calculate weighted sum with half-life decay
    total_weight = 0.0
    weighted_score_sum = 0.0

    for a in sorted_answers:
        age_days = max(0.0, (now - a.timestamp).total_seconds() / 86400.0)
        w = 0.5 ** (age_days / half_life_days)
        total_weight += w
        weighted_score_sum += w * float(a.raw_score)

    uncapped_aggregate = weighted_score_sum / max(1e-8, total_weight)

    # Apply 1.0 jump cap to the change in aggregate score per new answer
    prev_agg = previous_state.aggregate_score if previous_state is not None else prior_score
    delta = uncapped_aggregate - prev_agg

    if abs(delta) > max_jump:
        capped_aggregate = prev_agg + (max_jump if delta > 0 else -max_jump)
    else:
        capped_aggregate = uncapped_aggregate

    # Clamp aggregate score to valid range [0.0, 4.0]
    capped_aggregate = max(0.0, min(4.0, capped_aggregate))

    # Calculate confidence and Bayesian shrinkage effective score
    n_eff = total_weight
    confidence = n_eff / (n_eff + prior_weight)
    effective_score = prior_score + (capped_aggregate - prior_score) * confidence

    rounded_aggregate = round(capped_aggregate, 4)
    rounded_effective = round(effective_score, 4)
    rounded_confidence = round(confidence, 4)
    rounded_neff = round(n_eff, 4)

    is_eligible = (rounded_effective >= effective_threshold) and (rounded_confidence >= confidence_threshold)

    return MasteryState(
        claim_id=claim_id,
        aggregate_score=rounded_aggregate,
        effective_score=rounded_effective,
        confidence=rounded_confidence,
        n_effective=rounded_neff,
        total_answers=len(sorted_answers),
        is_eligible=is_eligible,
        last_updated=now,
    )
