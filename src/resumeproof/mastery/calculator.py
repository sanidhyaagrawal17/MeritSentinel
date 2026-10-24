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
    tolerance: float = 1e-4,
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

    # Determine starting aggregate and answer index for incremental jump-capping
    if previous_state is not None:
        if 0 < previous_state.total_answers < len(sorted_answers):
            start_idx = previous_state.total_answers
        elif previous_state.total_answers == 0 and len(sorted_answers) > 1 and previous_state.aggregate_score != prior_score:
            start_idx = len(sorted_answers) - 1
        elif previous_state.total_answers >= len(sorted_answers):
            start_idx = len(sorted_answers)
        else:
            start_idx = 0
        current_agg = previous_state.aggregate_score
    else:
        start_idx = 0
        current_agg = prior_score

    # Apply max_jump cap (1.0) sequentially per new answer
    for k in range(start_idx, len(sorted_answers)):
        sub_answers = sorted_answers[: k + 1]
        eval_t = sub_answers[-1].timestamp
        sub_weights = [
            0.5 ** (max(0.0, (eval_t - a.timestamp).total_seconds() / 86400.0) / half_life_days)
            for a in sub_answers
        ]
        total_sub_w = sum(sub_weights)
        uncapped_k = sum(w * float(a.raw_score) for w, a in zip(sub_weights, sub_answers)) / max(1e-8, total_sub_w)
        delta_k = uncapped_k - current_agg
        if abs(delta_k) > max_jump:
            current_agg += max_jump if delta_k > 0 else -max_jump
        else:
            current_agg = uncapped_k
        current_agg = max(0.0, min(4.0, current_agg))

    capped_aggregate = current_agg

    # Calculate total weight and n_effective at the target evaluation time
    total_weight = 0.0
    for a in sorted_answers:
        age_days = max(0.0, (now - a.timestamp).total_seconds() / 86400.0)
        w = 0.5 ** (age_days / half_life_days)
        total_weight += w

    # Calculate confidence and Bayesian shrinkage effective score
    n_eff = total_weight
    confidence = n_eff / (n_eff + prior_weight)
    effective_score = prior_score + (capped_aggregate - prior_score) * confidence

    rounded_aggregate = round(capped_aggregate, 4)
    rounded_effective = round(effective_score, 4)
    rounded_confidence = round(confidence, 4)
    rounded_neff = round(n_eff, 4)
    # Apply small tolerance (1e-4) to prevent boundary artifacts on threshold checks
    is_eligible = (
        (effective_score >= (effective_threshold - tolerance)) and
        (confidence >= (confidence_threshold - tolerance))
    )

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
