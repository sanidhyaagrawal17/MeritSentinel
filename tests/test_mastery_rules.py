"""Tests for mastery math rules: 1.0 aggregate jump cap, immutable raw scores, and half-life decay."""

from datetime import datetime, timezone, timedelta
from resumeproof.schemas.mastery import AnswerRecord, MasteryState
from resumeproof.mastery.calculator import compute_mastery_state

def test_raw_scores_remain_uncapped_and_immutable():
    claim_id = "claim_1"
    now = datetime.now(timezone.utc)
    # Answer record awarded maximum score 4
    ans = AnswerRecord(id="ans_1", claim_id=claim_id, raw_score=4, timestamp=now)
    assert ans.raw_score == 4

    state = compute_mastery_state(claim_id=claim_id, answers=[ans], evaluation_time=now)
    # Raw answer is still 4
    assert ans.raw_score == 4
    # Aggregate score jumped from prior 2.0 to capped 3.0 (jump cap of 1.0 applied)
    assert state.aggregate_score == 3.0

def test_max_jump_cap_on_positive_jump():
    claim_id = "claim_jump_up"
    now = datetime.now(timezone.utc)

    # Starting state with aggregate 2.0
    initial_state = MasteryState(
        claim_id=claim_id,
        aggregate_score=2.0,
        effective_score=2.0,
        confidence=0.0
    )

    # 1st near-perfect answer (raw_score=4)
    ans1 = AnswerRecord(id="a1", claim_id=claim_id, raw_score=4, timestamp=now)
    state1 = compute_mastery_state(
        claim_id=claim_id,
        answers=[ans1],
        previous_state=initial_state,
        evaluation_time=now,
        max_jump=1.0
    )
    # Even though candidate average is 4.0, aggregate jump is capped at 2.0 + 1.0 = 3.0
    assert state1.aggregate_score == 3.0

    # 2nd near-perfect answer (raw_score=4)
    ans2 = AnswerRecord(id="a2", claim_id=claim_id, raw_score=4, timestamp=now + timedelta(minutes=5))
    state2 = compute_mastery_state(
        claim_id=claim_id,
        answers=[ans1, ans2],
        previous_state=state1,
        evaluation_time=now + timedelta(minutes=5),
        max_jump=1.0
    )
    # Previous aggregate was 3.0; candidate average is 4.0. Jump is +1.0 -> aggregate becomes 4.0
    assert state2.aggregate_score == 4.0

def test_max_jump_cap_on_downward_drop():
    claim_id = "claim_jump_down"
    now = datetime.now(timezone.utc)

    # Candidate established mastery at aggregate 4.0
    established_state = MasteryState(
        claim_id=claim_id,
        aggregate_score=4.0,
        effective_score=3.5,
        confidence=0.75,
        n_effective=6.0
    )

    # Candidate gives a 0 answer (fails question completely)
    # Past 3 answers were 4, 4, 4; new answer is 0
    ans_list = [
        AnswerRecord(id=f"a_{i}", claim_id=claim_id, raw_score=4, timestamp=now - timedelta(hours=i))
        for i in range(1, 4)
    ]
    ans_new = AnswerRecord(id="a_fail", claim_id=claim_id, raw_score=0, timestamp=now)
    all_answers = ans_list + [ans_new]

    state = compute_mastery_state(
        claim_id=claim_id,
        answers=all_answers,
        previous_state=established_state,
        evaluation_time=now,
        max_jump=1.0
    )

    # Aggregate can only drop by at most 1.0 point (from 4.0 to 3.0)
    assert state.aggregate_score == 3.0
    # Raw score remains intact
    assert ans_new.raw_score == 0

def test_half_life_time_decay():
    claim_id = "claim_decay"
    eval_time = datetime.now(timezone.utc)
    # Answer given 30 days ago with score 4
    old_time = eval_time - timedelta(days=30)
    ans = AnswerRecord(id="a_old", claim_id=claim_id, raw_score=4, timestamp=old_time)

    state = compute_mastery_state(
        claim_id=claim_id,
        answers=[ans],
        evaluation_time=eval_time,
        half_life_days=30.0
    )
    # Weight should have decayed by 50% (from 1.0 to 0.5)
    assert abs(state.n_effective - 0.5) < 0.01
    # Confidence under Config 3 (k=1.5): 0.5 / (0.5 + 1.5) = 0.25
    assert abs(state.confidence - 0.25) < 0.01

def test_batch_replay_matches_sequential_execution():
    claim_id = "claim_replay"
    now = datetime.now(timezone.utc)
    ans1 = AnswerRecord(id="a1", claim_id=claim_id, raw_score=4, timestamp=now)
    ans2 = AnswerRecord(id="a2", claim_id=claim_id, raw_score=4, timestamp=now + timedelta(minutes=5))

    # Sequential execution:
    # 1st answer jumps aggregate from prior 2.0 to capped 3.0
    state1 = compute_mastery_state(claim_id=claim_id, answers=[ans1], evaluation_time=now)
    assert state1.aggregate_score == 3.0
    # 2nd answer jumps aggregate from 3.0 to capped 4.0
    state2 = compute_mastery_state(
        claim_id=claim_id,
        answers=[ans1, ans2],
        previous_state=state1,
        evaluation_time=now + timedelta(minutes=5)
    )
    assert state2.aggregate_score == 4.0

    # Batch replay from scratch (e.g. database reload without in-memory state):
    # Evaluates per new answer sequentially and MUST produce the exact same aggregate score 4.0
    replayed_state = compute_mastery_state(
        claim_id=claim_id,
        answers=[ans1, ans2],
        previous_state=None,
        evaluation_time=now + timedelta(minutes=5)
    )
    assert replayed_state.aggregate_score == state2.aggregate_score
    assert replayed_state.effective_score == state2.effective_score
    assert replayed_state.confidence == state2.confidence
    assert replayed_state.is_eligible == state2.is_eligible
