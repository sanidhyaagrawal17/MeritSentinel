"""Question planner implementing priority order (untested > weak > stale) and follow-up logic."""

from typing import Optional
from resumeproof.schemas.claims import Claim
from resumeproof.schemas.mastery import MasteryState
from resumeproof.schemas.interview import QuestionPriority

class QuestionPlanner:
    """Plans interview questions according to agreed priority and budget."""

    def __init__(
        self,
        claims: list[Claim],
        mastery_states: dict[str, MasteryState],
        budget: int = 5,
        max_followups_per_claim: int = 1,
    ):
        self.claims = {c.id: c for c in claims}
        self.mastery_states = mastery_states
        self.budget = budget
        self.max_followups_per_claim = max_followups_per_claim
        self.questions_asked = 0
        self.followup_counts: dict[str, int] = {c.id: 0 for c in claims}
        self.questions_per_claim: dict[str, int] = {c.id: 0 for c in claims}
        self.last_claim_id: Optional[str] = None
        self.last_was_followup: bool = False

    def update_mastery_state(self, claim_id: str, state: MasteryState) -> None:
        """Updates the planner's recorded mastery state for a claim."""
        self.mastery_states[claim_id] = state

    def is_budget_exhausted(self) -> bool:
        return self.questions_asked >= self.budget

    def should_followup_on_answer(
        self,
        claim_id: str,
        answer_text: str,
        last_score: Optional[int] = None,
    ) -> tuple[bool, Optional[str]]:
        """Determines if a follow-up should be triggered due to a vague or incomplete answer."""
        if self.is_budget_exhausted():
            return False, None

        if self.followup_counts.get(claim_id, 0) >= self.max_followups_per_claim:
            return False, None

        words = answer_text.strip().split()
        if len(words) < 15:
            return True, "Answer was terse and lacked technical specifics. Probe for concrete architectural mechanics."

        if last_score is not None and last_score <= 1:
            return True, "Answer scored 1 or lower due to vague concepts or missing verification. Request specific parameters and trade-offs."

        return False, None

    def plan_next_claim(
        self,
        last_answer: Optional[str] = None,
        last_score: Optional[int] = None,
    ) -> Optional[tuple[Claim, QuestionPriority, bool, Optional[str]]]:
        """Plans the next claim to test.

        Returns:
            Tuple of (Claim, QuestionPriority, is_followup, followup_reason) or None if budget reached.
        """
        if self.is_budget_exhausted():
            return None

        # Check follow-up condition on previous claim
        if self.last_claim_id and last_answer and not self.last_was_followup:
            should_fu, reason = self.should_followup_on_answer(self.last_claim_id, last_answer, last_score)
            if should_fu and self.last_claim_id in self.claims:
                self.questions_asked += 1
                self.followup_counts[self.last_claim_id] += 1
                self.questions_per_claim[self.last_claim_id] = self.questions_per_claim.get(self.last_claim_id, 0) + 1
                self.last_was_followup = True
                claim = self.claims[self.last_claim_id]
                return claim, QuestionPriority.WEAK, True, reason

        # Otherwise pick next claim by priority: untested > weak > stale
        untested: list[Claim] = []
        weak: list[tuple[Claim, Optional[MasteryState]]] = []
        stale: list[tuple[Claim, MasteryState]] = []

        for cid, claim in self.claims.items():
            state = self.mastery_states.get(cid)
            times_in_session = self.questions_per_claim.get(cid, 0)

            # Untested: never answered before AND not yet asked in this session
            if (state is None or state.total_answers == 0) and times_in_session == 0:
                untested.append(claim)
            elif (state is not None and (not state.is_eligible or state.effective_score < 2.5 or state.confidence < 0.5)) or (state is None and times_in_session > 0):
                weak.append((claim, state))
            elif state is not None:
                stale.append((claim, state))

        chosen_claim: Optional[Claim] = None
        chosen_priority: QuestionPriority = QuestionPriority.UNTESTED

        if untested:
            # Sort untested claims so we rotate through projects/claims evenly:
            projects_seen = []
            claims_by_project: dict[Optional[str], list[Claim]] = {}
            for c in untested:
                pid = c.project_id
                if pid not in claims_by_project:
                    projects_seen.append(pid)
                    claims_by_project[pid] = []
                claims_by_project[pid].append(c)

            # Prioritize project claims (pid is not None) over standalone skill claims (pid is None)
            proj_keys = [p for p in projects_seen if p is not None] + [p for p in projects_seen if p is None]

            # Rotate project order so we do not repeatedly ask the same project if multiple projects exist
            if self.last_claim_id and self.last_claim_id in self.claims and len(proj_keys) > 1:
                last_pid = self.claims[self.last_claim_id].project_id
                if last_pid in proj_keys:
                    idx = proj_keys.index(last_pid)
                    proj_keys = proj_keys[idx + 1:] + proj_keys[:idx + 1]

            # Interleave across distinct projects: round 0 takes 1st claim of each project, etc.
            rotated_untested: list[Claim] = []
            max_len = max(len(claims_by_project[p]) for p in proj_keys)
            for i in range(max_len):
                for p in proj_keys:
                    if i < len(claims_by_project[p]):
                        rotated_untested.append(claims_by_project[p][i])

            chosen_claim = rotated_untested[0]
            chosen_priority = QuestionPriority.UNTESTED
        elif weak:
            # Prefer claims asked fewer times in this session
            weak.sort(key=lambda x: (
                self.questions_per_claim.get(x[0].id, 0),
                x[1].effective_score if x[1] else 0.0
            ))
            chosen_claim = weak[0][0]
            chosen_priority = QuestionPriority.WEAK
        elif stale:
            # Prefer claims asked fewer times in this session, then oldest last_updated
            stale.sort(key=lambda x: (
                self.questions_per_claim.get(x[0].id, 0),
                x[1].last_updated
            ))
            chosen_claim = stale[0][0]
            chosen_priority = QuestionPriority.STALE
        elif self.claims:
            sorted_all = sorted(self.claims.values(), key=lambda c: self.questions_per_claim.get(c.id, 0))
            chosen_claim = sorted_all[0]
            chosen_priority = QuestionPriority.STALE
        else:
            return None

        self.questions_asked += 1
        self.questions_per_claim[chosen_claim.id] = self.questions_per_claim.get(chosen_claim.id, 0) + 1
        self.last_claim_id = chosen_claim.id
        self.last_was_followup = False

        return chosen_claim, chosen_priority, False, None
