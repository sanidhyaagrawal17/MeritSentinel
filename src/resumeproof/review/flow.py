"""Mandatory Review & Confirmation Flow for Profile Facts and Claims."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
import logging

from resumeproof.schemas.facts import ProfileFact
from resumeproof.schemas.claims import Claim, ClaimStatus, ClaimAudit
from resumeproof.store.claim_store import ClaimStore

logger = logging.getLogger("resumeproof.review")

@dataclass
class ReviewDecision:
    action: ClaimStatus  # CONFIRMED, EDITED, REJECTED
    user_text: Optional[str] = None
    user_tools: Optional[list[str]] = None
    user_metrics: Optional[list[str]] = None
    notes: Optional[str] = None

class ReviewSession:
    """Manages the review step where candidate confirms, edits, or rejects claims and facts."""

    def __init__(self, claim_store: ClaimStore):
        self.claim_store = claim_store

    def review_profile_fact(
        self,
        fact: ProfileFact,
        confirm: bool = True,
        correction: Optional[str] = None
    ) -> ProfileFact:
        """Processes review for a Category 1 Profile Fact (not interview-gated)."""
        fact.is_confirmed = confirm
        fact.user_correction = correction
        fact.confirmed_at = datetime.now(timezone.utc) if confirm else None
        self.claim_store.save_profile_fact(fact)
        logger.info(f"Reviewed profile fact {fact.fact_type}: confirmed={confirm}")
        return fact

    def review_claim(
        self,
        claim: Claim,
        decision: ReviewDecision
    ) -> Claim:
        """Processes review for a Category 2 Mastery Claim (interview-gated).

        Records the extractor output vs user correction in an immutable audit entry.
        """
        audit = ClaimAudit(
            extractor_text=claim.text,
            extractor_tools=list(claim.tools),
            extractor_metrics=list(claim.metric_values),
            user_correction_text=decision.user_text,
            user_correction_tools=decision.user_tools,
            user_correction_metrics=decision.user_metrics,
            action_taken=decision.action,
            reviewed_at=datetime.now(timezone.utc),
            notes=decision.notes
        )

        claim.status = decision.action
        claim.audit = audit

        # If edited, update the main fields to reflect the correction
        if decision.action == ClaimStatus.EDITED:
            if decision.user_text:
                claim.text = decision.user_text
            if decision.user_tools is not None:
                claim.tools = decision.user_tools
            if decision.user_metrics is not None:
                claim.metric_values = decision.user_metrics

        self.claim_store.save_claim(claim)
        logger.info(f"Reviewed claim [{claim.id}]: action={decision.action.value}")
        return claim

    def run_batch_review(
        self,
        facts: list[ProfileFact],
        claims: list[Claim],
        fact_callback: Optional[Callable[[ProfileFact], tuple[bool, Optional[str]]]] = None,
        claim_callback: Optional[Callable[[Claim], ReviewDecision]] = None,
        auto_confirm_all: bool = False
    ) -> tuple[list[ProfileFact], list[Claim]]:
        """Processes a batch of facts and claims using provided callbacks or auto-confirm."""
        reviewed_facts: list[ProfileFact] = []
        for fact in facts:
            if auto_confirm_all or fact_callback is None:
                reviewed_facts.append(self.review_profile_fact(fact, confirm=True, correction=None))
            else:
                confirm, correction = fact_callback(fact)
                reviewed_facts.append(self.review_profile_fact(fact, confirm=confirm, correction=correction))

        reviewed_claims: list[Claim] = []
        for claim in claims:
            if auto_confirm_all or claim_callback is None:
                decision = ReviewDecision(action=ClaimStatus.CONFIRMED)
            else:
                decision = claim_callback(claim)
            reviewed_claims.append(self.review_claim(claim, decision))

        return reviewed_facts, reviewed_claims
