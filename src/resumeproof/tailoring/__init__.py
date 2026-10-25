"""Tailoring engine and hybrid matcher for defensible resumes."""

from resumeproof.tailoring.matcher import HybridMatcher, MatchScore
from resumeproof.tailoring.engine import TailoringEngine, TailoringResult, GapReport, RequirementGap

__all__ = [
    "HybridMatcher",
    "MatchScore",
    "TailoringEngine",
    "TailoringResult",
    "GapReport",
    "RequirementGap",
]
