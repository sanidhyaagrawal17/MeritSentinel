"""Deterministic strict verifier and adversarial injection defenses."""

from resumeproof.verifier.strict_verifier import (
    StrictVerifier,
    VerificationReport,
    SentenceVerificationResult,
    VerificationBlockedError,
)
from resumeproof.verifier.injection import (
    AdversarialDetector,
    normalize_words_to_numbers,
    normalize_unicode_homoglyphs,
)

__all__ = [
    "StrictVerifier",
    "VerificationReport",
    "SentenceVerificationResult",
    "VerificationBlockedError",
    "AdversarialDetector",
    "normalize_words_to_numbers",
    "normalize_unicode_homoglyphs",
]
