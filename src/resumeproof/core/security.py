"""Security utilities: cryptographic session signing, CSRF verification, rate limiting, and answer screening."""

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from typing import Optional

from resumeproof.core.config import settings
from resumeproof.verifier.injection import AdversarialDetector, normalize_unicode_homoglyphs

_detector = AdversarialDetector()

# ---------------- Cryptographic Signed Session Tokens ----------------

def _get_secret() -> bytes:
    return settings.secret_key.encode("utf-8")

def create_signed_session_token(user_id: str, extra_data: Optional[dict] = None) -> str:
    """Generates an HMAC-SHA256 signed session token containing user_id and timestamp."""
    payload = {
        "user_id": user_id,
        "nonce": secrets.token_hex(8),
        "created_at": datetime.now(timezone.utc).timestamp(),
    }
    if extra_data:
        payload.update(extra_data)

    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")

    signature = hmac.new(_get_secret(), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"

def verify_signed_session_token(token: str, max_age_seconds: int = 604800) -> Optional[dict]:
    """Verifies HMAC signature and expiration on session token. Returns payload dict or None."""
    if not token or "." not in token:
        return None

    try:
        parts = token.split(".", 1)
        payload_b64, signature = parts[0], parts[1]

        expected_sig = hmac.new(_get_secret(), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None

        # Add padding back for base64 decode
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))

        # Check expiration
        created_at = payload.get("created_at", 0)
        now = datetime.now(timezone.utc).timestamp()
        if now - created_at > max_age_seconds:
            return None

        return payload
    except Exception:
        return None

# ---------------- CSRF Protection ----------------

def generate_csrf_token(seed: str) -> str:
    """Generates an HMAC CSRF token bound to a user or session seed."""
    msg = f"csrf:{seed}".encode("utf-8")
    sig = hmac.new(_get_secret(), msg, hashlib.sha256).hexdigest()[:32]
    return sig

def verify_csrf_token(token: str, seed: str) -> bool:
    """Verifies a CSRF token against the given seed."""
    if not token or not seed:
        return False
    expected = generate_csrf_token(seed)
    return hmac.compare_digest(token.strip(), expected)

# ---------------- Sliding Window Rate Limiter ----------------

class RateLimiter:
    """Thread-safe sliding window rate limiter per client identifier."""

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
        requests_per_minute: Optional[int] = None,
    ):
        if requests_per_minute is not None:
            max_requests = requests_per_minute
            window_seconds = 60
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.history: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            timestamps = self.history.get(client_id, [])
            # Purge timestamps outside the sliding window
            cutoff = now - self.window_seconds
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= self.max_requests:
                self.history[client_id] = timestamps
                return False

            timestamps.append(now)
            self.history[client_id] = timestamps
            return True

    def reset(self) -> None:
        with self._lock:
            self.history.clear()

global_rate_limiter = RateLimiter(max_requests=120, window_seconds=60)

# ---------------- Candidate Answer Security Screening ----------------

def screen_candidate_answer(answer_text: str) -> dict:
    """Screens candidate answers for prompt injection, zero-width characters, and adversarial obfuscation.

    Returns:
        dict: {"is_adversarial": bool, "reasons": list[str]}
    """
    reasons = []
    # Check zero-width characters and invisible formatting
    zw_matches = re.findall(r"[\u200B-\u200D\uFEFF\u00AD\u2060\u200E\u200F\u202A-\u202E\u2066-\u2069]", answer_text)
    if zw_matches:
        reasons.append(f"Adversarial evasion attempt: text contains {len(zw_matches)} hidden zero-width or formatting characters.")

    is_adv, det_reasons = _detector.inspect(answer_text)
    if is_adv:
        reasons.extend(det_reasons)

    return {
        "is_adversarial": len(reasons) > 0,
        "reasons": reasons,
    }
