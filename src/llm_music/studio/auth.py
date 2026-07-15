"""Shared-password auth: HMAC-signed expiring tokens in a cookie, plus a small
in-memory rate limiter so the login endpoint can't be brute-forced.

One trusted collaborator, one password — no accounts, no database. The token is
``<expiry>.<hmac(secret, expiry)>``; verification is constant-time.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict


def make_token(secret: bytes, ttl_seconds: int) -> str:
    exp = str(int(time.time()) + ttl_seconds)
    sig = hmac.new(secret, exp.encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def verify_token(secret: bytes, token: str | None) -> bool:
    if not token or "." not in token:
        return False
    exp, _, sig = token.partition(".")
    if not exp.isdigit() or int(exp) < time.time():
        return False
    want = hmac.new(secret, exp.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, sig)


def check_password(supplied: str, actual: str | None) -> bool:
    if not actual:
        return False
    return hmac.compare_digest(supplied.encode(), actual.encode())


class LoginLimiter:
    """Allow at most `max_failures` failed logins per IP per `window` seconds."""

    def __init__(self, max_failures: int = 5, window: float = 15 * 60):
        self.max_failures = max_failures
        self.window = window
        self._failures: dict[str, list[float]] = defaultdict(list)

    def _prune(self, ip: str) -> None:
        cutoff = time.time() - self.window
        self._failures[ip] = [t for t in self._failures[ip] if t > cutoff]

    def blocked(self, ip: str) -> bool:
        self._prune(ip)
        return len(self._failures[ip]) >= self.max_failures

    def record_failure(self, ip: str) -> None:
        self._prune(ip)
        self._failures[ip].append(time.time())

    def reset(self, ip: str) -> None:
        self._failures.pop(ip, None)
