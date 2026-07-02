"""Shared retry classification + backoff for LLM API calls.

Used by both generation (generate.py) and judging (judge.py) so the two paths
agree on what is worth retrying and how to back off.
"""

from __future__ import annotations

import random
import time

# Seeded so retry timing is reproducible across runs; jitter still spreads
# concurrent workers because each draw advances the shared stream.
_rng = random.Random(0)


def is_retryable(exc: Exception) -> bool:
    """Whether re-issuing the same request could plausibly succeed.

    An attached HTTP status decides: 408/409/429 and 5xx (incl. Anthropic's 529
    "overloaded") are transient; other 4xx — bad/unknown/unverified model, bad
    key, malformed request — are permanent, and retrying just burns attempts
    (e.g. an unverified org requesting `o3` 400s five times in a row). The
    status is checked *first* so a permanent error whose message happens to
    mention "connection" or "try again" isn't misread as transient.

    Without a status (SDKs surface stream drops and socket errors as plain
    exceptions), assume a network/transport hiccup and retry.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    return True


def backoff_sleep(attempt: int, cap: float = 20.0) -> float:
    """Exponential backoff with jitter: sleep 2^attempt (capped) scaled by a
    random factor in [0.5, 1.0] so parallel workers don't retry in lockstep.
    Returns the duration slept."""
    delay = min(2 ** attempt, cap) * _rng.uniform(0.5, 1.0)
    time.sleep(delay)
    return delay
