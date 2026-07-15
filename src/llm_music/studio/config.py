"""Studio configuration. Everything is read from the environment at call time
(not import time) so tests can monkeypatch and the module imports without a key.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from ..config import REPO_ROOT
from ..models.registry import MODEL_REGISTRY

COOKIE_NAME = "studio_auth"
TOKEN_TTL_SECONDS = 30 * 24 * 3600  # re-login monthly
MAX_TURN_STEPS = 8  # model calls per composer message (spend guardrail)

# Random per boot unless pinned: a restart just means re-entering the password.
_boot_secret = secrets.token_bytes(32)


def data_dir() -> Path:
    return Path(os.environ.get("STUDIO_DATA_DIR", str(REPO_ROOT / "studio_data")))


def password() -> str | None:
    return os.environ.get("STUDIO_PASSWORD") or None


def secret() -> bytes:
    env = os.environ.get("STUDIO_SECRET")
    return env.encode() if env else _boot_secret


def notify_url() -> str | None:
    """Optional webhook POSTed on session activity (so you hear about his visits)."""
    return os.environ.get("STUDIO_NOTIFY_URL") or None


def available_models() -> list[str]:
    """Friendly model ids the studio offers: every registry entry whose provider
    has a studio backend (anthropic, openai, openrouter — see backends.py).
    STUDIO_MODELS narrows the list, e.g. to models whose keys are on the server."""
    from .backends import BACKENDS

    env = os.environ.get("STUDIO_MODELS")
    supported = [m for m, spec in MODEL_REGISTRY.items() if spec[0] in BACKENDS]
    if env:
        wanted = [m.strip() for m in env.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in supported]
        if unknown:
            raise ValueError(
                f"STUDIO_MODELS entries without a studio backend: {unknown}"
            )
        return wanted
    return supported


def default_model() -> str:
    """Interactive default. opus-4.8 (no extended thinking) keeps turns snappy;
    fable-5's always-on thinking can sit silent for minutes, which is a poor
    first experience — it stays in the picker for when quality is worth the wait."""
    models = available_models()
    env = os.environ.get("STUDIO_DEFAULT_MODEL")
    if env:
        if env not in models:
            raise ValueError(f"STUDIO_DEFAULT_MODEL '{env}' not in {models}")
        return env
    return "opus-4.8" if "opus-4.8" in models else models[0]
