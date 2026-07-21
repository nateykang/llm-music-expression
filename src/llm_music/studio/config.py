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

_SECRET_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                "STUDIO_PASSWORD", "STUDIO_SECRET", "STUDIO_BACKUP_REPO")


def sanitize_env() -> None:
    """Strip stray whitespace from secret env vars. Console UIs (e.g. RunPod's
    env-var fields) make it easy to paste a trailing newline; a newline inside
    an API key becomes httpcore's 'Illegal header value', which the SDKs
    surface as a bogus 'Connection error' — hours of fun to diagnose."""
    for var in _SECRET_VARS:
        val = os.environ.get(var)
        if val and val.strip() != val:
            os.environ[var] = val.strip()

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


# Hidden from the studio picker (still registered for batch runs, and existing
# sessions that use them keep working). kimi-k3: ~10 min per generation via
# OpenRouter — one kimi cell stalls a whole comparison round.
HIDDEN_MODELS = {"kimi-k3"}


def available_models() -> list[str]:
    """Friendly model ids the studio offers: every registry entry whose provider
    has a studio backend (anthropic, openai, openrouter — see backends.py).
    STUDIO_MODELS narrows the list, e.g. to models whose keys are on the server
    (an explicit list also overrides HIDDEN_MODELS)."""
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
    return [m for m in supported if m not in HIDDEN_MODELS]


def redact(text: str) -> str:
    """Blank secret values out of text before it reaches logs, transcripts, or
    the browser. Exception messages can embed them — httpcore quotes the whole
    'illegal header value', i.e. the API key itself."""
    for var in _SECRET_VARS:
        val = (os.environ.get(var) or "").strip()
        if val and val in text:
            text = text.replace(val, f"[{var}]")
    return text


# At import so every entry point (server, __main__, tests) sees clean values.
sanitize_env()


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
