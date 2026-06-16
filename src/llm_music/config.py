"""Paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass

# Repo root = three parents up from this file (src/llm_music/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
DOCS_DIR = REPO_ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
SOUNDFONTS_DIR = REPO_ROOT / "soundfonts"


def find_soundfont() -> Path | None:
    """Locate a SoundFont for FluidSynth, or None if audio rendering is unavailable."""
    env = os.environ.get("SOUNDFONT_PATH")
    if env and Path(env).is_file():
        return Path(env)
    if SOUNDFONTS_DIR.is_dir():
        for pattern in ("*.sf2", "*.sf3"):
            hits = sorted(SOUNDFONTS_DIR.glob(pattern))
            if hits:
                return hits[0]
    return None
