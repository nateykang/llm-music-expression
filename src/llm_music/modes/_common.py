"""Shared helpers for parsing LLM responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ModeResult:
    """Outcome of a single generation attempt within a mode."""

    ok: bool
    title: str = ""
    short_description: str = ""
    long_description: str = ""
    midi_path: object = None  # pathlib.Path when ok
    musicxml_path: object = None  # pathlib.Path when ok
    error: str | None = None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull a single JSON object out of an LLM response, tolerating code fences."""
    text = text.strip()
    candidates = []
    fenced = _FENCE_RE.findall(text)
    candidates.extend(fenced)
    candidates.append(text)
    # Also try the substring between the first '{' and last '}'.
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{") : text.rindex("}") + 1])

    last_err: Exception | None = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(f"could not parse JSON from model response: {last_err}")
