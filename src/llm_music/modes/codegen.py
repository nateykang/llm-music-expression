"""Code-gen mode: the LLM writes music21 Python, executed in the sandbox.

Faithful to the original sara-fish experiment, but the untrusted code runs in an
isolated subprocess (see sandbox.py) instead of a bare exec.
"""

from __future__ import annotations

from pathlib import Path

from ..sandbox import run_music21_code
from ._common import ModeResult, extract_json

INSTRUCTIONS = """\
Respond with a single JSON object (and nothing else), with these string fields:
  - "code": a complete Python script using the `music21` library that builds the
    piece and binds it to a top-level variable named `score` (a music21 Score).
    Do NOT call .show() or .write(); just construct `score`. Only use real
    music21 instrument names.
  - "title": a short title for the piece.
  - "short_description": one sentence on what you expressed.
  - "long_description": a paragraph reflecting on your intent and choices.
"""


def build_user_prompt(base: str, prior_error: str | None) -> str:
    if not prior_error:
        return base
    return (
        base
        + "\n\nYour previous attempt failed with this error. Fix it and try again:\n"
        + f"```\n{prior_error}\n```"
    )


def generate(response_text: str, work_dir: Path) -> ModeResult:
    try:
        obj = extract_json(response_text)
    except ValueError as e:
        return ModeResult(ok=False, error=str(e))

    code = obj.get("code")
    if not isinstance(code, str) or not code.strip():
        return ModeResult(ok=False, error="response JSON missing non-empty 'code' field")

    sandbox = run_music21_code(code, work_dir)
    if not sandbox.ok:
        return ModeResult(ok=False, error=sandbox.error)

    return ModeResult(
        ok=True,
        title=str(obj.get("title", "Untitled")),
        short_description=str(obj.get("short_description", "")),
        long_description=str(obj.get("long_description", "")),
        midi_path=sandbox.midi_path,
        musicxml_path=sandbox.musicxml_path,
    )
