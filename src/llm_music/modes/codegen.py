"""Code-gen mode: the LLM writes music21 Python, executed in the sandbox.

Faithful to the original sara-fish experiment, but the untrusted code runs in an
isolated subprocess (see sandbox.py) instead of a bare exec.
"""

from __future__ import annotations

from pathlib import Path

from ..sandbox import run_music21_code
from ._common import ModeResult, extract_json

# The "## Outputs" section of the prompt. Mirrors sara-fish's prompt.md, with the
# code contract adapted to our sandbox (bind a top-level `score`, no render()).
OUTPUTS = """\
## Outputs

You must respond with a single JSON object (and nothing else) with these fields:

- `code`: Your complete Python script as a string. It should import `music21`,
  build a music21 Score, and bind it to a top-level variable named `score`. Do
  not call `.show()` or `.write()` — just construct `score`. Only use real
  music21 instrument names.
- `title`: A short title for your piece.
- `short_description`: A single sentence describing your musical intent.
- `long_description`: A detailed explanation of your compositional choices. Can be any length.
"""

# Codegen gets the music21 toolkit documentation appended (see prompts/toolkit.md).
USES_TOOLKIT = True


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
