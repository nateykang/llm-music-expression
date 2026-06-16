"""ABC-direct mode: the LLM writes ABC notation, parsed by music21.

No code execution — the safe/reproducible path. The ABC body is parsed with
music21.converter, which then exports MIDI + MusicXML like any other Score.
"""

from __future__ import annotations

from pathlib import Path

from ..render import write_score
from ._common import ModeResult, extract_json

INSTRUCTIONS = """\
Respond with a single JSON object (and nothing else), with these string fields:
  - "abc": the complete piece written in ABC notation (include the header fields
    X, T, M, L, K and all the music). Multiple voices are welcome via V: lines.
  - "title": a short title for the piece.
  - "short_description": one sentence on what you expressed.
  - "long_description": a paragraph reflecting on your intent and choices.
"""


def build_user_prompt(base: str, prior_error: str | None) -> str:
    if not prior_error:
        return base
    return (
        base
        + "\n\nYour previous ABC could not be parsed. Fix it and try again:\n"
        + f"```\n{prior_error}\n```"
    )


def generate(response_text: str, work_dir: Path) -> ModeResult:
    try:
        obj = extract_json(response_text)
    except ValueError as e:
        return ModeResult(ok=False, error=str(e))

    abc = obj.get("abc")
    if not isinstance(abc, str) or not abc.strip():
        return ModeResult(ok=False, error="response JSON missing non-empty 'abc' field")

    try:
        from music21 import converter

        score = converter.parse(abc, format="abc")
    except Exception as e:  # music21 raises various parse errors
        return ModeResult(ok=False, error=f"ABC parse failed: {e}")

    work_dir.mkdir(parents=True, exist_ok=True)
    midi_path = work_dir / "piece.mid"
    xml_path = work_dir / "piece.musicxml"
    try:
        write_score(score, midi_path, xml_path)
    except Exception as e:
        return ModeResult(ok=False, error=f"score export failed: {e}")

    return ModeResult(
        ok=True,
        title=str(obj.get("title", "Untitled")),
        short_description=str(obj.get("short_description", "")),
        long_description=str(obj.get("long_description", "")),
        midi_path=midi_path,
        musicxml_path=xml_path,
    )
