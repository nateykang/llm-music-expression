"""ABC-direct mode: the LLM writes ABC notation; abcjs renders it client-side.

No code execution. The raw ABC text is the stored artifact — the browser engraves
and plays it with abcjs, which (unlike music21's ABC reader) handles multi-voice
ABC correctly. We keep only a coarse server-side syntax gate to trigger retries.
"""

from __future__ import annotations

from pathlib import Path

from ._common import ModeResult, extract_json

# The "## Outputs" section of the prompt. ABC mode is our addition (sara is
# codegen-only); it executes no code and needs no toolkit documentation.
OUTPUTS = """\
## Outputs

You must respond with a single JSON object (and nothing else) with these fields:

- `abc`: The complete piece written in ABC notation, including the header fields
  X, T, M, L, K and all the music. Multiple voices are welcome via V: lines.
- `title`: A short title for your piece.
- `short_description`: A single sentence describing your musical intent.
- `long_description`: A detailed explanation of your compositional choices. Can be any length.
"""

USES_TOOLKIT = False


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

    # The raw ABC is the artifact: abcjs engraves and plays it client-side, which
    # handles multi-voice ABC correctly (music21's ABC reader silently collapses
    # voices into one staff). We only do a coarse syntax sanity check here so a
    # truly malformed body triggers a retry; we do NOT use music21's output.
    err = _abc_syntax_error(abc)
    if err:
        return ModeResult(ok=False, error=f"ABC looks malformed: {err}")

    abc = abc.strip()
    # Pre-bake audio via the canonical ABC->MIDI tool (abc2midi); generate_piece
    # turns the MIDI into MP3. The raw ABC is still stored for abcjs notation.
    from ..render import abc_to_midi

    midi_path = abc_to_midi(abc, work_dir)

    return ModeResult(
        ok=True,
        title=str(obj.get("title", "Untitled")),
        short_description=str(obj.get("short_description", "")),
        long_description=str(obj.get("long_description", "")),
        abc=abc,
        midi_path=midi_path,
    )


def _abc_syntax_error(abc: str) -> str | None:
    """Coarse validity gate: require the essential ABC headers. Returns an error
    string to trigger a regeneration, or None if the body looks like real ABC."""
    has_key = any(line.lstrip().startswith("K:") for line in abc.splitlines())
    has_index = any(line.lstrip().startswith("X:") for line in abc.splitlines())
    if not (has_key and has_index):
        return "missing required X: and/or K: header lines"
    return None
