"""SMT-ABC mode: synchronized, bar-interleaved multi-track ABC.

Same machinery as ABC mode (raw ABC stored, abcjs renders it client-side) — only
the output instructions differ. Based on MuPT's Synchronized Multi-Track ABC
(arXiv:2404.06393): writing each bar's voices together keeps the parts aligned in
time, the documented fix for multi-voice ABC misalignment. This arm tests whether
that format helps frontier models zero-shot.
"""

from __future__ import annotations

from . import abc as _abc

# Reuse ABC mode's response handling (extract JSON, store raw ABC, coarse gate).
generate = _abc.generate
build_user_prompt = _abc.build_user_prompt
USES_TOOLKIT = False

OUTPUTS = """\
## Outputs

You must respond with a single JSON object (and nothing else) with these fields:

- `abc`: The complete piece as **multi-track ABC written in synchronized,
  bar-interleaved form**. Two rules make this work:

  1. **Declare every voice once in the header**, each with a clef, a name, and an
     explicit MIDI program so it plays with the right instrument, e.g.

         V:V1 clef=treble name="Violin I"
         %%MIDI program 40
         V:V2 clef=treble name="Violin II"
         %%MIDI program 40
         V:Va clef=alto   name="Viola"
         %%MIDI program 41
         V:Vc clef=bass   name="Cello"
         %%MIDI program 42
         %%score (V1 V2 Va Vc)

  2. **Write the music one bar at a time, all voices together.** For each measure,
     give that bar's content for every voice in order before moving to the next
     measure:

         [V:V1] <bar 1 of Violin I> |
         [V:V2] <bar 1 of Violin II> |
         [V:Va] <bar 1 of Viola> |
         [V:Vc] <bar 1 of Cello> |
         [V:V1] <bar 2 of Violin I> |
         ...

  Every voice must have the **same number of bars**, and **each bar must have the
  same total duration in every voice** — fill silence with rests (`z`). This keeps
  the parts locked in sync. (Single-voice pieces can ignore all of this and just
  write one voice.)
- `title`: A short title for your piece.
- `short_description`: A single sentence describing your musical intent.
- `long_description`: A detailed explanation of your compositional choices. Can be any length.
"""
