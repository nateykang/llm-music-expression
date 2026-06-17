#!/usr/bin/env python3
"""Re-bake a batch's audio (and clefs) from its stored MusicXML.

Repairs batches generated before audio was derived from MusicXML: parses each
score, inserts missing clefs, and re-renders the .ogg from that score so the
audio matches the engraving (in particular, no dropped grand-staff bass staff).
No model calls. Usage:

    python scripts/rebake_audio.py docs/data/<batch_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from music21 import converter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_music.render import ensure_clefs, midi_to_audio  # noqa: E402


def rebake(batch: Path) -> None:
    scores = sorted(batch.glob("scores/*/*.musicxml"))
    if not scores:
        print(f"no scores under {batch}")
        return
    for xml in scores:
        rel = xml.relative_to(batch / "scores").with_suffix("")
        audio = batch / "audio" / rel.with_suffix(".ogg")
        score = converter.parse(str(xml))
        ensure_clefs(score)
        score.write("musicxml", fp=str(xml))  # rewrite with clefs
        mid = xml.with_suffix(".mid")
        score.write("midi", fp=str(mid))
        ok = midi_to_audio(mid, audio)
        mid.unlink(missing_ok=True)
        print(f"  {'ok ' if ok else 'NO-AUDIO'} {rel}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    rebake(Path(sys.argv[1]))
