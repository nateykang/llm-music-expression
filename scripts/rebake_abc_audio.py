#!/usr/bin/env python3
"""Re-bake ABC-mode audio from each piece's stored ABC (abc2midi -> MP3).

Repairs pieces baked before the blank-line fix (a blank line in the ABC body
made abc2midi truncate the tune, producing silent/short audio). Only re-bakes
pieces whose ABC actually contains a blank line — the rest were already fine.
No model calls.

Usage:  python scripts/rebake_abc_audio.py docs/data/<batch_dir> [more_dirs...]
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_music.render import abc_to_midi, midi_to_audio  # noqa: E402


def rebake(batch: Path) -> tuple[int, int]:
    manifest = json.loads((batch / "data.json").read_text(encoding="utf-8"))
    fixed = touched = 0
    for p in manifest["pieces"]:
        abc, audio_rel = p.get("abc"), p.get("audio")
        if not abc or not audio_rel:
            continue
        if not any(not ln.strip() for ln in abc.splitlines()):
            continue  # no blank line -> audio was already correct
        touched += 1
        with tempfile.TemporaryDirectory() as td:
            midi = abc_to_midi(abc, Path(td))
            ok = bool(midi) and midi_to_audio(midi, batch / audio_rel)
        fixed += ok
        tag = f"s{p.get('sample', 0)}"
        print(f"  {'ok ' if ok else 'FAIL'} {p['model']:18} {tag:3} {p['title'][:34]}")
    return touched, fixed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    t = f = 0
    for d in sys.argv[1:]:
        print(f"=== {d} ===")
        ti, fi = rebake(Path(d))
        t += ti
        f += fi
    print(f"\nre-baked {f}/{t} affected pieces")
