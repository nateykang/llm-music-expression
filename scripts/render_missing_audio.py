#!/usr/bin/env python3
"""Render MP3 audio for any free-form piece that doesn't have it yet, so the full
set is available for the audio-emotion (Music2Emo) leg. Uses the project's existing
fluidsynth+lame pipeline (render.midi_to_audio). Updates each batch's data.json with
the audio path. Idempotent: skips pieces whose MP3 already exists.

Usage:  python scripts/render_missing_audio.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from music21 import converter

    from llm_music.render import abc_to_midi, audio_available, midi_to_audio

    if not audio_available():
        print("fluidsynth / lame / soundfont not all available — cannot render", file=sys.stderr)
        return 1

    done = fail = skip = 0
    for bd in sorted((ROOT / "docs/data").glob("2026*")):
        dj = bd / "data.json"
        if not dj.exists():
            continue
        manifest = json.loads(dj.read_text(encoding="utf-8"))
        changed = False
        for p in manifest["pieces"]:
            if p.get("prompt") != "free-form" or not p.get("ok"):
                continue
            suffix = f"_s{p['sample']}" if p.get("sample") else ""
            arel = f"audio/free-form/{p['model']}{suffix}.mp3"
            apath = bd / arel
            if apath.exists():  # already rendered (maybe just missing the manifest field)
                if p.get("audio") != arel:
                    p["audio"] = arel
                    changed = True
                skip += 1
                continue
            apath.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="render_aud_") as td:
                tdp = Path(td)
                midi = None
                try:
                    if p.get("abc"):
                        midi = abc_to_midi(p["abc"], tdp)
                    elif p.get("score"):
                        midi = tdp / "piece.mid"
                        converter.parse(str(bd / p["score"])).write("midi", fp=str(midi))
                except Exception:
                    midi = None
                ok = bool(midi) and Path(midi).exists() and midi_to_audio(Path(midi), apath)
            if ok:
                p["audio"] = arel
                changed = True
                done += 1
            else:
                fail += 1
            if (done + fail) % 25 == 0:
                print(f"  rendered {done}, failed {fail} (skipped {skip})", flush=True)
        if changed:
            dj.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"=== AUDIO RENDER DONE: {done} rendered, {fail} failed, {skip} already present ===")


if __name__ == "__main__":
    sys.exit(main())
