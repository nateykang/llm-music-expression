"""Render agent-composed music into per-session version folders.

Each successful render becomes ``pieces/v<N>/`` holding the source (music21
code or raw ABC), MusicXML/MIDI where applicable, baked MP3 audio, and a
meta.json. Failed renders leave no version behind — the error goes back to the
model so it can fix its own output, mirroring the retry loop in generate.py.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from ..modes.abc import _abc_syntax_error
from ..render import abc_to_midi, midi_to_audio
from ..sandbox import run_music21_code

log = logging.getLogger(__name__)

VERSION_FILES = ("piece.mp3", "piece.musicxml", "piece.mid", "piece.abc",
                 "source.py", "meta.json")


def next_version(pieces_dir: Path) -> int:
    pieces_dir.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[1:]) for p in pieces_dir.iterdir()
             if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()]
    return max(taken, default=0) + 1


def render_codegen_version(pieces_dir: Path, code: str, title: str, note: str,
                           version: int | None = None, extra: dict | None = None) -> dict:
    version = version if version is not None else next_version(pieces_dir)
    vdir = pieces_dir / f"v{version}"
    shutil.rmtree(vdir, ignore_errors=True)  # explicit-version reruns overwrite cleanly
    sandbox = run_music21_code(code, vdir)
    if not sandbox.ok:
        shutil.rmtree(vdir, ignore_errors=True)
        return {"ok": False, "error": sandbox.error}
    (vdir / "source.py").write_text(code, encoding="utf-8")
    if sandbox.midi_path:
        midi_to_audio(sandbox.midi_path, vdir / "piece.mp3")
    return _finish(vdir, version, "codegen", title, note, extra)


def render_abc_version(pieces_dir: Path, abc: str, title: str, note: str,
                       version: int | None = None, extra: dict | None = None) -> dict:
    err = _abc_syntax_error(abc)
    if err:
        return {"ok": False, "error": f"ABC looks malformed: {err}"}
    abc = abc.strip()
    version = version if version is not None else next_version(pieces_dir)
    vdir = pieces_dir / f"v{version}"
    shutil.rmtree(vdir, ignore_errors=True)
    vdir.mkdir(parents=True, exist_ok=True)
    midi_path = abc_to_midi(abc, vdir)
    # abc_to_midi wrote a normalized piece.abc for abc2midi; store the raw ABC
    # (that is what abcjs engraves client-side, same as the main site).
    (vdir / "piece.abc").write_text(abc + "\n", encoding="utf-8")
    if midi_path:
        midi_to_audio(midi_path, vdir / "piece.mp3")
    return _finish(vdir, version, "abc", title, note, extra)


def _finish(vdir: Path, version: int, mode: str, title: str, note: str,
            extra: dict | None = None) -> dict:
    analysis = _analyze(vdir)
    meta = {
        "ok": True,
        "version": version,
        "mode": mode,
        "title": title,
        "note": note,
        "ts": time.time(),
        "files": [f for f in VERSION_FILES if f != "meta.json" and (vdir / f).exists()],
        "analysis": analysis,
        **(extra or {}),
    }
    (vdir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def _analyze(vdir: Path) -> dict:
    """Quick musical facts fed back to the agent (and shown next to the score).
    Every probe is best-effort — analysis must never fail a render."""
    out: dict = {}
    midi = vdir / "piece.mid"
    if midi.exists():
        try:
            import pretty_midi

            pm = pretty_midi.PrettyMIDI(str(midi))
            out["duration_seconds"] = round(pm.get_end_time(), 1)
            names = [i.name for i in pm.instruments if i.name]
            if names:
                out["midi_tracks"] = names
        except Exception as e:
            log.debug("pretty_midi analysis failed on %s: %s", midi, e)
    xml = vdir / "piece.musicxml"
    if xml.exists():
        try:
            from music21 import converter

            score = converter.parse(str(xml))
            parts = list(score.parts)
            out["n_parts"] = len(parts)
            out["parts"] = [p.partName for p in parts if p.partName][:8]
            if parts:
                out["n_measures"] = len(parts[0].getElementsByClass("Measure"))
            keys = score.flatten().getElementsByClass("KeySignature")
            if keys:
                out["key_signature"] = str(keys[0])
            tempi = score.flatten().getElementsByClass("MetronomeMark")
            if tempi and tempi[0].number:
                out["tempo_bpm"] = tempi[0].number
        except Exception as e:
            log.debug("music21 analysis failed on %s: %s", xml, e)
    return out
