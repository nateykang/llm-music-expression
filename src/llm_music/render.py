"""Rendering helpers: music21 Score -> MIDI/MusicXML, and MIDI -> audio.

Audio rendering uses FluidSynth + a SoundFont. If either is unavailable the
pipeline degrades gracefully: scores still render (Verovio engraves MusicXML in
the browser), only the pre-baked audio file is skipped.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import find_soundfont


def write_score(score, midi_path: Path, musicxml_path: Path) -> None:
    """Write a music21 Score to MIDI and MusicXML (used by ABC mode).

    music21's ABC export is fragile in two ways that otherwise burn generation
    retries:
      1. the parser inserts the same context object (e.g. a TimeSignature) into
         two Streams, so ``.write()`` raises "already found in this Stream";
      2. multi-voice ABC can yield Parts without Measure objects, so MIDI's
         repeat expansion raises "cannot process repeats on Stream that does not
         have measures".

    We try a direct write first, and on failure fall back to a normalized
    variant: dedupe duplicate instances, run ``makeNotation`` to guarantee
    measures, and (for MIDI only) strip repeats so playback export can't choke.
    Repeats are kept in the engraved MusicXML.
    """
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    musicxml_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_clefs(score)
    _write_one(score, "musicxml", musicxml_path, strip_repeats=False)
    _write_one(score, "midi", midi_path, strip_repeats=True)


def _write_one(score, fmt: str, path: Path, strip_repeats: bool) -> None:
    try:
        score.write(fmt, fp=str(path))
        return
    except Exception:
        safe = _normalize_for_export(score, strip_repeats=strip_repeats)
        safe.write(fmt, fp=str(path))  # let a second failure propagate


def _normalize_for_export(score, strip_repeats: bool):
    """Return an export-safe deepcopy: deduped, measured, optionally repeat-free."""
    import copy

    clean = copy.deepcopy(score)

    # (1) Drop duplicate object *instances* (deepcopy keeps shared refs shared).
    seen: set[int] = set()
    for container in clean.recurse(streamsOnly=True, includeSelf=True):
        for el in list(container):
            if id(el) in seen:
                container.remove(el)
            else:
                seen.add(id(el))

    # (2) Strip repeat marks/barlines so MIDI export won't try to expand them.
    if strip_repeats:
        from music21 import bar, repeat

        for el in list(clean.recurse().getElementsByClass(repeat.RepeatMark)):
            if el.activeSite is not None:
                el.activeSite.remove(el)
        for b in list(clean.recurse().getElementsByClass(bar.Repeat)):
            if b.activeSite is not None:
                b.activeSite.remove(b)

    # (3) Guarantee measures exist.
    try:
        clean = clean.makeNotation(inPlace=False)
    except Exception:
        pass
    return clean


def ensure_clefs(score) -> None:
    """Give every part a clef if it lacks one.

    Models often omit clefs; music21's MusicXML export then emits a part with no
    clef (Verovio falls back to treble, which is wrong for a bass/LH staff).
    Insert a best-guess clef from each part's pitch range so the engraving is
    correct. In-place; safe to call on any Score/Part.
    """
    from music21 import clef

    parts = list(getattr(score, "parts", [])) or [score]
    for p in parts:
        if list(p.recurse().getElementsByClass(clef.Clef)):
            continue
        try:
            best = clef.bestClef(p, recurse=True)
        except Exception:
            continue
        target = p.recurse().getElementsByClass("Measure").first() or p
        target.insert(0, best)


def audio_available() -> bool:
    return (
        shutil.which("fluidsynth") is not None
        and find_soundfont() is not None
        and shutil.which("lame") is not None
    )


def midi_to_audio(midi_path: Path, audio_path: Path, timeout: int = 120) -> bool:
    """Render MIDI -> MP3 (FluidSynth to WAV, then lame to MP3). Returns False if
    skipped (no FluidSynth/SoundFont/lame).

    MP3 rather than FluidSynth's direct Ogg output: that Ogg carries broken length
    metadata, so browsers misreport the duration (e.g. 254s for a 67s piece) and
    playback breaks in Chrome. MP3 has reliable duration and plays everywhere
    (incl. Safari/iOS, which can't play Ogg Vorbis at all).
    """
    import os
    import tempfile

    fluidsynth = shutil.which("fluidsynth")
    soundfont = find_soundfont()
    lame = shutil.which("lame")
    if not fluidsynth or not soundfont or not lame:
        return False

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llm_music_wav_") as tmp:
        wav = os.path.join(tmp, "render.wav")
        try:
            r1 = subprocess.run(
                [fluidsynth, "-ni", "-F", wav, "-T", "wav", "-r", "44100",
                 str(soundfont), str(midi_path)],
                capture_output=True, text=True, timeout=timeout,
            )
            if r1.returncode != 0 or not os.path.exists(wav):
                return False
            r2 = subprocess.run(
                [lame, "--quiet", "-V", "4", wav, str(audio_path)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
    return audio_path.exists()
