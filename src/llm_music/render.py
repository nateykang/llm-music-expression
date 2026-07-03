"""Rendering helpers: ABC -> MIDI (abc2midi) and MIDI -> audio (FluidSynth).

Audio rendering uses FluidSynth + a SoundFont, then lame for MP3. If any tool is
unavailable the pipeline degrades gracefully: scores still render (Verovio/abcjs
engrave client-side), only the pre-baked audio file is skipped. Code-gen MIDI is
produced by the sandbox (derived from the MusicXML); this module only turns
existing MIDI into MP3 and ABC text into MIDI.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .config import find_soundfont

log = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _warn_once(msg: str) -> None:
    """Warn once per distinct message — batch runs render hundreds of pieces and
    a missing tool would otherwise flood the log with the same line."""
    log.warning(msg)


def _tail(text, limit: int = 500) -> str:
    """Last `limit` chars of subprocess output (str or bytes), for error messages."""
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return text.strip()[-limit:]


# General MIDI program by instrument-name keyword (mirrors the frontend so the
# pre-baked audio and the abcjs notation agree on instruments).
_GM_BY_NAME = [
    ("contrabass", 43), ("double bass", 43), ("violoncello", 42), ("cello", 42),
    ("viola", 41), ("violin", 40), ("harp", 46), ("piccolo", 72), ("flute", 73),
    ("oboe", 68), ("clarinet", 71), ("bassoon", 70), ("trumpet", 56),
    ("trombone", 57), ("tuba", 58), ("horn", 60), ("timpani", 47),
    ("guitar", 24), ("organ", 19), ("harpsichord", 6), ("sax", 65),
    ("piano", 0), ("keyboard", 0),
    ("soprano", 52), ("alto", 52), ("tenor", 52), ("bass", 52),
    ("choir", 52), ("voice", 52), ("vocal", 52),
]


def _gm_program(name: str):
    n = name.lower()
    for kw, prog in _GM_BY_NAME:
        if kw in n:
            return prog
    return None


def _prepare_abc_for_audio(abc: str, gchords: bool = True) -> str:
    """Normalize ABC so abc2midi renders it faithfully: fix bare [V1] -> [V:V1]
    voice markers (else abc2midi reads a chord), and inject a %%MIDI program per
    named voice when the model gave none (else everything defaults to piano).

    `gchords`: abc2midi plays "Em"/"G" chord symbols as strummed accompaniment by
    default. The baked audio keeps that ON (deliberate: the symbols are part of
    the notation, so a listener hears them performed — and the whole corpus was
    synthesized this way). Pass gchords=False to add %%MIDI gchordoff, so feature
    extraction measures only the notes the model actually wrote."""
    import re

    # abc2midi treats a blank line as end-of-tune, so a blank line between the
    # header/voice declarations and the music (or between sections — several models
    # do this) silently truncates the tune to ZERO notes. Our pieces are single
    # tunes, so drop blank lines first.
    lines = [ln for ln in abc.splitlines() if ln.strip()]
    if not gchords:
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("K:"):
                lines.insert(i + 1, "%%MIDI gchordoff")
                break
    abc = "\n".join(lines)

    voices = list(dict.fromkeys(re.findall(r"(?m)^\s*V:\s*(\S+)", abc)))
    for v in voices:
        abc = re.sub(r"\[" + re.escape(v) + r"\]", "[V:" + v + "]", abc)

    if "%%MIDI program" not in abc:
        out = []
        for line in abc.split("\n"):
            out.append(line)
            m = re.match(r"^\s*V:\s*\S+.*\bname=(?:\"([^\"]+)\"|(\S+))", line)
            if m:
                prog = _gm_program(m.group(1) or m.group(2))
                if prog is not None:
                    out.append(f"%%MIDI program {prog}")
        abc = "\n".join(out)
    return abc


def abc_to_midi(abc: str, work_dir: Path, gchords: bool = True) -> "Path | None":
    """Convert ABC text to MIDI with abc2midi (the reference ABC->MIDI tool).
    Returns the MIDI path, or None if abc2midi is unavailable or fails.
    See _prepare_abc_for_audio for the `gchords` switch."""
    abc2midi = shutil.which("abc2midi")
    if not abc2midi:
        _warn_once("abc2midi not found — ABC audio baking skipped "
                   "(install with `brew install abcmidi`)")
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    abc_file = work_dir / "piece.abc"
    abc_file.write_text(_prepare_abc_for_audio(abc, gchords=gchords), encoding="utf-8")
    midi_path = work_dir / "piece.mid"
    try:
        proc = subprocess.run([abc2midi, str(abc_file), "-o", str(midi_path)],
                              capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        log.warning("abc2midi timed out after 60s (%s): %s",
                    abc_file, _tail(e.stderr) or _tail(e.stdout))
        return None
    if proc.returncode != 0 or not midi_path.exists():
        # abc2midi exits non-zero even on recoverable warnings, so keep the MIDI
        # if it was produced — but surface what it complained about either way.
        detail = _tail(proc.stderr) or _tail(proc.stdout)
        log.warning("abc2midi %s (exit %d): %s",
                    "failed" if not midi_path.exists() else "warned",
                    proc.returncode, detail or "no output")
    return midi_path if midi_path.exists() else None


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
        missing = [name for name, ok in
                   (("fluidsynth", fluidsynth), ("a SoundFont", soundfont), ("lame", lame))
                   if not ok]
        _warn_once(f"audio baking skipped — missing {', '.join(missing)} "
                   "(see scripts/setup_soundfont.sh)")
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
                log.warning("fluidsynth failed on %s (exit %d): %s", midi_path,
                            r1.returncode, _tail(r1.stderr) or _tail(r1.stdout))
                return False
            r2 = subprocess.run(
                [lame, "--quiet", "-V", "4", wav, str(audio_path)],
                capture_output=True, text=True, timeout=timeout,
            )
            if r2.returncode != 0:
                log.warning("lame failed on %s (exit %d): %s", midi_path,
                            r2.returncode, _tail(r2.stderr) or _tail(r2.stdout))
        except subprocess.TimeoutExpired as e:
            log.warning("audio render timed out after %ds on %s: %s", timeout,
                        midi_path, _tail(e.stderr) or _tail(e.stdout))
            return False
    return audio_path.exists()
