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
    """Write a music21 Score to MIDI and MusicXML (used by ABC mode)."""
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    musicxml_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(midi_path))
    score.write("musicxml", fp=str(musicxml_path))


def audio_available() -> bool:
    return shutil.which("fluidsynth") is not None and find_soundfont() is not None


def midi_to_audio(midi_path: Path, audio_path: Path, timeout: int = 120) -> bool:
    """Render MIDI to an audio file via FluidSynth. Returns False if skipped."""
    fluidsynth = shutil.which("fluidsynth")
    soundfont = find_soundfont()
    if not fluidsynth or not soundfont:
        return False

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    # FluidSynth picks output format from the extension (.ogg / .wav).
    cmd = [
        fluidsynth,
        "-ni",
        "-F",
        str(audio_path),
        "-r",
        "44100",
        str(soundfont),
        str(midi_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0 and audio_path.exists()
