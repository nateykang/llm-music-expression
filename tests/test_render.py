"""Regression tests for the music21 ABC export workarounds in render.py."""

from pathlib import Path

import pytest

from llm_music.render import write_score

FIXTURES = Path(__file__).parent / "fixtures"


def _parse(abc_text: str):
    from music21 import converter

    return converter.parse(abc_text, format="abc")


def test_write_score_recovers_repeat_export_failure(tmp_path):
    """A real gpt-4.1 ABC with repeats that breaks music21's direct MIDI export.

    Direct ``score.write('midi')`` raises "cannot process repeats on Stream that
    does not contain measures"; write_score must still produce both files.
    """
    abc_text = (FIXTURES / "abc_repeat_fail.abc").read_text()
    score = _parse(abc_text)

    midi = tmp_path / "p.mid"
    xml = tmp_path / "p.musicxml"
    write_score(score, midi, xml)

    assert midi.exists() and midi.stat().st_size > 0
    assert xml.exists()
    assert "<score-partwise" in xml.read_text(encoding="utf-8")


def test_write_score_handles_plain_abc(tmp_path):
    score = _parse("X:1\nT:Plain\nM:4/4\nL:1/4\nK:C\nC D E F | G2 E2 |]\n")
    midi = tmp_path / "p.mid"
    xml = tmp_path / "p.musicxml"
    write_score(score, midi, xml)
    assert midi.exists() and xml.exists()
