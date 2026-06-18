import json

from llm_music.modes import abc

SAMPLE_ABC = """X:1
T:Test Tune
M:4/4
L:1/4
K:C
C D E F | G A B c |
"""


def _response(abc_text: str) -> str:
    return json.dumps(
        {
            "abc": abc_text,
            "title": "Test Tune",
            "short_description": "a scale",
            "long_description": "an ascending C major scale across two bars",
        }
    )


def test_abc_mode_keeps_raw_abc(tmp_path):
    # ABC mode stores the raw ABC verbatim (abcjs renders it client-side); it no
    # longer routes through music21, so there is no MIDI/MusicXML artifact.
    result = abc.generate(_response(SAMPLE_ABC), tmp_path)
    assert result.ok, result.error
    assert result.title == "Test Tune"
    assert "K:C" in result.abc and "C D E F" in result.abc
    assert result.midi_path is None
    assert result.musicxml_path is None


def test_abc_mode_rejects_non_abc_body(tmp_path):
    # Missing X:/K: headers -> coarse syntax gate fails -> triggers a retry.
    bad = json.dumps({"abc": "just some prose, not notation", "title": "x"})
    result = abc.generate(bad, tmp_path)
    assert not result.ok
    assert result.error


def test_abc_mode_handles_fenced_json(tmp_path):
    fenced = "```json\n" + _response(SAMPLE_ABC) + "\n```"
    result = abc.generate(fenced, tmp_path)
    assert result.ok, result.error


def test_abc_mode_reports_bad_json(tmp_path):
    result = abc.generate("not json at all", tmp_path)
    assert not result.ok
    assert result.error


def test_abc_mode_reports_missing_abc(tmp_path):
    result = abc.generate(json.dumps({"title": "x"}), tmp_path)
    assert not result.ok
