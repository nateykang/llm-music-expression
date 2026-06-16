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


def test_abc_mode_produces_midi_and_musicxml(tmp_path):
    result = abc.generate(_response(SAMPLE_ABC), tmp_path)
    assert result.ok, result.error
    assert result.title == "Test Tune"
    assert result.midi_path.exists()
    assert result.musicxml_path.exists()
    xml = result.musicxml_path.read_text(encoding="utf-8")
    assert "<score-partwise" in xml


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
