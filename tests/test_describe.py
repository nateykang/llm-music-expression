import json

from llm_music.describe import (
    SYSTEM_PROMPT,
    build_description_prompt,
    describe_music,
    install_description,
    music_from_entry,
    scrub_abc,
    scrub_music21_code,
    scrub_musicxml,
)
from llm_music.generate import generate_piece
from llm_music.modes import MODES
from llm_music.modes._common import ModeResult


class FakeClient:
    name = "test-model"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, system, user, json_mode=False):
        self.calls.append((system, user, json_mode))
        return self.response


class SequenceClient:
    name = "test-model"

    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, system, user, json_mode=False):
        return next(self.responses)


def test_prompt_matches_original_description_fields_without_original_text():
    prompt = build_description_prompt("X:1\nT:Piece\nK:C\nC4", "abc")
    assert "single sentence describing the musical intent" in prompt
    assert "detailed explanation of the compositional choices" in prompt
    assert "neutral" not in prompt.lower()
    assert "evidence" not in prompt.lower()
    assert "X:1\nT:Piece\nK:C\nC4" in prompt


def test_describe_music_parses_fresh_call():
    client = FakeClient(json.dumps({
        "short_description": "A short description.",
        "long_description": "A long description.",
    }))
    result = describe_music(client, "X:1\nK:C\nC4", "abc")
    assert result.ok
    assert result.short_description == "A short description."
    assert client.calls[0][0] == SYSTEM_PROMPT
    assert client.calls[0][2] is True


def test_install_preserves_composer_descriptions():
    entry = {
        "short_description": "Composer short.",
        "long_description": "Composer long.",
    }
    result = describe_music(FakeClient(json.dumps({
        "short_description": "Fresh short.",
        "long_description": "Fresh long.",
    })), "X:1\nK:C\nC4", "abc")
    install_description(entry, result, "test-model", "abc")
    assert entry["original_short_description"] == "Composer short."
    assert entry["original_long_description"] == "Composer long."
    assert entry["short_description"] == "Fresh short."
    assert entry["independent_description"]["model"] == "test-model"


def test_music_from_entry_prefers_abc_then_musicxml_then_code(tmp_path):
    assert music_from_entry({"abc": "X:1\nK:C\nC4", "code": "score = 1"}, tmp_path) == (
        "X:1\nK:C\nC4", "abc"
    )
    assert music_from_entry({"code": "score = stream.Score()"}, tmp_path) == (
        "score = stream.Score()", "music21"
    )
    score = tmp_path / "piece.musicxml"
    score.write_text("<score-partwise/>", encoding="utf-8")
    assert music_from_entry(
        {"score": "piece.musicxml", "code": "score = 1"}, tmp_path
    ) == ("<score-partwise/>", "musicxml")


def test_scrub_abc_removes_title_and_comments():
    abc = (
        "X:1\n"
        "T:Where the Light Settles\n"
        "T:a subtitle of yearning\n"
        "% a quiet opening, full of hope\n"
        "M:3/4\n"
        "K:D\n"
        "DFA d2 A | % the theme blooms\n"
        "w: la la la\n"
    )
    scrubbed = scrub_abc(abc)
    assert scrubbed == (
        "X:1\n"
        "M:3/4\n"
        "K:D\n"
        "DFA d2 A |\n"
        "w: la la la"
    )


def test_scrub_code_keeps_sharps_and_blanks_titles():
    code = (
        "# --- Section A: solitude, a single line searching ---\n"
        "n = note.Note('C#4')  # yearning upward\n"
        "md.title = \"Fragments of Longing\"\n"
        "s.append(n)\n"
    )
    scrubbed = scrub_music21_code(code)
    assert "solitude" not in scrubbed
    assert "yearning" not in scrubbed
    assert "note.Note('C#4')" in scrubbed
    assert 'md.title = ""' in scrubbed
    assert "s.append(n)" in scrubbed


def test_scrub_musicxml_blanks_titles_and_credits():
    xml = (
        "<score-partwise>\n"
        "  <work>\n"
        "    <work-title>Emergent Reflections</work-title>\n"
        "  </work>\n"
        "  <movement-title>Emergent Reflections</movement-title>\n"
        "  <identification>\n"
        '    <creator type="composer">AI Composer</creator>\n'
        '    <creator type="lyricist" />\n'
        "  </identification>\n"
        "  <credit><credit-words>Emergent Reflections</credit-words></credit>\n"
        "  <part-list/>\n"
        "</score-partwise>"
    )
    scrubbed = scrub_musicxml(xml)
    assert "Emergent Reflections" not in scrubbed
    assert "AI Composer" not in scrubbed
    for tag in ("creator", "credit", "work-title", "movement-title", "work"):
        assert f"<{tag}" not in scrubbed
    assert "<part-list/>" in scrubbed
    assert "<identification>" in scrubbed


def test_generation_option_replaces_notes_and_preserves_originals(monkeypatch, tmp_path):
    class FakeMode:
        OUTPUTS = "## Outputs\nFake"
        USES_TOOLKIT = False

        @staticmethod
        def build_user_prompt(base, prior_error):
            return base

        @staticmethod
        def generate(response, work_dir):
            return ModeResult(
                ok=True,
                title="Title",
                short_description="Composer short.",
                long_description="Composer long.",
                abc="X:1\nT:Title\nK:C\nC4",
            )

    monkeypatch.setitem(MODES, "fake-description-mode", FakeMode)
    client = SequenceClient([
        "unused composition response",
        json.dumps({
            "short_description": "Fresh short.",
            "long_description": "Fresh long.",
        }),
    ])
    result = generate_piece(
        client,
        "free-form",
        "fake-description-mode",
        tmp_path,
        bake_audio=False,
        independent_description=True,
    )
    assert result.ok
    assert result.short_description == "Fresh short."
    assert result.original_short_description == "Composer short."
    assert result.independent_description["representation"] == "abc"
