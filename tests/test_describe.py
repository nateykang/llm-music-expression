import json

from llm_music.describe import (
    SYSTEM_PROMPT,
    build_description_prompt,
    describe_music,
    install_description,
    music_from_entry,
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


def test_music_from_entry_prefers_abc_then_code_then_musicxml(tmp_path):
    assert music_from_entry({"abc": "X:1\nK:C\nC4", "code": "score = 1"}, tmp_path) == (
        "X:1\nK:C\nC4", "abc"
    )
    assert music_from_entry({"code": "score = stream.Score()"}, tmp_path) == (
        "score = stream.Score()", "music21"
    )
    score = tmp_path / "piece.musicxml"
    score.write_text("<score-partwise/>", encoding="utf-8")
    assert music_from_entry({"score": "piece.musicxml"}, tmp_path) == (
        "<score-partwise/>", "musicxml"
    )


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
