"""Prompt assembly: sara's prompt.md frame + form_instructions.csv + toolkit.md."""

import csv

import pytest

from llm_music.config import PROMPTS_DIR
from llm_music.generate import _form_row, _load_prompt, prompt_label
from llm_music.modes import MODES


def _all_ids() -> list[str]:
    with (PROMPTS_DIR / "form_instructions.csv").open(encoding="utf-8", newline="") as f:
        return [row["id"] for row in csv.DictReader(f)]


def test_all_eleven_prompts_present():
    ids = _all_ids()
    assert len(ids) == 11
    assert "free-form" in ids and "postmodern" in ids


@pytest.mark.parametrize("prompt_id", _all_ids())
@pytest.mark.parametrize("mode_name", list(MODES))
def test_every_prompt_assembles_in_every_mode(prompt_id, mode_name):
    text = _load_prompt(prompt_id, MODES[mode_name])
    # Frame + form instruction + self-expression closing are always present.
    assert "expressing yourself" in text
    assert _form_row(prompt_id)["instruction"] in text
    assert "self-expression" in text.lower()


def test_codegen_injects_toolkit_but_abc_does_not():
    cg = _load_prompt("free-form", MODES["codegen"])
    ab = _load_prompt("free-form", MODES["abc"])
    assert "Music Composition Toolkit" in cg
    assert "bind it to a top-level variable named `score`" in cg
    assert "Music Composition Toolkit" not in ab
    # The adapted toolkit must not reference sara's render_audio contract.
    assert "render_audio" not in cg


def test_constraint_text_lands_in_constrained_prompts():
    assert "three-part fugue" in _load_prompt("fugue", MODES["abc"])
    assert "string quartet" in _load_prompt("string-quartet", MODES["abc"])


def test_prompt_label_maps_id_to_human_label():
    assert prompt_label("free-form") == "Free form"
    assert prompt_label("string-quartet") == "String quartet"
    assert prompt_label("stab-voicing") == "STAB voicing"


def test_unknown_prompt_raises():
    with pytest.raises(KeyError):
        _load_prompt("does-not-exist", MODES["abc"])
