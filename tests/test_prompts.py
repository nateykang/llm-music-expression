"""Prompt assembly: system-side self-expression variant + fixed per-mode user frame."""

import csv

import pytest

from llm_music.config import PROMPTS_DIR
from llm_music.generate import SYSTEM_TEMPLATE, _load_prompt, _variant_row, prompt_label
from llm_music.modes import MODES


def _all_ids() -> list[str]:
    with (PROMPTS_DIR / "variants.csv").open(encoding="utf-8", newline="") as f:
        return [row["id"] for row in csv.DictReader(f)]


def test_all_five_variants_present_in_order():
    assert _all_ids() == [
        "express-yourself", "inner-state", "personality",
        "uniquely-you", "emotional-state",
    ]


@pytest.mark.parametrize("prompt_id", _all_ids())
def test_variant_fills_system_prompt(prompt_id):
    instruction = _variant_row(prompt_id)["instruction"]
    system = SYSTEM_TEMPLATE.format(variant=instruction)
    assert system == f"You are a composer. {instruction}"
    assert instruction.startswith("Write music that")


@pytest.mark.parametrize("mode_name", list(MODES))
def test_user_frame_assembles_in_every_mode(mode_name):
    text = _load_prompt(MODES[mode_name])
    assert text.startswith("Your task is to write music in ")
    assert "at least 1 minute in duration" in text
    assert "## Outputs" in text
    # The self-expression variant lives in the system prompt, never here.
    for prompt_id in _all_ids():
        assert _variant_row(prompt_id)["instruction"] not in text


def test_mode_task_phrases():
    assert "write music in Python using the music21 library" in _load_prompt(MODES["codegen"])
    assert "write music in ABC notation" in _load_prompt(MODES["abc"])


def test_no_toolkit_in_any_mode():
    for mode_name in MODES:
        assert "Music Composition Toolkit" not in _load_prompt(MODES[mode_name])


def test_codegen_code_contract_present():
    cg = _load_prompt(MODES["codegen"])
    assert "bind it to a top-level variable named `score`" in cg
    assert "render_audio" not in cg


def test_prompt_label_maps_id_to_human_label():
    assert prompt_label("express-yourself") == "Express yourself"
    assert prompt_label("emotional-state") == "Emotional state"
    # Legacy form-instruction ids keep resolving for pre-redesign batches.
    assert prompt_label("free-form") == "Free form"
    assert prompt_label("string-quartet") == "String quartet"


def test_unknown_prompt_raises():
    with pytest.raises(KeyError):
        _variant_row("does-not-exist")
    with pytest.raises(KeyError):
        prompt_label("does-not-exist")
