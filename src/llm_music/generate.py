"""Orchestration: prompt a model, run the chosen mode, render, with retries."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import PROMPTS_DIR
from .models.base import LLMClient
from .modes import MODES
from .render import midi_to_audio
from .retry import backoff_sleep, is_retryable

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a composer expressing yourself through music. Follow the output "
    "format exactly."
)


@dataclass
class PieceResult:
    ok: bool
    model: str
    prompt: str
    mode: str
    prompt_label: str = ""
    prompt_text: str = ""
    system_prompt: str = ""
    title: str = ""
    short_description: str = ""
    long_description: str = ""
    attempts: int = 0
    midi_path: Path | None = None
    musicxml_path: Path | None = None
    audio_path: Path | None = None
    abc: str = ""
    code: str = ""  # raw music21 code (codegen modes; last attempt's if all failed)
    error: str | None = None
    errors: list[str] = field(default_factory=list)


def _form_row(prompt_name: str) -> dict:
    """Look up a prompt's row (id, label, instruction) from sara's CSV."""
    path = PROMPTS_DIR / "form_instructions.csv"
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["id"] == prompt_name:
                return row
    with path.open(encoding="utf-8", newline="") as f:
        known = ", ".join(p["id"] for p in csv.DictReader(f))
    raise KeyError(f"unknown prompt '{prompt_name}'. Known: {known}")


def prompt_label(prompt_name: str) -> str:
    """Human-readable label for a prompt id (e.g. 'free-form' -> 'Free form')."""
    return _form_row(prompt_name).get("label") or prompt_name


def _load_prompt(prompt_name: str, mode_mod) -> str:
    """Assemble the full prompt: sara's prompt.md frame + form instruction + the
    mode's Outputs section (plus the music21 toolkit doc for codegen)."""
    template = (PROMPTS_DIR / "prompt.md").read_text(encoding="utf-8")
    mode_block = mode_mod.OUTPUTS.strip()
    if getattr(mode_mod, "USES_TOOLKIT", False):
        toolkit = (PROMPTS_DIR / "toolkit.md").read_text(encoding="utf-8").strip()
        mode_block += "\n\n# Music documentation\n\n" + toolkit
    return template.format(
        form_instruction=_form_row(prompt_name)["instruction"], mode_block=mode_block
    )


def generate_piece(
    client: LLMClient,
    prompt_name: str,
    mode: str,
    work_dir: Path,
    max_attempts: int = 5,
    bake_audio: bool = True,
) -> PieceResult:
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'. Known: {', '.join(MODES)}")
    mode_mod = MODES[mode]
    base_user = _load_prompt(prompt_name, mode_mod)

    result = PieceResult(
        ok=False,
        model=client.name,
        prompt=prompt_name,
        mode=mode,
        prompt_label=prompt_label(prompt_name),
        prompt_text=base_user,
        system_prompt=SYSTEM_PROMPT,
    )
    prior_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        user = mode_mod.build_user_prompt(base_user, prior_error)
        try:
            response = client.complete(SYSTEM_PROMPT, user)
        except Exception as e:  # API/network failure
            prior_error = f"API error: {e}"
            result.errors.append(prior_error)
            if not is_retryable(e):
                log.warning("%s × %s: permanent API error, giving up: %s",
                            client.name, prompt_name, e)
                break  # e.g. 400 unknown/unverified model, bad key — retrying won't help
            log.warning("%s × %s: attempt %d/%d failed (%s), backing off",
                        client.name, prompt_name, attempt, max_attempts, e)
            backoff_sleep(attempt)  # exponential backoff (esp. for overload)
            continue

        outcome = mode_mod.generate(response, work_dir)
        if getattr(outcome, "code", ""):
            result.code = outcome.code  # keep the latest attempt's code, pass or fail
        if outcome.ok:
            result.ok = True
            result.title = outcome.title
            result.short_description = outcome.short_description
            result.long_description = outcome.long_description
            result.midi_path = outcome.midi_path
            result.musicxml_path = outcome.musicxml_path
            result.abc = outcome.abc
            break
        prior_error = outcome.error
        result.errors.append(prior_error or "unknown error")

    if not result.ok:
        result.error = result.errors[-1] if result.errors else "generation failed"
        return result

    # Pre-render audio for code-gen (MIDI -> FluidSynth). ABC pieces carry no MIDI:
    # abcjs engraves and plays the raw ABC client-side.
    if bake_audio and result.midi_path:
        audio_path = work_dir / "piece.mp3"
        if midi_to_audio(result.midi_path, audio_path):
            result.audio_path = audio_path
    return result
