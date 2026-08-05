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
from .retry import backoff_sleep, is_overloaded, is_rate_limited, is_retryable

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
    independent_description: dict | None = None
    independent_description_error: str | None = None
    attempts: int = 0
    midi_path: Path | None = None
    musicxml_path: Path | None = None
    audio_path: Path | None = None
    abc: str = ""
    code: str = ""  # raw music21 code (codegen modes; last attempt's if all failed)
    error: str | None = None
    errors: list[str] = field(default_factory=list)
    # one {error, code} record per FAILED attempt (code is "" for API/parse
    # failures where no code was produced) — drafts models later revise carry
    # things final code doesn't, e.g. thinking-out-loud comments
    failed_attempts: list[dict] = field(default_factory=list)


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
    independent_description: bool = False,
    description_max_attempts: int = 3,
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

    # 529-class overloads are provider capacity backpressure, not a property of
    # the model, so they never consume one of the model's attempts (attempts is
    # a reliability covariate in analyses). They get their own bounded budget so
    # a sustained outage still terminates; only past that budget do they start
    # charging attempts like any other API error.
    OVERLOAD_RETRIES = 6
    overloads = 0
    attempt = 0
    while attempt < max_attempts:
        user = mode_mod.build_user_prompt(base_user, prior_error)
        try:
            # json_mode enforces at the decoder what the prompt already demands
            # ("a single JSON object and nothing else") — on OpenRouter reasoning
            # models it stops the answer being stranded/truncated in the reasoning
            # trace (same fix the judge path has always used). Prompt text unchanged.
            response = client.complete(SYSTEM_PROMPT, user, json_mode=True)
        except Exception as e:  # API/network failure
            if (is_overloaded(e) or is_rate_limited(e)) and overloads < OVERLOAD_RETRIES:
                # 529 overload and 429 throttle alike: infrastructure signals,
                # never charged against the model's attempts.
                overloads += 1
                result.errors.append(f"infra backpressure (attempt not charged): {e}")
                log.warning("%s × %s: provider backpressure (%d/%d), waiting — "
                            "not counted as an attempt",
                            client.name, prompt_name, overloads, OVERLOAD_RETRIES)
                backoff_sleep(overloads + 2)  # capacity dips need longer waits
                continue
            attempt += 1
            result.attempts = attempt
            prior_error = f"API error: {e}"
            result.errors.append(prior_error)
            result.failed_attempts.append({"error": prior_error, "code": ""})
            if not is_retryable(e):
                log.warning("%s × %s: permanent API error, giving up: %s",
                            client.name, prompt_name, e)
                break  # e.g. 400 unknown/unverified model, bad key — retrying won't help
            log.warning("%s × %s: attempt %d/%d failed (%s), backing off",
                        client.name, prompt_name, attempt, max_attempts, e)
            backoff_sleep(attempt)  # exponential backoff
            continue
        attempt += 1
        result.attempts = attempt

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
        result.failed_attempts.append({"error": prior_error or "unknown error",
                                       "code": getattr(outcome, "code", "") or ""})

    if not result.ok:
        result.error = result.errors[-1] if result.errors else "generation failed"
        return result

    if independent_description:
        from .describe import describe_music, description_record, scrubbed_music

        xml = None
        if result.musicxml_path and result.musicxml_path.is_file():
            xml = result.musicxml_path.read_text(encoding="utf-8")
        music, representation = scrubbed_music(result.abc, xml, result.code)
        described = describe_music(
            client, music, representation, max_attempts=description_max_attempts
        )
        if described.ok:
            result.independent_description = description_record(
                described, client.name, representation
            )
        else:
            result.independent_description_error = described.error

    # Pre-render audio for code-gen (MIDI -> FluidSynth). ABC pieces carry no MIDI:
    # abcjs engraves and plays the raw ABC client-side.
    if bake_audio and result.midi_path:
        audio_path = work_dir / "piece.mp3"
        if midi_to_audio(result.midi_path, audio_path):
            result.audio_path = audio_path
    return result
