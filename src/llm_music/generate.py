"""Orchestration: prompt a model, run the chosen mode, render, with retries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import PROMPTS_DIR
from .models.base import LLMClient
from .modes import MODES
from .render import midi_to_audio

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
    title: str = ""
    short_description: str = ""
    long_description: str = ""
    attempts: int = 0
    midi_path: Path | None = None
    musicxml_path: Path | None = None
    audio_path: Path | None = None
    error: str | None = None
    errors: list[str] = field(default_factory=list)


def _load_prompt(prompt_name: str, mode_instructions: str) -> str:
    template = (PROMPTS_DIR / "freeform.md").read_text(encoding="utf-8")
    if prompt_name == "freeform":
        constraints = ""
    else:
        cpath = PROMPTS_DIR / "constraints" / f"{prompt_name}.md"
        if not cpath.exists():
            raise FileNotFoundError(f"no constraint prompt '{prompt_name}' at {cpath}")
        constraints = cpath.read_text(encoding="utf-8").strip()
    return template.format(constraints=constraints, mode_instructions=mode_instructions)


def generate_piece(
    client: LLMClient,
    prompt_name: str,
    mode: str,
    work_dir: Path,
    max_attempts: int = 5,
) -> PieceResult:
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'. Known: {', '.join(MODES)}")
    mode_mod = MODES[mode]
    base_user = _load_prompt(prompt_name, mode_mod.INSTRUCTIONS)

    result = PieceResult(ok=False, model=client.name, prompt=prompt_name, mode=mode)
    prior_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        user = mode_mod.build_user_prompt(base_user, prior_error)
        try:
            response = client.complete(SYSTEM_PROMPT, user)
        except Exception as e:  # API/network failure
            prior_error = f"API error: {e}"
            result.errors.append(prior_error)
            continue

        outcome = mode_mod.generate(response, work_dir)
        if outcome.ok:
            result.ok = True
            result.title = outcome.title
            result.short_description = outcome.short_description
            result.long_description = outcome.long_description
            result.midi_path = outcome.midi_path
            result.musicxml_path = outcome.musicxml_path
            break
        prior_error = outcome.error
        result.errors.append(prior_error or "unknown error")

    if not result.ok:
        result.error = result.errors[-1] if result.errors else "generation failed"
        return result

    # Pre-render audio (skipped gracefully if FluidSynth/SoundFont unavailable).
    audio_path = work_dir / "piece.ogg"
    if midi_to_audio(result.midi_path, audio_path):
        result.audio_path = audio_path
    return result
