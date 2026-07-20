"""Describe an existing symbolic score with a fresh, stateless model call.

The prompt deliberately mirrors the description fields in the composition
prompt. It does not ask the model to be neutral, critical, or evidence-bound:
the intervention is only that the music is supplied without the first call's
descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models.base import LLMClient
from .modes._common import extract_json
from .retry import backoff_sleep, is_retryable


SYSTEM_PROMPT = (
    "You are a composer expressing yourself through music. Follow the output "
    "format exactly."
)


@dataclass
class DescriptionResult:
    ok: bool
    short_description: str = ""
    long_description: str = ""
    attempts: int = 0
    error: str | None = None


def build_description_prompt(music: str, representation: str) -> str:
    """Build the exact second-pass prompt used by generation and replay."""
    labels = {
        "abc": ("ABC notation", "abc"),
        "music21": ("music21 Python representation", "python"),
        "musicxml": ("MusicXML representation", "xml"),
    }
    if representation not in labels:
        raise ValueError(f"unsupported music representation: {representation}")
    label, fence = labels[representation]
    return (
        "Your task is to describe the music provided below.\n\n"
        "## Outputs\n\n"
        "You must respond with a single JSON object (and nothing else) with these fields:\n\n"
        "- `short_description`: A single sentence describing the musical intent.\n"
        "- `long_description`: A detailed explanation of the compositional choices. "
        "Can be any length.\n\n"
        f"Here is the complete piece in {label}:\n\n"
        f"```{fence}\n{music}\n```"
    )


def describe_music(
    client: LLMClient,
    music: str,
    representation: str,
    max_attempts: int = 3,
) -> DescriptionResult:
    """Generate descriptions from music alone, retrying failed samples."""
    prompt = build_description_prompt(music, representation)
    last_error = "description generation failed"
    for attempt in range(1, max_attempts + 1):
        try:
            raw = client.complete(SYSTEM_PROMPT, prompt, json_mode=True)
            obj = extract_json(raw)
            short = obj.get("short_description")
            long = obj.get("long_description")
            if not isinstance(short, str) or not short.strip():
                raise ValueError("response missing non-empty 'short_description'")
            if not isinstance(long, str) or not long.strip():
                raise ValueError("response missing non-empty 'long_description'")
            return DescriptionResult(
                ok=True,
                short_description=short.strip(),
                long_description=long.strip(),
                attempts=attempt,
            )
        except Exception as exc:
            last_error = str(exc)
            # A malformed response can improve on a fresh sample. Permanent API
            # failures cannot and should return immediately.
            if not isinstance(exc, ValueError) and not is_retryable(exc):
                return DescriptionResult(ok=False, attempts=attempt, error=last_error)
            if attempt < max_attempts:
                backoff_sleep(attempt - 1, cap=8.0)
    return DescriptionResult(ok=False, attempts=max_attempts, error=last_error)


def music_from_entry(entry: dict, batch: Path) -> tuple[str, str]:
    """Resolve the best symbolic artifact available in a manifest entry."""
    if isinstance(entry.get("abc"), str) and entry["abc"].strip():
        return entry["abc"], "abc"
    if isinstance(entry.get("code"), str) and entry["code"].strip():
        return entry["code"], "music21"
    score = entry.get("score")
    if score:
        path = batch / score
        if path.is_file():
            return path.read_text(encoding="utf-8"), "musicxml"
    raise ValueError("piece has no stored ABC, music21 code, or MusicXML score")


def description_metadata(model: str, representation: str, attempts: int) -> dict:
    return {
        "model": model,
        "representation": representation,
        "attempts": attempts,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_template": build_description_prompt("<MUSIC ARTIFACT>", representation),
        "method": "music-only fresh call",
    }


def install_description(entry: dict, result: DescriptionResult, model: str,
                        representation: str) -> None:
    """Replace canonical descriptions while retaining the composing call's text."""
    entry.setdefault("original_short_description", entry.get("short_description", ""))
    entry.setdefault("original_long_description", entry.get("long_description", ""))
    entry["short_description"] = result.short_description
    entry["long_description"] = result.long_description
    entry["independent_description"] = description_metadata(
        model, representation, result.attempts
    )
    entry.pop("independent_description_error", None)
