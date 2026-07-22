"""Describe an existing symbolic score with a fresh, stateless model call.

The prompt deliberately mirrors the description fields in the composition
prompt. It does not ask the model to be neutral, critical, or evidence-bound:
the intervention is only that the music is supplied without the first call's
descriptions. To keep that true, composer-authored scaffolding text is
scrubbed from the artifact before it is shown: titles, comment lines, and
credits all quote the first call's framing. Text that is part of the score
itself (lyrics, performance directions) is kept.
"""

from __future__ import annotations

import io
import re
import tokenize
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


def scrub_abc(abc: str) -> str:
    """Drop %-comments and the composer's title (T: is optional in ABC)."""
    out = []
    for line in abc.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith("T:"):
            continue
        if "%" in line:
            line = line.split("%", 1)[0].rstrip()
            if not line:
                continue
        out.append(line)
    return "\n".join(out)


_TITLE_ASSIGNMENT = re.compile(
    r"((?:title|movementName)\s*=\s*)(['\"])(?:[^'\"\\]|\\.)*\2"
)


def scrub_music21_code(code: str) -> str:
    """Strip # comments and blank title strings.

    Tokenizing rather than splitting on '#' keeps sharps inside string
    literals ("C#4") intact. Generated code has already executed, so a
    tokenizer failure is unexpected; if it happens anyway, only whole-line
    comments are dropped.
    """
    lines = code.splitlines()
    try:
        comment_col = {}
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                row, col = tok.start
                comment_col[row] = min(col, comment_col.get(row, col))
        out = []
        for row, line in enumerate(lines, start=1):
            if row in comment_col:
                line = line[: comment_col[row]].rstrip()
                if not line:
                    continue
            out.append(line)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        out = [l for l in lines if not l.lstrip().startswith("#")]
    return _TITLE_ASSIGNMENT.sub(r"\1\2\2", "\n".join(out))


def scrub_musicxml(xml: str) -> str:
    """Drop title elements and creator/credit/rights metadata entirely."""
    for tag in ("work-title", "movement-title", "creator", "credit", "rights"):
        xml = re.sub(rf"[ \t]*<{tag}\b[^>]*>.*?</{tag}>\n?", "", xml, flags=re.S)
        xml = re.sub(rf"[ \t]*<{tag}\b[^>]*/>\n?", "", xml)
    xml = re.sub(r"[ \t]*<work>\s*</work>\n?", "", xml)
    return xml


def scrubbed_music(abc: str | None, musicxml: str | None,
                   code: str | None) -> tuple[str, str]:
    """Pick the best artifact and scrub composer-authored scaffolding text.

    MusicXML outranks raw code: it is what the code deterministically
    produced, and it carries no comments to leak the composing call's
    framing.
    """
    if abc and abc.strip():
        return scrub_abc(abc), "abc"
    if musicxml and musicxml.strip():
        return scrub_musicxml(musicxml), "musicxml"
    if code and code.strip():
        return scrub_music21_code(code), "music21"
    raise ValueError("piece has no stored ABC, MusicXML score, or music21 code")


def music_from_entry(entry: dict, batch: Path) -> tuple[str, str]:
    """Resolve the best symbolic artifact available in a manifest entry."""
    xml = None
    score = entry.get("score")
    if score and (batch / score).is_file():
        xml = (batch / score).read_text(encoding="utf-8")
    abc = entry.get("abc") if isinstance(entry.get("abc"), str) else None
    code = entry.get("code") if isinstance(entry.get("code"), str) else None
    return scrubbed_music(abc, xml, code)


def description_metadata(model: str, representation: str, attempts: int) -> dict:
    return {
        "model": model,
        "representation": representation,
        "attempts": attempts,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_template": build_description_prompt("<MUSIC ARTIFACT>", representation),
        "method": "music-only fresh call (titles/comments/credits scrubbed)",
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
