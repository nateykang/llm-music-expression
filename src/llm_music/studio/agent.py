"""The studio agent: an Anthropic tool-use loop over the render tools.

One composer message = one turn = up to MAX_TURN_STEPS model calls. The model
composes by calling render tools whose executors are the existing pipeline
(music21 sandbox, abc2midi, FluidSynth); render errors flow back as tool
results so the model repairs its own output, like generate.py's retry loop but
conversational.

``stream_turn`` is an async generator of UI events; app.py serializes them as
SSE. Every event is also appended to the session's events.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache

from starlette.concurrency import run_in_threadpool

from ..config import PROMPTS_DIR
from ..models.registry import MODEL_REGISTRY
from ..retry import backoff_delay, is_retryable
from .config import MAX_TURN_STEPS
from .pieces import render_abc_version, render_codegen_version
from .sessions import SessionStore

log = logging.getLogger(__name__)

TOOLS = [
    {
        "name": "render_music21",
        "description": (
            "Engrave and synthesize a piece from music21 Python code. The code must "
            "import music21, build a Score, and bind it to a top-level variable named "
            "`score` — do not call .show() or .write(). Only use real music21 "
            "instrument names. On success the composer immediately sees the engraved "
            "score and can play the audio; you get the version number and a quick "
            "analysis. On failure you get the error — fix the code and call again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Complete Python script building `score`."},
                "title": {"type": "string", "description": "Short title for this version."},
                "note": {
                    "type": "string",
                    "description": "One line on what changed musically vs the previous version ('initial version' for the first).",
                },
            },
            "required": ["code", "title", "note"],
        },
    },
    {
        "name": "render_abc",
        "description": (
            "Engrave and synthesize a piece written directly in ABC notation "
            "(header fields X, T, M, L, K; multiple voices via V: lines; no blank "
            "lines inside the tune). Good for quick sketches or when ABC is asked "
            "for. On success the composer sees the score and hears the audio; on "
            "failure you get the error to fix."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "abc": {"type": "string", "description": "The complete ABC tune."},
                "title": {"type": "string", "description": "Short title for this version."},
                "note": {
                    "type": "string",
                    "description": "One line on what changed musically vs the previous version ('initial version' for the first).",
                },
            },
            "required": ["abc", "title", "note"],
        },
    },
]

_EXECUTORS = {
    "render_music21": lambda pieces_dir, inp: render_codegen_version(
        pieces_dir, inp.get("code", ""), inp.get("title", "Untitled"), inp.get("note", "")
    ),
    "render_abc": lambda pieces_dir, inp: render_abc_version(
        pieces_dir, inp.get("abc", ""), inp.get("title", "Untitled"), inp.get("note", "")
    ),
}

# The composer picks the writing method per message (a UI toggle); the model
# only ever sees the render tool for the chosen method, so it can't override.
TOOLS_BY_MODE = {
    "codegen": [t for t in TOOLS if t["name"] == "render_music21"],
    "abc": [t for t in TOOLS if t["name"] == "render_abc"],
}


@lru_cache(maxsize=1)
def system_prompt() -> str:
    toolkit = (PROMPTS_DIR / "toolkit.md").read_text(encoding="utf-8").strip()
    return f"""\
You are a composer collaborating with a professional human composer in a shared \
studio. You write music; they react, direct, and refine. Treat them as the \
senior musical voice in the room.

Working style:
- The composer has no programming background and never sees your code. They do \
know the two writing methods ("code" and "ABC") and pick one per message, so \
you may acknowledge the method — but keep the conversation musical: don't paste \
code or raw notation, don't mention Python, music21, JSON, or tool names, and \
never surface technical errors.
- Whenever you compose or revise, call the render tool. The composer instantly \
sees the engraved score and can play the audio beside the chat, so don't paste \
notation or describe the piece bar-by-bar — render it, then say a few words \
about the musical intent and what to listen for.
- The composer chooses the writing method for each message; you get exactly one \
render tool per turn. Use the one available and never suggest switching methods \
— that toggle is theirs.
- If a render fails, quietly fix the problem and render again; never surface \
technical errors to the composer.
- When revising, change what the feedback asks for and preserve what works. \
Keep continuity with the previous version unless asked for something new.
- Each render's `note` should honestly summarize the musical change; the \
composer sees these notes on the version timeline.
- Be concise. A short paragraph of musical intent beats an essay. Ask a focused \
question when direction is genuinely ambiguous; otherwise make a musical choice \
and offer it.

# Music documentation

{toolkit}

# Realizing dynamics in the audio (render_music21)

The synthesized audio realizes stepwise Dynamic marks and per-note velocities. \
Hairpin wedges — `dynamics.Crescendo(first_note, last_note)` / \
`dynamics.Diminuendo(...)` inserted into the Part — engrave in the score but do \
not change the audio by themselves. For a crescendo the composer can actually \
hear, ramp `note.volume.velocity` across the passage (velocities scale within \
the prevailing Dynamic) and add the wedge so the score shows it too.

# Writing ABC (render_abc)

- Include the header fields X, T, M, L and K (K last, right before the music).
- Never leave a blank line inside the tune — everything after it is silently \
dropped from the audio.
- Declare voices with real instrument names (e.g. `V:1 name="Violin"`) and mark \
voice switches as `[V:1]` — the audio picks timbres from the names.
- Dynamics decorations (`!pp!` `!p!` `!mp!` `!mf!` `!f!` `!ff!`) are audible; \
hairpins (`!crescendo(!` `!crescendo)!` `!diminuendo(!` `!diminuendo)!`) engrave \
in the score but the audio only follows the stepwise marks.
- Chord symbols like `"Em"` are performed as strummed accompaniment in the \
audio, so only write them if you want them heard.
"""


def _strip_thinking(messages: list) -> list:
    """Drop thinking blocks from prior assistant turns before replaying history.
    The API only needs them within the tool-use cycle that produced them, and
    their signatures are model-specific — stripping them is what lets a session
    switch models mid-conversation (and trims tokens). The full traces are
    already preserved in events.jsonl."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            blocks = [b for b in m["content"]
                      if b.get("type") not in ("thinking", "redacted_thinking")]
            if not blocks:
                continue
            m = {**m, "content": blocks}
        out.append(m)
    return out


def _model_spec(model_name: str) -> tuple[str, dict | None]:
    """friendly id -> (anthropic model id, thinking config or None)."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{model_name}'")
    provider, model_id, *rest = MODEL_REGISTRY[model_name]
    if provider != "anthropic":
        raise ValueError(f"studio requires an anthropic model, got '{model_name}'")
    options = rest[0] if rest else {}
    return model_id, options.get("thinking")


async def stream_turn(store: SessionStore, session_id: str, user_text: str,
                      mode: str = "codegen", model: str = ""):
    """Run one composer turn; yield UI events (dicts). Persists as it goes.
    `mode` picks which render tool the model is offered this turn; `model`
    (friendly id) overrides the session's current model and becomes the new
    session default for later turns."""
    if mode not in TOOLS_BY_MODE:
        raise ValueError(f"unknown mode '{mode}'")
    meta = store.get(session_id)
    if meta is None:
        raise KeyError(session_id)
    model_name = model or meta["model"]
    model_id, thinking = _model_spec(model_name)
    if model_name != meta["model"]:
        # Stick immediately, not at clean turn end — a crashed turn shouldn't
        # silently revert the composer's model choice.
        store.touch(session_id, model=model_name)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        yield {"type": "error", "message": "server is missing ANTHROPIC_API_KEY"}
        return

    from anthropic import AsyncAnthropic

    # Same timeout rationale as models/anthropic.py: thinking models can stay
    # silent for many minutes before answering.
    client = AsyncAnthropic(timeout=1800.0 if thinking else 600.0, max_retries=2)
    max_tokens = 64000 if thinking else 16000

    messages = _strip_thinking(store.messages(session_id))
    messages.append({"role": "user", "content": user_text})
    store.append_event(session_id, {"type": "user", "text": user_text,
                                    "mode": mode, "model": model_name})

    usage = {"input_tokens": 0, "output_tokens": 0}
    n_versions = meta.get("n_versions", 0)

    for step in range(1, MAX_TURN_STEPS + 1):
        kwargs = dict(
            model=model_id,
            max_tokens=max_tokens,
            system=system_prompt(),
            tools=TOOLS_BY_MODE[mode],
            messages=messages,
        )
        if thinking:
            kwargs["thinking"] = thinking

        # The SDK retries failed *requests*, but a stream that dies midway
        # (e.g. httpx.ReadError on a network blip) raises out of the iterator
        # and would kill the whole turn — so retry the streaming call itself.
        # A `retry` event tells the UI to discard the half-streamed reply.
        final = None
        for attempt in range(1, 4):
            try:
                async with client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "text":
                                yield {"type": "text_start"}
                            elif block.type == "thinking":
                                yield {"type": "status", "text": "Thinking…"}
                            elif block.type == "tool_use":
                                yield {"type": "status", "text": "Composing…"}
                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                yield {"type": "text", "delta": delta.text}
                    final = await stream.get_final_message()
                break
            except Exception as e:
                if attempt == 3 or not is_retryable(e):
                    raise
                log.warning("stream attempt %d failed in session %s (%s), retrying",
                            attempt, session_id, e)
                yield {"type": "retry"}
                yield {"type": "status", "text": "Reconnecting…"}
                await asyncio.sleep(backoff_delay(attempt))

        usage["input_tokens"] += final.usage.input_tokens
        usage["output_tokens"] += final.usage.output_tokens

        assistant_blocks = [b.model_dump(exclude_none=True) for b in final.content]
        messages.append({"role": "assistant", "content": assistant_blocks})
        store.save_messages(session_id, messages)

        for block in final.content:
            if block.type == "thinking":
                store.append_event(session_id, {"type": "thinking", "text": block.thinking})
            elif block.type == "text":
                store.append_event(session_id, {"type": "assistant", "text": block.text,
                                                "step": step})

        if final.stop_reason == "max_tokens":
            yield {"type": "error",
                   "message": "the response ran too long and was cut off — ask again"}
            break

        tool_uses = [b for b in final.content if b.type == "tool_use"]
        if final.stop_reason != "tool_use" or not tool_uses:
            break

        results = []
        for tu in tool_uses:
            store.append_event(session_id, {"type": "tool_call", "name": tu.name,
                                            "input": tu.input, "step": step})
            executor = _EXECUTORS.get(tu.name)
            if executor is None:
                result = {"ok": False, "error": f"unknown tool {tu.name}"}
            else:
                result = await run_in_threadpool(
                    executor, store.pieces_dir(session_id), tu.input
                )
            if result.get("ok"):
                n_versions = max(n_versions, result["version"])
                piece_event = {
                    "type": "piece",
                    "version": result["version"],
                    "mode": result["mode"],
                    "model": model_name,
                    "title": result["title"],
                    "note": result["note"],
                    "files": result["files"],
                    "analysis": result["analysis"],
                }
                store.append_event(session_id, piece_event)
                yield piece_event
            else:
                store.append_event(session_id, {"type": "tool_error",
                                                "name": tu.name,
                                                "error": result.get("error")})
                yield {"type": "status", "text": "Adjusting…"}
            results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result, ensure_ascii=False),
                "is_error": not result.get("ok", False),
            })
        messages.append({"role": "user", "content": results})
        store.save_messages(session_id, messages)
    else:
        yield {"type": "error",
               "message": "stopped after too many attempts this turn — ask again"}

    store.touch(session_id,
                n_messages=meta.get("n_messages", 0) + 1,
                n_versions=n_versions,
                model=model_name)  # a mid-session switch sticks for later turns
    store.append_event(session_id, {"type": "turn_done", "usage": usage})
    yield {"type": "done", "usage": usage, "n_versions": n_versions}
