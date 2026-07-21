"""Provider backends for the studio agent loop.

The studio stores every session transcript in Anthropic message-block format
(the canonical form); each backend translates canonical <-> provider wire
format per call. That is what lets one conversation move between Claude, GPT,
and OpenRouter models mid-session.

Each backend is an async generator: it yields UI events (dicts with "type")
while streaming, then finally yields ``{"type": "_result", ...}`` carrying the
canonical assistant blocks, stop reason, usage, and any reasoning text (for
the research log). agent.stream_turn consumes "_result" without forwarding it.
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

REQUIRED_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


# -- Anthropic (native Messages API; extended thinking supported) -------------


async def anthropic_step(model_id: str, options: dict, system: str,
                         tools: list, messages: list):
    from anthropic import AsyncAnthropic

    thinking = (options or {}).get("thinking")
    # Same timeout rationale as models/anthropic.py: thinking models can stay
    # silent for many minutes before answering.
    client = AsyncAnthropic(timeout=1800.0 if thinking else 600.0, max_retries=2)
    kwargs = dict(model=model_id, max_tokens=64000 if thinking else 16000,
                  system=system, tools=tools, messages=messages)
    if thinking:
        kwargs["thinking"] = thinking

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

    stop = final.stop_reason if final.stop_reason in ("tool_use", "max_tokens") else "end_turn"
    yield {
        "type": "_result",
        "blocks": [b.model_dump(exclude_none=True) for b in final.content],
        "stop_reason": stop,
        "usage": {"input_tokens": final.usage.input_tokens,
                  "output_tokens": final.usage.output_tokens},
    }


# -- OpenAI-compatible Chat Completions (api.openai.com and OpenRouter) -------


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in tools]


def to_openai_messages(system: str, messages: list) -> list[dict]:
    """Canonical (Anthropic-block) transcript -> Chat Completions messages.

    tool_use blocks become assistant tool_calls; tool_result blocks become
    role:"tool" messages. Tool-call ids pass through unchanged in both
    directions — neither API enforces an id format, only that calls and
    results match up.
    """
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "user":
            for b in content:
                if b.get("type") == "tool_result":
                    body = b.get("content")
                    if not isinstance(body, str):
                        body = json.dumps(body, ensure_ascii=False)
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": body})
        else:
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"],
                                   "arguments": json.dumps(b.get("input") or {},
                                                           ensure_ascii=False)}}
                     for b in content if b.get("type") == "tool_use"]
            if not text and not calls:
                continue
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
    return out


async def openai_compat_step(model_id: str, options: dict, system: str,
                             tools: list, messages: list,
                             base_url: str | None = None,
                             api_key_env: str = "OPENAI_API_KEY",
                             stream_text: bool = True):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url,
                         api_key=os.environ.get(api_key_env) or None,
                         timeout=600.0, max_retries=2)
    kwargs = dict(
        model=model_id,
        messages=to_openai_messages(system, messages),
        tools=to_openai_tools(tools),
    )
    if (options or {}).get("reasoning_effort"):
        # Native OpenAI reasoning dial (minimal..high) — the registry's
        # thinking/non-thinking delineation for gpt-5.x models.
        kwargs["reasoning_effort"] = options["reasoning_effort"]
    # api.openai.com reasoning models reject the legacy param; OpenRouter's
    # many providers only translate the classic one reliably.
    if base_url is None:
        kwargs["max_completion_tokens"] = 16000
    else:
        kwargs["max_tokens"] = 16000
        # usage flag: OpenRouter reports token usage through its own accounting
        # param. require_parameters: only route to upstream providers that
        # support every request param — without it, some gemini providers
        # silently drop `tools`, and the model chats about music it never
        # rendered (observed: text + zero usage, no tool call).
        kwargs["extra_body"] = {"usage": {"include": True},
                                "provider": {"require_parameters": True}}

    if not stream_text:
        # Non-streaming path: some OpenRouter providers (observed with
        # gemini-2.5-pro) stall indefinitely when a tool call has to be
        # streamed, while the same request completes in seconds without
        # streaming. The whole reply arrives at once; SSE pings keep the
        # browser connection alive meanwhile.
        resp = await client.chat.completions.create(**kwargs, stream=False)
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        if choice is not None and choice.finish_reason == "error":
            # OpenRouter reports upstream provider failures as a normal-looking
            # response with finish_reason "error": partial text, no tool call,
            # no usage. Raise so the agent's step retry re-issues the request
            # (observed flaky on google/gemini-2.5-pro; retries succeed).
            raise RuntimeError("upstream provider error (finish_reason=error)")
        blocks: list[dict] = []
        reasoning_text = (getattr(msg, "reasoning", None) or "") if msg else ""
        if msg and msg.content:
            yield {"type": "text_start"}
            yield {"type": "text", "delta": msg.content}
            blocks.append({"type": "text", "text": msg.content})
        for i, tc in enumerate(msg.tool_calls or [] if msg else []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                log.warning("tool-call arguments did not parse (%s); passing empty input", e)
                args = {}
            blocks.append({"type": "tool_use", "id": tc.id or f"call_{i}",
                           "name": tc.function.name, "input": args})
        finish = choice.finish_reason if choice else None
        stop = {"tool_calls": "tool_use", "length": "max_tokens"}.get(finish, "end_turn")
        if msg is not None and msg.tool_calls and stop == "end_turn":
            stop = "tool_use"
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                 "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0}
        yield {"type": "_result", "blocks": blocks, "stop_reason": stop,
               "usage": usage, "reasoning_text": reasoning_text}
        return

    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}
    stream = await client.chat.completions.create(**kwargs)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict] = {}
    finish = None
    usage = {"input_tokens": 0, "output_tokens": 0}
    text_started = False
    composing = False

    async for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = {"input_tokens": chunk.usage.prompt_tokens or 0,
                     "output_tokens": chunk.usage.completion_tokens or 0}
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta is not None:
            # OpenRouter surfaces reasoning-model traces as delta.reasoning;
            # captured for the research log, never replayed.
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                if not reasoning_parts:
                    yield {"type": "status", "text": "Thinking…"}
                reasoning_parts.append(reasoning)
            if delta.content:
                if not text_started:
                    text_started = True
                    yield {"type": "text_start"}
                text_parts.append(delta.content)
                yield {"type": "text", "delta": delta.content}
            for tc in delta.tool_calls or []:
                if not composing:
                    composing = True
                    yield {"type": "status", "text": "Composing…"}
                entry = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    entry["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        entry["name"] = tc.function.name
                    if tc.function.arguments:
                        entry["args"] += tc.function.arguments
        if choice.finish_reason:
            finish = choice.finish_reason

    blocks: list[dict] = []
    text = "".join(text_parts)
    if text:
        blocks.append({"type": "text", "text": text})
    for i in sorted(calls):
        entry = calls[i]
        try:
            args = json.loads(entry["args"]) if entry["args"].strip() else {}
        except json.JSONDecodeError as e:
            # Empty input makes the executor return a "missing field" error,
            # which flows back to the model as a normal fixable tool failure.
            log.warning("tool-call arguments did not parse (%s); passing empty input", e)
            args = {}
        blocks.append({"type": "tool_use", "id": entry["id"] or f"call_{i}",
                       "name": entry["name"], "input": args})

    if finish == "error":
        raise RuntimeError("upstream provider error (finish_reason=error)")
    stop = {"tool_calls": "tool_use", "length": "max_tokens"}.get(finish, "end_turn")
    if calls and stop == "end_turn":
        stop = "tool_use"  # some providers report plain 'stop' despite tool calls
    yield {
        "type": "_result",
        "blocks": blocks,
        "stop_reason": stop,
        "usage": usage,
        "reasoning_text": "".join(reasoning_parts),
    }


# -- OpenRouter (JSON-in-text protocol, no function calling) ------------------
#
# Function calling through OpenRouter proved unreliable for some providers
# (google/gemini-2.5-pro upstream-errors on most tool requests, and silently
# drops tools on others). The batch pipeline already talks to these same
# models by asking for a JSON object in plain text — 651 corpus pieces were
# generated that way — so this backend speaks that proven protocol and
# synthesizes canonical tool_use blocks from the parsed JSON. The agent loop
# can't tell the difference.


def _json_protocol(tools: list[dict]) -> str:
    lines = ["\n\n# Response protocol\n",
             "You cannot call functions directly. Instead:\n",
             "- To render music, respond with ONLY a single JSON object "
             "(no prose before or after) of this form:\n"]
    for t in tools:
        props = t["input_schema"]["properties"]
        fields = ", ".join(f'"{k}": <{v.get("description", k)}>' for k, v in props.items())
        lines.append(f'  {{"tool": "{t["name"]}", {fields}}}\n')
    lines.append(
        "- To speak to the composer without rendering, respond with plain text "
        "and no JSON object.\n"
        "Never mix prose and JSON in one response. Escape newlines inside JSON "
        "strings as \\n.")
    return "".join(lines)


def _to_json_messages(system: str, tools: list, messages: list) -> list[dict]:
    """Canonical transcript -> plain-text chat where past tool calls appear as
    the JSON the model 'wrote' and tool results as user-side text."""
    out: list[dict] = [{"role": "system", "content": system + _json_protocol(tools)}]
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "user":
            parts = []
            for b in content:
                if b.get("type") == "tool_result":
                    body = b.get("content")
                    if not isinstance(body, str):
                        body = json.dumps(body, ensure_ascii=False)
                    parts.append(f"RENDER RESULT: {body}")
            if parts:
                out.append({"role": "user", "content": "\n".join(parts)})
        else:
            parts = [b.get("text", "") for b in content if b.get("type") == "text"]
            for b in content:
                if b.get("type") == "tool_use":
                    parts.append(json.dumps({"tool": b["name"], **(b.get("input") or {})},
                                            ensure_ascii=False))
            joined = "\n".join(p for p in parts if p)
            if joined:
                out.append({"role": "assistant", "content": joined})
    return out


async def openrouter_step(model_id: str, options: dict, system: str,
                          tools: list, messages: list):
    from openai import AsyncOpenAI

    from ..modes._common import extract_json

    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL,
                         api_key=os.environ.get("OPENROUTER_API_KEY") or None,
                         timeout=600.0, max_retries=2)
    extra_body: dict = {"usage": {"include": True}}
    if (options or {}).get("reasoning"):
        # OpenRouter's unified reasoning control ({"effort": ...} or
        # {"enabled": false}), translated per upstream provider — how the
        # registry delineates thinking/non-thinking arms for these models.
        extra_body["reasoning"] = options["reasoning"]
    resp = await client.chat.completions.create(
        model=model_id,
        messages=_to_json_messages(system, tools, messages),
        max_tokens=32000,
        extra_body=extra_body)
    choice = resp.choices[0] if resp.choices else None
    if choice is not None and choice.finish_reason == "error":
        raise RuntimeError("upstream provider error (finish_reason=error)")
    msg = choice.message if choice else None
    text = (msg.content or "") if msg else ""
    reasoning_text = (getattr(msg, "reasoning", None) or "") if msg else ""
    if not text.strip() and reasoning_text.strip():
        # Some reasoning models strand the answer in the reasoning field
        # (same fallback the batch pipeline uses).
        text = reasoning_text

    blocks: list[dict] = []
    stop = "end_turn"
    try:
        obj = extract_json(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and "tool" in obj:
        # Any {"tool": ...} JSON becomes a tool_use block — even a tool that
        # isn't offered this turn. The agent's mode guard then returns a
        # corrective error the model can act on; dumping the raw JSON into the
        # chat as text is never right.
        name = obj.pop("tool")
        blocks.append({"type": "tool_use", "id": f"jsontool_{len(messages)}",
                       "name": name, "input": obj})
        stop = "tool_use"
    elif text.strip():
        yield {"type": "text_start"}
        yield {"type": "text", "delta": text}
        blocks.append({"type": "text", "text": text})

    usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
             "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0}
    yield {"type": "_result", "blocks": blocks, "stop_reason": stop,
           "usage": usage, "reasoning_text": reasoning_text}


BACKENDS = {
    "anthropic": anthropic_step,
    "openai": openai_compat_step,
    "openrouter": openrouter_step,
}
