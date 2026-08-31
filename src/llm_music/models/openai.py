"""OpenAI adapter.

Uses the Responses API (`client.responses.create`), which is the unified entry
point covering both standard chat models (gpt-4o, gpt-4.1, …) and reasoning
models (o-series, gpt-5-class). Falls back to Chat Completions if a deployment
doesn't expose Responses.
"""

from __future__ import annotations

import os

from .base import load_sdk


class OpenAIClient:
    """LLMClient implementation backed by the OpenAI API."""

    def __init__(self, name: str, model_id: str, max_output_tokens: int = 16000,
                 reasoning_effort: str | None = None):
        self.name = name
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        # Reasoning dial (minimal..high) — how the registry delineates the
        # thinking/non-thinking arms of one gpt-5.x model. None = provider default.
        self.reasoning_effort = reasoning_effort
        self._sdk = load_sdk("openai", "OpenAI")
        self._client = None  # lazy: the key is only needed when actually used

    def _ensure_client(self):
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to .env "
                    "(see .env.example)."
                )
            self._client = self._sdk()
        return self._client

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return self.complete_full(system, user, json_mode)[0]

    def complete_full(self, system: str, user: str,
                      json_mode: bool = False) -> tuple[str, dict]:
        client = self._ensure_client()  # json_mode unused: gpt-* already return clean JSON
        kwargs = {}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
            if self.reasoning_effort != "none":
                # Raw CoT is never returned; summaries are the retrievable trace.
                kwargs["reasoning"]["summary"] = "auto"
        try:
            resp = client.responses.create(
                model=self.model_id,
                instructions=system,
                input=user,
                max_output_tokens=self.max_output_tokens,
                **kwargs,
            )
            text = getattr(resp, "output_text", None) or _output_text_from_items(resp)
            summary = _reasoning_summary_from_items(resp)
            return text, ({"reasoning": summary} if summary else {})
        except (AttributeError, TypeError):
            # Older SDK without Responses API -> fall back to Chat Completions
            # (which never exposes reasoning).
            return self._chat_fallback(client, system, user), {}

    def _chat_fallback(self, client, system: str, user: str) -> str:
        # Reasoning models reject `temperature` and use `max_completion_tokens`.
        kwargs = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def _output_text_from_items(resp) -> str:
    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


def _reasoning_summary_from_items(resp) -> str:
    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "reasoning":
            for s in getattr(item, "summary", []) or []:
                text = getattr(s, "text", None)
                if text:
                    parts.append(text)
    return "\n".join(parts)
