"""OpenAI adapter.

Uses the Responses API (`client.responses.create`), which is the unified entry
point covering both standard chat models (gpt-4o, gpt-4.1, …) and reasoning
models (o-series, gpt-5-class). Falls back to Chat Completions if a deployment
doesn't expose Responses.
"""

from __future__ import annotations

import os


class OpenAIClient:
    """LLMClient implementation backed by the OpenAI API."""

    def __init__(self, name: str, model_id: str, max_output_tokens: int = 16000):
        self.name = name
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self._client = None  # lazy: import/key only needed when actually used

    def _ensure_client(self):
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to .env "
                    "(see .env.example)."
                )
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def complete(self, system: str, user: str) -> str:
        client = self._ensure_client()
        try:
            resp = client.responses.create(
                model=self.model_id,
                instructions=system,
                input=user,
                max_output_tokens=self.max_output_tokens,
            )
            text = getattr(resp, "output_text", None)
            if text:
                return text
            # Defensive: assemble from output items if output_text is empty.
            return _output_text_from_items(resp)
        except (AttributeError, TypeError):
            # Older SDK without Responses API -> fall back to Chat Completions.
            return self._chat_fallback(client, system, user)

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
