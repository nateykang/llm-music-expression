"""Anthropic (Claude) adapter."""

from __future__ import annotations

import os


class AnthropicClient:
    """LLMClient implementation backed by the Anthropic Messages API."""

    def __init__(self, name: str, model_id: str, max_tokens: int = 16000):
        self.name = name
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._client = None  # lazily constructed so import never needs a key

    def _ensure_client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            from anthropic import Anthropic

            self._client = Anthropic()
        return self._client

    def complete(self, system: str, user: str) -> str:
        client = self._ensure_client()
        resp = client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
