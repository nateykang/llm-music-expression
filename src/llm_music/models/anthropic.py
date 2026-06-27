"""Anthropic (Claude) adapter."""

from __future__ import annotations

import os


class AnthropicClient:
    """LLMClient implementation backed by the Anthropic Messages API."""

    def __init__(self, name: str, model_id: str, max_tokens: int = 16000,
                 thinking: dict | None = None):
        self.name = name
        self.model_id = model_id
        self.thinking = thinking
        # Thinking tokens count toward max_tokens. Some models over-think hard tasks
        # (sonnet-4.6-thinking spends ~31k thinking on free-form ABC) and at 32k they
        # hit the cap mid-thought and emit NO answer. 64k leaves room for thinking +
        # the answer; adaptive thinking only uses what it needs, so this is a ceiling.
        self.max_tokens = 64000 if thinking else max_tokens
        self._client = None  # lazily constructed so import never needs a key

    def _ensure_client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            from anthropic import Anthropic

            # Bound each request so a hung call can't pin a worker thread forever
            # (a stalled run got stuck with all workers blocked for 38 min, no timeout).
            # Generous (10 min) so reasoning/thinking models get their full think time —
            # the cap is a hang backstop, NOT a budget on legitimate reasoning.
            self._client = Anthropic(timeout=600.0, max_retries=2)
        return self._client

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:  # noqa: ARG002 (opus returns clean JSON)
        client = self._ensure_client()
        kwargs = dict(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.thinking:
            # Extended thinking requires temperature=1 (the default, so left unset)
            # and, with our high max_tokens, streaming (the SDK refuses non-streaming
            # for potentially-long requests). Thinking blocks are dropped; only the
            # answer text is returned, identical in shape to the non-thinking path.
            kwargs["thinking"] = self.thinking
            with client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
            return "".join(b.text for b in msg.content if b.type == "text")
        resp = client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")
