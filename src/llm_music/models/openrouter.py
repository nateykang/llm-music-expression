"""OpenRouter adapter — one key, many providers (Gemini, Llama, DeepSeek, Qwen,
Mistral, Grok, Kimi, …). OpenRouter exposes an OpenAI-compatible Chat Completions
API; the model id is the OpenRouter slug, e.g. ``google/gemini-2.5-pro``.
"""

from __future__ import annotations

import os

from .base import load_sdk

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """LLMClient implementation backed by OpenRouter's OpenAI-compatible API."""

    def __init__(self, name: str, model_id: str, max_output_tokens: int = 32000,
                 reasoning: dict | None = None):
        self.name = name
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        # OpenRouter's unified reasoning control ({"effort": ...}, {"max_tokens": ...}
        # or {"enabled": False}), translated per upstream provider — how the registry
        # delineates thinking/non-thinking arms. None = provider default.
        self.reasoning = reasoning
        self._sdk = load_sdk("openai", "OpenAI")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set. Add it to .env (see .env.example)."
                )
            # Bound each request so a hung call can't pin a worker thread forever.
            # Generous (10 min) so reasoning models (gemini, deepseek, …) get their full
            # think time — the cap is a hang backstop, NOT a budget on legitimate reasoning.
            self._client = self._sdk(base_url=_BASE_URL, api_key=key, timeout=600.0,
                                     max_retries=2)
        return self._client

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return self.complete_full(system, user, json_mode)[0]

    def complete_full(self, system: str, user: str,
                      json_mode: bool = False) -> tuple[str, dict]:
        client = self._ensure_client()
        kwargs = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_output_tokens,
        )
        if self.reasoning:
            # Must ride extra_body: the OpenAI SDK rejects unknown top-level kwargs.
            kwargs["extra_body"] = {"reasoning": self.reasoning}
        if json_mode:
            # Force valid JSON into `content` so reasoning models can't strand the
            # answer in their reasoning trace (the gemini judge-parse-failure fix).
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0] if resp.choices else None
        if not choice or not choice.message:
            return "", {}
        msg = choice.message
        reasoning = getattr(msg, "reasoning", None) or None
        if not reasoning:  # deepseek-style field, or the newer details list
            extra = getattr(msg, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or None
            if not reasoning and extra.get("reasoning_details"):
                parts = [d.get("text") or d.get("summary") or ""
                         for d in extra["reasoning_details"] if isinstance(d, dict)]
                reasoning = "\n".join(p for p in parts if p) or None
        content = msg.content or ""
        # Reasoning models (gemini, deepseek, …) occasionally return empty content
        # with the answer stranded in the reasoning field — fall back so the JSON
        # extractor can still find it.
        if not content.strip():
            content = reasoning or ""
        return content, ({"reasoning": reasoning} if reasoning else {})
