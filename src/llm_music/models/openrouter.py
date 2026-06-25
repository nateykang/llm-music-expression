"""OpenRouter adapter — one key, many providers (Gemini, Llama, DeepSeek, Qwen,
Mistral, Grok, Kimi, …). OpenRouter exposes an OpenAI-compatible Chat Completions
API; the model id is the OpenRouter slug, e.g. ``google/gemini-2.5-pro``.
"""

from __future__ import annotations

import os

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """LLMClient implementation backed by OpenRouter's OpenAI-compatible API."""

    def __init__(self, name: str, model_id: str, max_output_tokens: int = 32000):
        self.name = name
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set. Add it to .env (see .env.example)."
                )
            from openai import OpenAI

            # Bound each request so a hung call can't pin a worker thread forever.
            # Generous (10 min) so reasoning models (gemini, deepseek, …) get their full
            # think time — the cap is a hang backstop, NOT a budget on legitimate reasoning.
            self._client = OpenAI(base_url=_BASE_URL, api_key=key, timeout=600.0, max_retries=2)
        return self._client

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        client = self._ensure_client()
        kwargs = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_output_tokens,
        )
        if json_mode:
            # Force valid JSON into `content` so reasoning models can't strand the
            # answer in their reasoning trace (the gemini judge-parse-failure fix).
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0] if resp.choices else None
        if not choice or not choice.message:
            return ""
        content = choice.message.content or ""
        # Reasoning models (gemini, deepseek, …) occasionally return empty content
        # with the answer stranded in the reasoning field — fall back so the JSON
        # extractor can still find it.
        if not content.strip():
            content = getattr(choice.message, "reasoning", "") or ""
        return content
