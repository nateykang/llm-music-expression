"""Model registry — the one place to add the newest model.

Each entry maps a friendly id (used in CLI/filenames) to a (provider, model_id)
pair. To add a model, append a line. To add a *provider*, write an adapter
module exposing the LLMClient protocol and extend ``_build_client``.
"""

from __future__ import annotations

from .base import LLMClient

# friendly id -> (provider, provider-specific model id)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "opus-4.8": ("anthropic", "claude-opus-4-8"),
    "sonnet-4.6": ("anthropic", "claude-sonnet-4-6"),
    # Add the newest model here, e.g.:
    # "haiku-4.5": ("anthropic", "claude-haiku-4-5-20251001"),
    # "gpt-5.2":   ("openai", "gpt-5.2"),                 # needs models/openai.py
    # "gemini-3-pro": ("openrouter", "google/gemini-3-pro"),  # needs models/openrouter.py
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY)


def get_client(name: str) -> LLMClient:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Known: {', '.join(list_models()) or '(none)'}"
        )
    provider, model_id = MODEL_REGISTRY[name]
    return _build_client(name, provider, model_id)


def _build_client(name: str, provider: str, model_id: str) -> LLMClient:
    if provider == "anthropic":
        from .anthropic import AnthropicClient

        return AnthropicClient(name=name, model_id=model_id)
    # elif provider == "openai":
    #     from .openai import OpenAIClient
    #     return OpenAIClient(name=name, model_id=model_id)
    raise ValueError(f"No adapter for provider '{provider}' (model '{name}').")
