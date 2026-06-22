"""Model registry — the one place to add the newest model.

Each entry maps a friendly id (used in CLI/filenames) to a (provider, model_id)
pair. To add a model, append a line. To add a *provider*, write an adapter
module exposing the LLMClient protocol and extend ``_build_client``.
"""

from __future__ import annotations

from .base import LLMClient

# friendly id -> (provider, provider-specific model id)
# The model id on the right must match exactly what your account/org exposes;
# adjust to taste. Run `llm-music models` to see what's registered.
MODEL_REGISTRY: dict[str, tuple] = {
    # Anthropic. A 3rd tuple element carries provider options (e.g. extended
    # thinking) — used for the thinking-on/off ablation on the same model.
    "opus-4.8": ("anthropic", "claude-opus-4-8"),
    "opus-4.8-thinking": ("anthropic", "claude-opus-4-8", {"thinking": {"type": "adaptive"}}),
    "sonnet-4.6": ("anthropic", "claude-sonnet-4-6"),
    "sonnet-4.6-thinking": ("anthropic", "claude-sonnet-4-6", {"thinking": {"type": "adaptive"}}),
    # OpenAI (confirmed available on this org; extend as new ones ship)
    "gpt-5.5": ("openai", "gpt-5.5"),
    "gpt-5.2": ("openai", "gpt-5.2"),
    "gpt-4.1": ("openai", "gpt-4.1"),
    # "o3": ("openai", "o3"),  # requires org verification — re-enable once verified
    # OpenRouter — frontier models from other labs (slugs verified live). The
    # study roster is the five closed frontier labs + the strongest open model:
    #   opus-4.8, gpt-5.5, gemini-2.5-pro, grok-4.3, deepseek-v4-pro, qwen3-max
    "gemini-2.5-pro": ("openrouter", "google/gemini-2.5-pro"),
    "grok-4.3": ("openrouter", "x-ai/grok-4.3"),
    "deepseek-v4-pro": ("openrouter", "deepseek/deepseek-v4-pro"),
    "qwen3-max": ("openrouter", "qwen/qwen3-max"),
    "llama-4-maverick": ("openrouter", "meta-llama/llama-4-maverick"),
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY)


def get_client(name: str) -> LLMClient:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Known: {', '.join(list_models()) or '(none)'}"
        )
    provider, model_id, *rest = MODEL_REGISTRY[name]
    options = rest[0] if rest else {}
    return _build_client(name, provider, model_id, options)


def _build_client(name: str, provider: str, model_id: str, options: dict | None = None) -> LLMClient:
    options = options or {}
    if provider == "anthropic":
        from .anthropic import AnthropicClient

        return AnthropicClient(name=name, model_id=model_id, thinking=options.get("thinking"))
    if provider == "openai":
        from .openai import OpenAIClient

        return OpenAIClient(name=name, model_id=model_id)
    if provider == "openrouter":
        from .openrouter import OpenRouterClient

        return OpenRouterClient(name=name, model_id=model_id)
    raise ValueError(f"No adapter for provider '{provider}' (model '{name}').")
