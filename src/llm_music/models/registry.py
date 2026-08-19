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
    # Pre-adaptive 4.5s (v3 roster): thinking arms use enabled+budget_tokens.
    "sonnet-4.5": ("anthropic", "claude-sonnet-4-5-20250929"),
    "sonnet-4.5-thinking": ("anthropic", "claude-sonnet-4-5-20250929",
                            {"thinking": {"type": "enabled", "budget_tokens": 12000}}),
    "opus-4.5": ("anthropic", "claude-opus-4-5-20251101"),
    "opus-4.5-thinking": ("anthropic", "claude-opus-4-5-20251101",
                          {"thinking": {"type": "enabled", "budget_tokens": 12000}}),
    # Fable 5 — thinking is always on (adaptive is the only accepted config;
    # budget_tokens is rejected), so its ± pair is the output_config effort dial:
    # low = the non-thinking-analog arm, high (API default) = the thinking arm.
    "fable-5": ("anthropic", "claude-fable-5", {"thinking": {"type": "adaptive"}, "effort": "low"}),
    "fable-5-thinking": ("anthropic", "claude-fable-5", {"thinking": {"type": "adaptive"}, "effort": "high"}),
    # OpenAI (confirmed available on this org; extend as new ones ship).
    # v3 non-thinking arms are pinned to the reasoning floor ("none", 5.1+);
    # unpinned = provider default (medium) — what the v2 corpus ran.
    "gpt-5.5": ("openai", "gpt-5.5", {"reasoning_effort": "none"}),
    "gpt-5.5-thinking": ("openai", "gpt-5.5", {"reasoning_effort": "high"}),
    "gpt-5.2": ("openai", "gpt-5.2", {"reasoning_effort": "none"}),
    "gpt-5.2-thinking": ("openai", "gpt-5.2", {"reasoning_effort": "high"}),
    "gpt-4.1": ("openai", "gpt-4.1"),
    # 5.1/5.4 aren't listed on the native org — routed via OpenRouter.
    "gpt-5.1": ("openrouter", "openai/gpt-5.1", {"reasoning": {"enabled": False}}),
    "gpt-5.1-thinking": ("openrouter", "openai/gpt-5.1", {"reasoning": {"effort": "high"}}),
    "gpt-5.4": ("openrouter", "openai/gpt-5.4", {"reasoning": {"enabled": False}}),
    "gpt-5.4-thinking": ("openrouter", "openai/gpt-5.4", {"reasoning": {"effort": "high"}}),
    # v3 roster opener (Aug 2025). Routed via OpenRouter: the native API gates
    # gpt-5 behind org verification (404 on this org). Thinking pair via the
    # unified reasoning param; "minimal" is gpt-5's floor ("none" is 5.1+).
    "gpt-5": ("openrouter", "openai/gpt-5", {"reasoning": {"effort": "minimal"}}),
    "gpt-5-thinking": ("openrouter", "openai/gpt-5", {"reasoning": {"effort": "high"}}),
    # "o3": ("openai", "o3"),  # requires org verification — re-enable once verified
    # OpenRouter — frontier models from other labs (slugs verified live). The
    # study roster is the five closed frontier labs + the strongest open model:
    #   opus-4.8, gpt-5.5, gemini-2.5-pro, grok-4.3, deepseek-v4-pro, qwen3-max
    "gemini-2.5-pro": ("openrouter", "google/gemini-2.5-pro"),
    # opus-4.1 (Aug 2025, v3 roster opener) is retired from the native Anthropic
    # API (oldest still served there is sonnet-4.5) but lives on OpenRouter.
    # Thinking pinned per arm; 4.1 predates adaptive, so the thinking arm gets an
    # explicit budget rather than an effort level.
    "opus-4.1": ("openrouter", "anthropic/claude-opus-4.1", {"reasoning": {"enabled": False}}),
    "opus-4.1-thinking": ("openrouter", "anthropic/claude-opus-4.1", {"reasoning": {"max_tokens": 12000}}),
    # Gemini 3.1 Pro ships only as -preview on OpenRouter — and it is the OLDEST
    # living text Pro: gemini-3-pro was retired from Google's API and OpenRouter
    # (only -image variants remain). Pro tier cannot fully disable thinking, so
    # the pair is effort low vs high, pinned explicitly.
    "gemini-3.1-pro": ("openrouter", "google/gemini-3.1-pro-preview", {"reasoning": {"effort": "low"}}),
    "gemini-3.1-pro-thinking": ("openrouter", "google/gemini-3.1-pro-preview", {"reasoning": {"effort": "high"}}),
    # Google flash lineage, ± pinned. Plain ids were provider-default in the
    # v2 corpus. Only 3-flash (Dec 2025 preview — oldest still-served Google
    # model, candidate for the retired gemini-3-pro roster slot) honors a true
    # reasoning disable; from 3.5-flash on the API rejects enabled:false
    # ("Reasoning is mandatory"), so those pairs are effort low vs high like
    # the pro tier.
    "gemini-3-flash": ("openrouter", "google/gemini-3-flash-preview", {"reasoning": {"enabled": False}}),
    "gemini-3-flash-thinking": ("openrouter", "google/gemini-3-flash-preview", {"reasoning": {"effort": "high"}}),
    "gemini-3.5-flash": ("openrouter", "google/gemini-3.5-flash", {"reasoning": {"effort": "low"}}),
    "gemini-3.5-flash-thinking": ("openrouter", "google/gemini-3.5-flash", {"reasoning": {"effort": "high"}}),
    "gemini-3.6-flash": ("openrouter", "google/gemini-3.6-flash", {"reasoning": {"effort": "low"}}),
    "gemini-3.6-flash-thinking": ("openrouter", "google/gemini-3.6-flash", {"reasoning": {"effort": "high"}}),
    "gemini-3.7-flash": ("openrouter", "google/gemini-3.7-flash", {"reasoning": {"effort": "low"}}),
    "gemini-3.7-flash-thinking": ("openrouter", "google/gemini-3.7-flash", {"reasoning": {"effort": "high"}}),
    # grok-4.1 (the one Grok shipped with an explicit non-thinking variant) is no
    # longer served anywhere we can reach; 4.3 honors reasoning enabled:false via
    # OpenRouter (probe: 167 -> 0 reasoning tokens), so it carries the ± pair.
    "grok-4.3": ("openrouter", "x-ai/grok-4.3", {"reasoning": {"enabled": False}}),
    "grok-4.3-thinking": ("openrouter", "x-ai/grok-4.3", {"reasoning": {"effort": "high"}}),
    "grok-4.5": ("openrouter", "x-ai/grok-4.5"),
    "deepseek-v4-pro": ("openrouter", "deepseek/deepseek-v4-pro"),
    # v2-corpus DeepSeek slot: the 0731 retrain outscores V4-Pro on all nine
    # of DeepSeek's published agent/coding benchmarks. Pinned snapshot (not
    # -latest) for corpus reproducibility.
    "deepseek-v4-flash": ("openrouter", "deepseek/deepseek-v4-flash-0731"),
    "qwen3-max": ("openrouter", "qwen/qwen3-max"),
    "llama-4-maverick": ("openrouter", "meta-llama/llama-4-maverick"),
    # K3 always reasons (reasoning_effort low/high/max, default max) — its ± pair
    # is effort low vs high, pinned via OpenRouter's unified reasoning param.
    "kimi-k3": ("openrouter", "moonshotai/kimi-k3", {"reasoning": {"effort": "low"}}),
    "kimi-k3-thinking": ("openrouter", "moonshotai/kimi-k3", {"reasoning": {"effort": "high"}}),
    # The K2 thinking pair is two checkpoints, not a toggle: k2-0905 (pinned
    # snapshot) never reasons, k2-thinking (Nov 2025 continued-train) always does.
    "kimi-k2": ("openrouter", "moonshotai/kimi-k2-0905"),
    "kimi-k2-thinking": ("openrouter", "moonshotai/kimi-k2-thinking"),
    # gpt-5.6 ships only as codenamed variants; 'sol' is the one we expose (via
    # OpenRouter, since the native OpenAI org doesn't list gpt-5.6). Reasoning
    # is pinned per variant (OpenRouter's unified `reasoning` param) so the
    # thinking/non-thinking arms are explicit rather than provider defaults.
    "gpt-5.6": ("openrouter", "openai/gpt-5.6-sol", {"reasoning": {"enabled": False}}),
    "gpt-5.6-thinking": ("openrouter", "openai/gpt-5.6-sol", {"reasoning": {"effort": "high"}}),
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

        return AnthropicClient(name=name, model_id=model_id, thinking=options.get("thinking"),
                               effort=options.get("effort"))
    if provider == "openai":
        from .openai import OpenAIClient

        return OpenAIClient(name=name, model_id=model_id,
                            reasoning_effort=options.get("reasoning_effort"))
    if provider == "openrouter":
        from .openrouter import OpenRouterClient

        return OpenRouterClient(name=name, model_id=model_id,
                                reasoning=options.get("reasoning"))
    raise ValueError(f"No adapter for provider '{provider}' (model '{name}').")
