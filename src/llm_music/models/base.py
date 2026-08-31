"""Provider-agnostic LLM client interface.

Adding a new provider = implement this Protocol in a new module and register
its models in registry.py. Nothing else in the pipeline needs to change.
"""

from __future__ import annotations

import importlib
import threading
from typing import Protocol, runtime_checkable

# Provider SDKs import their submodules lazily, so two worker threads that first
# touch one at the same moment can trip CPython's import machinery ("deadlock
# detected by _ModuleLock('openai.resources.chat')"). Adapters load their SDK
# when the client is *constructed* — which every call site does on the main
# thread, before fanning out — and this lock serializes it for any that don't.
_SDK_LOCK = threading.Lock()


def load_sdk(module: str, attr: str):
    """Import a provider SDK class. Needs no API key, so it is safe to call
    whenever a client is built; the key is only required to send a request."""
    with _SDK_LOCK:
        return getattr(importlib.import_module(module), attr)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface the generation pipeline needs from any model."""

    #: Friendly id used in CLI args, filenames, and the data.json manifest.
    name: str

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Return the model's text response to a system + user prompt.

        json_mode requests strict JSON output from providers that support it (used
        by the LLM-judge so reasoning models don't strand the answer in a non-JSON
        reasoning trace). Providers that don't support it ignore the flag.
        """
        ...

    def complete_full(self, system: str, user: str,
                      json_mode: bool = False) -> tuple[str, dict]:
        """Like complete(), but also returns response metadata.

        The dict carries "reasoning" — the model's reasoning trace (or provider
        summary) when one comes back, else absent. Retrieving it is free: reasoning
        tokens are billed at generation time whether or not they are returned, and
        requesting them changes nothing the model sees.
        """
        ...
