"""Provider-agnostic LLM client interface.

Adding a new provider = implement this Protocol in a new module and register
its models in registry.py. Nothing else in the pipeline needs to change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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
