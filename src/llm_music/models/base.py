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

    def complete(self, system: str, user: str) -> str:
        """Return the model's text response to a system + user prompt."""
        ...
