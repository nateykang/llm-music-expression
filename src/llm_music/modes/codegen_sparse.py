"""Sparse code-gen mode: identical to codegen but WITHOUT the music21 toolkit
doc — the model gets only the ABC-sized Outputs contract. An experimental arm
for testing how much the 104-line toolkit actually does; the canonical codegen
mode (and prompts/toolkit.md) are untouched.
"""

from __future__ import annotations

from .codegen import OUTPUTS, build_user_prompt, generate  # noqa: F401

USES_TOOLKIT = False
