"""Generation modes. Each turns an LLM response into a renderable artifact."""

from . import abc, codegen, smt_abc

MODES = {"codegen": codegen, "abc": abc, "smt-abc": smt_abc}

__all__ = ["MODES", "abc", "codegen", "smt_abc"]
