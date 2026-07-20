"""Generation modes. Each turns an LLM response into a renderable artifact."""

from . import abc, codegen, codegen_sparse, smt_abc

MODES = {"codegen": codegen, "abc": abc, "smt-abc": smt_abc,
         "codegen-sparse": codegen_sparse}

__all__ = ["MODES", "abc", "codegen", "codegen_sparse", "smt_abc"]
