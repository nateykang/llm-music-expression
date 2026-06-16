"""Generation modes. Each turns an LLM response into MIDI + MusicXML."""

from . import abc, codegen

MODES = {"codegen": codegen, "abc": abc}

__all__ = ["MODES", "abc", "codegen"]
