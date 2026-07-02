"""Run untrusted LLM music21 code in an isolated subprocess.

This is the security boundary for code-gen mode: the LLM's Python never runs in
our process. It executes via ``_sandbox_runner`` in a child process with CPU /
memory limits and a hard wall-clock timeout, in a scratch working directory.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


from .render import _tail


@dataclass
class SandboxResult:
    ok: bool
    midi_path: Path | None
    musicxml_path: Path | None
    error: str | None = None


def run_music21_code(code: str, out_dir: Path, timeout: int = 60) -> SandboxResult:
    """Execute ``code`` (which must build a music21 ``score``) and capture outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = out_dir / "piece.mid"
    xml_path = out_dir / "piece.musicxml"

    with tempfile.TemporaryDirectory(prefix="llm_music_") as scratch:
        # Not "code.py": cwd is on sys.path for the subprocess, so a module named
        # `code` would shadow the stdlib `code` module (imported by pdb et al.)
        # and corrupt tracebacks when the model's code raises.
        code_file = Path(scratch) / "_generated_piece.py"
        code_file.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "llm_music._sandbox_runner",
                    str(code_file),
                    str(midi_path),
                    str(xml_path),
                ],
                cwd=scratch,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            # TimeoutExpired carries the output captured before the kill (as
            # bytes, even with text=True) — include it so a hang isn't opaque.
            detail = _tail(e.stderr) or _tail(e.stdout)
            msg = f"timed out after {timeout}s"
            if detail:
                msg += f"; output before timeout: {detail}"
            return SandboxResult(False, None, None, msg)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return SandboxResult(False, None, None, err or f"exit {proc.returncode}")
    if not midi_path.exists() or not xml_path.exists():
        return SandboxResult(False, None, None, "no output files produced")
    return SandboxResult(True, midi_path, xml_path, None)
