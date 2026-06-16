"""Subprocess entry point that executes untrusted LLM-generated music21 code.

Run as:  python -m llm_music._sandbox_runner <code_file> <midi_out> <musicxml_out>

The code is expected to build a music21 Score and bind it to a top-level name
``score`` (or ``s``). This runner applies resource limits, executes the code,
and writes MIDI + MusicXML. It NEVER imports anything from the parent package,
so the untrusted code runs with a minimal surface.
"""

from __future__ import annotations

import sys
import traceback


def _apply_limits() -> None:
    try:
        import resource
    except ImportError:  # not available on Windows
        return
    # 30s CPU and ~1.5 GiB address space; enough for a minute of music, not for abuse.
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        soft_as = 1536 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft_as, soft_as))
    except (ValueError, OSError):
        pass


def main() -> int:
    code_file, midi_out, xml_out = sys.argv[1], sys.argv[2], sys.argv[3]
    _apply_limits()

    with open(code_file, "r", encoding="utf-8") as fh:
        code = fh.read()

    ns: dict = {}
    try:
        exec(compile(code, "<llm_code>", "exec"), ns)  # noqa: S102 (intentional sandbox)
    except Exception:
        traceback.print_exc()
        return 2

    score = ns.get("score") or ns.get("s")
    if score is None:
        # Fall back: maybe a function returns the score.
        builder = ns.get("build") or ns.get("compose") or ns.get("main")
        if callable(builder):
            try:
                score = builder()
            except Exception:
                traceback.print_exc()
                return 2

    if score is None:
        print(
            "ERROR: code did not define a top-level `score` (a music21 Score).",
            file=sys.stderr,
        )
        return 3

    try:
        score.write("midi", fp=midi_out)
        score.write("musicxml", fp=xml_out)
    except Exception:
        traceback.print_exc()
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
