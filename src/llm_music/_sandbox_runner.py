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


def _relocate_tempo_marks(score, parts) -> None:
    """Move Score-/Part-level MetronomeMarks into the Measure covering their offset.

    music21's MusicXML export only writes a MetronomeMark that lives inside a
    Measure; a mark inserted directly on the Score, or on a Part that contains
    explicit Measures, is silently dropped — and our MIDI is derived from the
    MusicXML, so the piece's tempo is lost end to end (the analyzer then falls
    back to 120 BPM). The toolkit historically told models to insert tempo at
    part level, so honor that placement here. Parts holding loose notes are left
    alone: export's makeMeasures already carries their marks into bar 1.
    """
    from music21 import stream as m21stream, tempo as m21tempo

    holders = {}
    if parts:
        holders[id(score)] = (score, parts[0])  # score-level marks -> first part
    for p in parts:
        holders[id(p)] = (p, p)
    for holder, part in holders.values():
        for mm in list(holder.getElementsByClass(m21tempo.MetronomeMark)):
            measures = list(part.getElementsByClass(m21stream.Measure))
            if not measures:
                continue
            target = measures[0]
            for m in measures:
                if float(m.offset) <= float(mm.offset):
                    target = m
                else:
                    break
            holder.remove(mm)
            target.insert(max(0.0, float(mm.offset) - float(target.offset)), mm)


def _validate_exported(parsed) -> "str | None":
    """Reject degenerate exports: a part with zero notes, or a long silent tail.

    Both patterns come from music21 misuse that exec() can't catch (e.g. two
    Voices appended sequentially instead of stacked), and both previously
    shipped as 'ok' pieces. Validating the re-parsed MusicXML checks what will
    actually be heard and engraved.
    """
    from music21 import stream as m21stream

    parts = list(getattr(parsed, "parts", [])) or [parsed]
    empties = []
    total_m = 0
    last_sound = 0
    for p in parts:
        measures = list(p.getElementsByClass(m21stream.Measure))
        total_m = max(total_m, len(measures))
        n_notes = 0
        for mi, m in enumerate(measures, 1):
            k = len(m.flatten().notes)
            if k:
                n_notes += k
                last_sound = max(last_sound, mi)
        if n_notes == 0:
            try:
                name = p.partName or p.getInstrument(returnDefault=True).instrumentName
            except Exception:
                name = None
            empties.append(name or "unnamed part")
    if empties:
        return (
            "Degenerate score: instrument part(s) contain zero notes after export: "
            + ", ".join(empties)
            + ". Every part you create must end up with notes in it — check that you "
            "appended your measures/voices into the container you attached to the score."
        )
    silent_tail = total_m - last_sound
    if total_m and silent_tail > max(4, 0.2 * total_m):
        return (
            f"Degenerate score: the final {silent_tail} of {total_m} measures contain no "
            "notes in any part. This usually means content landed at the wrong offset — "
            "e.g. two Voices appended one after another (sequential) when they should "
            "sound together; build parallel lines as separate Parts, or put Voices "
            "inside the same Measure."
        )
    return None


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
        from music21 import clef, converter

        # Give every part a clef (models often omit them, which breaks engraving
        # of bass/LH staves).
        parts = list(getattr(score, "parts", [])) or [score]
        for p in parts:
            if list(p.recurse().getElementsByClass(clef.Clef)):
                continue
            try:
                best = clef.bestClef(p, recurse=True)
            except Exception:
                continue
            target = p.recurse().getElementsByClass("Measure").first() or p
            target.insert(0, best)

        _relocate_tempo_marks(score, parts)

        # Write MusicXML first, then derive MIDI FROM it. music21's direct MIDI
        # export drops a grand-staff's second (bass) part while its MusicXML keeps
        # it; rendering MIDI from the MusicXML guarantees audio == engraving.
        score.write("musicxml", fp=xml_out)
        parsed = converter.parse(xml_out)
        problem = _validate_exported(parsed)
        if problem:
            print("ERROR: " + problem, file=sys.stderr)
            return 5
        parsed.write("midi", fp=midi_out)
    except Exception:
        traceback.print_exc()
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
