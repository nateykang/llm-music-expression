#!/usr/bin/env python3
"""Build the relative-key relabel dataset: for every free-form ABC piece whose
K: declares a plain major or minor key, produce a variant with every K: line
swapped to the RELATIVE key (C <-> Am, Bb <-> Gm, ...). Relative keys share a
key signature, so the swap changes ZERO notes — only the declared mode the
judge reads. Judging originals vs variants isolates label-driven mode bias.

    python scripts/relabel_keys.py          # writes docs/analysis/relabel_experiment.json
                                            # + verifies note-identity on a sample

Pieces skipped: modal K: (dorian etc.), unparseable K:, or no K: at all.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

# relative pairs by shared key signature (majors -> relative minors)
REL = {"C": "Am", "G": "Em", "D": "Bm", "A": "F#m", "E": "C#m", "B": "G#m",
       "F#": "D#m", "C#": "A#m", "F": "Dm", "Bb": "Gm", "Eb": "Cm", "Ab": "Fm",
       "Db": "Bbm", "Gb": "Ebm", "Cb": "Abm"}
REL.update({v: k for k, v in list(REL.items())})

# K:<tonic token> [mode word] [anything else]. Parsed strictly: the tonic token
# and (optionally) the next whitespace-separated word must be EXACTLY a plain
# major/minor spelling — anything else (dor, mix, phr, lyd, loc, exp, ...)
# rejects the piece rather than risking a mangled signature (K:Ddor -> K:Bmdor
# was a real bug this strictness fixes).
KLINE = re.compile(r"^(?P<pre>[ \t]*K:[ \t]*)(?P<body>\S+)(?P<rest>.*)$")
TONIC = re.compile(r"^(?P<tonic>[A-Ga-g])(?P<acc>[#b]?)(?P<suffix>.*)$")
MAJOR_WORDS = {"", "maj", "major"}
MINOR_WORDS = {"m", "min", "minor"}


def swap_key_line(line: str):
    m = KLINE.match(line)
    if not m:
        return None
    t = TONIC.match(m.group("body"))
    if not t:
        return None
    suffix = t.group("suffix").lower()
    rest = m.group("rest")
    if suffix == "":
        # mode may be the NEXT word ("K:D minor"); consume it if it's plain
        nxt = re.match(r"^([ \t]+)([A-Za-z]+)(.*)$", rest)
        if nxt and nxt.group(2).lower() in MAJOR_WORDS | MINOR_WORDS:
            suffix = nxt.group(2).lower()
            rest = nxt.group(3)
    if suffix in MAJOR_WORDS:
        cur = t.group("tonic").upper() + t.group("acc")
    elif suffix in MINOR_WORDS:
        cur = t.group("tonic").upper() + t.group("acc") + "m"
    else:
        return None  # modal / unrecognized — caller skips the piece
    # trailing bare words are mode qualifiers ("K:Gm Dorian") — reject those;
    # key=value settings (clef=, name=, middle=) are layout-only and safe.
    first_rest = rest.strip().split(" ", 1)[0] if rest.strip() else ""
    if first_rest and first_rest.isalpha():
        return None
    new = REL.get(cur)
    if new is None:
        return None
    return m.group("pre") + new + rest, cur, new


def relabel(abc: str):
    """Swap every plain major/minor K: line. Returns (new_abc, orig_key, new_key)
    for the FIRST K:, or None if any K: line is modal/unparseable."""
    out, first = [], None
    for ln in abc.splitlines():
        if ln.lstrip().upper().startswith("K:"):
            r = swap_key_line(ln)
            if r is None:
                return None
            ln, cur, new = r
            if first is None:
                first = (cur, new)
        out.append(ln)
    if first is None:
        return None
    return "\n".join(out), first[0], first[1]


def main():
    rows, skipped = [], 0
    for dj in sorted((ROOT / "docs/data").glob("*/data.json")):
        batch = dj.parent
        for p in json.loads(dj.read_text(encoding="utf-8"))["pieces"]:
            if not (p.get("ok") and p.get("abc") and p.get("prompt") == "free-form"):
                continue
            r = relabel(p["abc"])
            if r is None:
                skipped += 1
                continue
            new_abc, orig_key, new_key = r
            rows.append({"model": p["model"], "mode": p.get("mode"),
                         "title": p.get("title", ""), "sample": p.get("sample", 0),
                         "batch": batch.name, "orig_key": orig_key, "new_key": new_key,
                         "abc": new_abc})
    out = ROOT / "docs/analysis/relabel_experiment.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    n_min2maj = sum(1 for r in rows if r["orig_key"].endswith("m"))
    print(f"relabeled {len(rows)} pieces ({n_min2maj} minor->major, "
          f"{len(rows) - n_min2maj} major->minor); skipped {skipped} "
          f"(modal / unparseable K:) → {out}")

    # ---- verification: the swap must change ZERO rendered notes ----
    from llm_music.render import abc_to_midi
    import pretty_midi

    def notes(abc):
        with tempfile.TemporaryDirectory() as td:
            m = abc_to_midi(abc, Path(td), gchords=False)
            if not m:
                return None
            try:
                pm = pretty_midi.PrettyMIDI(str(m))
            except Exception:  # e.g. abc2midi running-status MIDIs mido rejects
                return None
            return sorted((round(n.start, 4), n.pitch, round(n.end, 4))
                          for i in pm.instruments for n in i.notes)

    sample = rows  # verify EVERY piece — a single mangled signature poisons the experiment
    orig_by = {}
    for dj in sorted((ROOT / "docs/data").glob("*/data.json")):
        for p in json.loads(dj.read_text(encoding="utf-8"))["pieces"]:
            # free-form only: models reuse titles ACROSS prompts within a batch
            # (sonnet titled its free-form and 'modern' pieces both "Liminal"),
            # so a prompt-blind key pairs the wrong original.
            if p.get("ok") and p.get("abc") and p.get("prompt") == "free-form":
                orig_by[(p["model"], p.get("mode"), p.get("title", ""),
                         p.get("sample", 0), dj.parent.name)] = p["abc"]
    ok = bad = unrenderable = 0
    for r in sample:
        o = orig_by[(r["model"], r["mode"], r["title"], r["sample"], r["batch"])]
        a, b = notes(o), notes(r["abc"])
        if a is None or b is None:
            unrenderable += 1
        elif a == b:
            ok += 1
        else:
            bad += 1
            print(f"  NOTE MISMATCH: {r['model']} '{r['title'][:30]}' "
                  f"{r['orig_key']}->{r['new_key']}")
    print(f"note-identity check on {len(sample)} random pieces: "
          f"{ok} identical, {bad} MISMATCHED, {unrenderable} unrenderable")
    if bad:
        sys.exit("relabeling changed notes somewhere — do not judge this set")


if __name__ == "__main__":
    main()
