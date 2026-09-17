"""Render the transposed note-listing representations for the pilot.

For every manifest piece × shift: parse the MusicXML source of truth, transpose
by k semitones (music21 handles key signatures and spelling), and render with
the SAME _score_to_text used for Part 2 — so the judge input format is
byte-identical up to the pitches. Validates that each transposition preserves
the token structure and shifts pitch classes correctly. Writes
part1_run/pilot_reps.json keyed batch|model|express-yourself|codegen|sample|shift.
"""
import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE = re.compile(r"\b([A-G])([#-]{0,2})(\d)/")


def pcs(text):
    out = []
    for m in NOTE.finditer(text):
        p = PC[m.group(1)] + m.group(2).count("#") - m.group(2).count("-")
        out.append(p % 12)
    return out


# For k semitones, the two enharmonic interval spellings and their effect on the
# key signature (circle-of-fifths delta). The chosen spelling minimizes the
# resulting |sharps|, so a transposition never lands on an 8-sharp signature
# when the 4-flat spelling of the same pitches exists. Note tokens are further
# confined to the base corpus vocabulary: no double accidentals, no E#/B#/F-/C-
# (0.09 per rep there) — pitches unchanged, spelling normalized.
CANDIDATES = {
    1: [("m2", -5), ("a1", 7)], 2: [("M2", 2), ("d3", -10)], 3: [("m3", -3), ("a2", 9)],
    4: [("M3", 4), ("d4", -8)], 5: [("P4", -1), ("a3", 11)], 6: [("d5", -6), ("a4", 6)],
}
for _k, _c in list(CANDIDATES.items()):
    CANDIDATES[-_k] = [(n[0] + "-" + n[1:], -d) for n, d in _c]


def desugar_doubles(score):
    """Respell double-sharp/flat pitches to their simple enharmonic (F## -> G).
    The base corpus the judges calibrated on contains zero double accidentals,
    so leaving them in would be a notation novelty confound, not a key effect."""
    for n in score.recurse().notes:
        for pt in n.pitches:
            while pt.accidental is not None and (
                    abs(pt.accidental.alter) >= 2
                    or (pt.step, pt.accidental.alter) in
                    {("E", 1), ("B", 1), ("F", -1), ("C", -1)}):
                e = pt.getEnharmonic()
                pt.step, pt.octave = e.step, e.octave
                pt.accidental = e.accidental
    return score


def best_interval(score, k):
    from music21 import interval, key as m21key
    ks = next(iter(score.recurse().getElementsByClass(m21key.KeySignature)), None)
    s0 = ks.sharps if ks is not None else 0
    name, _ = min(CANDIDATES[k], key=lambda cd: (abs(s0 + cd[1]), cd[1]))
    return interval.Interval(name)


def main() -> int:
    from music21 import converter
    from llm_music.judge import _score_to_text
    man = json.loads((ROOT / "part1_run/pilot_manifest.json").read_text())
    batch_dir = ROOT / "docs/data" / man["batch"]
    reps, bad = {}, []
    for arm, picks in man["picked"].items():
        for p in picks:
            s = converter.parse(str(batch_dir / p["score"]))
            base = _score_to_text(s)
            base_pcs = pcs(base)
            for k in man["shifts"]:
                t = _score_to_text(desugar_doubles(s.transpose(best_interval(s, k))))
                key = f"{man['batch']}|{arm}|express-yourself|codegen|{p['sample']}|{k}"
                reps[key] = t
                got, want = pcs(t), [(x + k) % 12 for x in base_pcs]
                if len(got) != len(want) or got != want:
                    bad.append((key, len(got), len(want),
                                sum(1 for a, b in zip(got, want) if a != b)))
    out = ROOT / "part1_run/pilot_reps.json"
    out.write_text(json.dumps(reps))
    print(f"rendered {len(reps)} transposed reps → {out}")
    print("pitch-class validation:", "ALL EXACT" if not bad else f"{len(bad)} MISMATCHES: {bad[:5]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
