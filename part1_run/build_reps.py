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
                t = _score_to_text(s.transpose(k))
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
