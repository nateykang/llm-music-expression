"""Seeded, quality-stratified selection of the Part 1 pilot pieces.

For each of the 40 arms (opus-4.1± excluded): its own express-yourself CODEGEN
pieces, filtered to scores that load, transpose ±6 semitones, and re-render;
then 5 picked at quantile positions of the Part 2 panel quality (so the sample
spans the arm's quality range instead of being curated). Writes
part1_run/pilot_manifest.json before any judging happens.
"""
import csv
import json
import random
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

SEED = 20260917
BATCH = "20260820_061907__models_42_prompts_3"
EXCLUDE = {"opus-4.1", "opus-4.1-thinking"}
SHIFTS = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
QUANTS = [0.1, 0.3, 0.5, 0.7, 0.9]


def transposable(pc, batch_dir) -> bool:
    from music21 import converter
    from llm_music.judge import _score_to_text
    try:
        s = converter.parse(str(batch_dir / pc["score"]))
        base = _score_to_text(s)
        for k in (-6, 5):
            t = _score_to_text(s.transpose(k))
            if abs(len(t.split()) - len(base.split())) > 2:
                return False
        return True
    except Exception:
        return False


def main() -> int:
    random.seed(SEED)
    batch_dir = ROOT / "docs/data" / BATCH
    pieces = [p for p in json.loads((batch_dir / "data.json").read_text())["pieces"]
              if p.get("ok") and p["prompt"] == "express-yourself" and p.get("mode") == "codegen"
              and p["model"] not in EXCLUDE]
    quality = {}
    with open(ROOT / "part2_run/analysis/part2_selfpref.csv") as f:
        for r in csv.DictReader(f):
            if r["mode"] == "codegen" and r["prompt"] == "express-yourself":
                quality[(r["model"], int(r["sample"]))] = float(r["overall"])
    by_arm, dropped = {}, []
    for p in pieces:
        key = (p["model"], p.get("sample", 0))
        if key not in quality:
            dropped.append((key, "no part2 score")); continue
        if not transposable(p, batch_dir):
            dropped.append((key, "not transposable")); continue
        by_arm.setdefault(p["model"], []).append((quality[key], p.get("sample", 0), p["score"]))
    manifest = {"seed": SEED, "batch": BATCH, "shifts": SHIFTS, "mode": "codegen",
                "prompt": "express-yourself", "picked": {}, "dropped": dropped}
    for arm in sorted(by_arm):
        pool = sorted(by_arm[arm])
        idx = sorted({min(len(pool) - 1, round(q * (len(pool) - 1))) for q in QUANTS})
        while len(idx) < min(5, len(pool)):  # collision at small pools: fill nearest unused
            idx = sorted(set(idx) | {random.choice([i for i in range(len(pool)) if i not in idx])})
        manifest["picked"][arm] = [{"sample": pool[i][1], "overall": pool[i][0], "score": pool[i][2]}
                                   for i in idx[:5]]
    out = ROOT / "part1_run/pilot_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    n = sum(len(v) for v in manifest["picked"].values())
    print(f"picked {n} pieces across {len(manifest['picked'])} arms "
          f"(dropped {len(dropped)}: {dropped if dropped else 'none'}) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
