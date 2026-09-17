"""Caio common-set analysis: 9 pieces x 11 shifts x 40 judges, panel baseline.

Per judge: deviation from the panel on the SAME piece x shift (idiosyncratic
key preference, piece- and transposition-artifact-controlled), profile spread +
within-piece permutation flatness test; shared panel profile by shift and by
target tonic; favorite-key bonus per judge on deviations. Seeded."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1_run"))
from analyze_pilot import q8  # noqa: E402

SHIFTS = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
PCN = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
rng = np.random.default_rng(20260917)

ck = json.loads((ROOT / "part1_run/caio_ckpt.json").read_text())
reps = json.loads((ROOT / "part1_run/caio_reps.json").read_text())
grid = defaultdict(dict)   # (piece_stem, shift) -> judge -> q
tonic = {}
for key, v in ck.items():
    s = q8(v)
    if s is None:
        continue
    stem_shift, judge = key.rsplit("|", 1)
    stem, sh = stem_shift.rsplit("|", 1)
    grid[(stem, int(sh))][judge] = s
# target tonic = origin tonic (music21-analyzed, from the manifest resolution) + shift
ORIGIN = {"gpt-5.6|express-yourself|5": 0, "gemini-3.7-flash|express-yourself|9": 11,
          "fable-5|express-yourself|4": 0, "gpt-5.6|uniquely-you|18": 0,
          "gemini-3.7-flash|uniquely-you|13": 2, "fable-5|uniquely-you|17": 0,
          "gemini-3.7-flash|emotional-state|3": 11, "fable-5|emotional-state|16": 2,
          "gpt-5.6|emotional-state|6": 4}
for (stem, sh) in list(grid):
    parts = stem.split("|")
    tonic[(stem, sh)] = (ORIGIN[f"{parts[1]}|{parts[2]}|{parts[4]}"] + sh) % 12
judges = sorted({j for c in grid.values() for j in c})
pieces = sorted({p for p, _ in grid})

# shared profile: panel mean per cell, centered within piece across shifts
panel = {cell: np.mean(list(js.values())) for cell, js in grid.items()}
cent = {}
for p in pieces:
    mu = np.mean([panel[(p, sh)] for sh in SHIFTS])
    for sh in SHIFTS:
        cent[(p, sh)] = panel[(p, sh)] - mu
print("shared panel profile by shift (all 40 judges x 9 pieces, ±2SE over pieces):")
for sh in SHIFTS:
    vals = [cent[(p, sh)] for p in pieces]
    print(f"  {sh:+d}: {np.mean(vals):+.3f} ± {2*np.std(vals)/np.sqrt(len(vals)):.3f}")
by_t = defaultdict(list)
for (p, sh), v in cent.items():
    if tonic[(p, sh)] is not None:
        by_t[tonic[(p, sh)]].append(v)
print("\nshared profile by target tonic:")
print("  " + "  ".join(f"{PCN[t]}:{np.mean(by_t[t]):+.2f}(n={len(by_t[t])})" for t in sorted(by_t)))

# per-judge deviations from the panel of the other 39
dev = defaultdict(dict)
for cell, js in grid.items():
    for j, s in js.items():
        others = [x for k, x in js.items() if k != j]
        dev[j][cell] = s - np.mean(others)
rows = []
for j in judges:
    prof = {sh: np.mean([dev[j][(p, sh)] for p in pieces]) for sh in SHIFTS}
    mu_j = np.mean(list(prof.values()))
    prof = {sh: v - mu_j for sh, v in prof.items()}
    spread = float(np.std(list(prof.values())))
    null = []
    for _ in range(2000):
        fake = defaultdict(list)
        for p in pieces:
            shp = rng.permutation(SHIFTS)
            for s2, sh in zip(shp, SHIFTS):
                fake[s2].append(dev[j][(p, sh)])
        null.append(np.std([np.mean(fake[sh]) for sh in SHIFTS]))
    pval = float(np.mean([n >= spread for n in null]))
    rows.append({"judge": j, "spread": spread, "p": pval, "profile": {str(k): float(v) for k, v in prof.items()}})
rows.sort(key=lambda r: r["p"])
sig = [r for r in rows if r["p"] < 0.05]
print(f"\nper-judge idiosyncratic key profiles: {len(sig)}/40 non-flat at p<0.05 (chance ~2)")
for r in rows[:6]:
    pr = {int(k): v for k, v in r["profile"].items()}
    b, w = max(pr, key=pr.get), min(pr, key=pr.get)
    print(f"  {r['judge']:26s} spread {r['spread']:.3f}  p={r['p']:.3f}  best {b:+d} ({pr[b]:+.2f}) worst {w:+d} ({pr[w]:+.2f})")

# favorite-key bonus on deviations (generation keys from the corpus scan)
import re
PCM = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
keys_by_arm = defaultdict(Counter)
for bd in (ROOT / "docs/data").iterdir():
    f = bd / "data.json"
    if not f.exists() or not bd.name.startswith("202608"):
        continue
    for pc in json.loads(f.read_text())["pieces"]:
        if pc.get("ok") and pc.get("mode") == "abc":
            m = re.search(r"^K:\s*([A-G])([#b]?)", pc.get("abc", ""), re.M)
            if m:
                keys_by_arm[pc["model"]][(PCM[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)) % 12] += 1
bonus = []
for j in judges:
    if not keys_by_arm.get(j):
        continue
    fav = keys_by_arm[j].most_common(1)[0][0]
    own = [dev[j][c] for c in dev[j] if tonic[c] == fav]
    oth = [dev[j][c] for c in dev[j] if tonic[c] is not None and tonic[c] != fav]
    if len(own) >= 3:
        bonus.append({"judge": j, "fav": PCN[fav], "bonus": float(np.mean(own) - np.mean(oth)), "n": len(own)})
bs = np.array([b["bonus"] for b in bonus])
print(f"\nfavorite-key bonus (deviation-based): pooled {bs.mean():+.3f} ± {2*bs.std()/np.sqrt(len(bs)):.3f} over {len(bonus)} judges")
top = sorted(bonus, key=lambda b: -abs(b["bonus"]))[:3]
print("  largest:", [(b["judge"], round(b["bonus"], 2), f"fav {b['fav']}", f"n={b['n']}") for b in top])
(ROOT / "part1_run/caio_analysis.json").write_text(json.dumps(
    {"judges": rows, "bonus": bonus, "pooled_bonus": [float(bs.mean()), float(2*bs.std()/np.sqrt(len(bs)))],
     "shared_by_shift": {str(sh): [float(np.mean([cent[(p, sh)] for p in pieces])),
                                   float(2*np.std([cent[(p, sh)] for p in pieces])/np.sqrt(len(pieces)))] for sh in SHIFTS}}, indent=1))
print("\nwrote part1_run/caio_analysis.json")
