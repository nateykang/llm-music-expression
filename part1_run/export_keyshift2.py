"""Export the fixed-pieces key-shift EDA (Part 1 stage 2) for the website:
Q1/Q2 figures + a JSON with the Q3 favorite-key table, into docs/analysis/.
Spreads/p-values are taken from caio_analysis.json (the reported numbers);
noise bars, tonic profile and bonus CIs are recomputed from the checkpoint
with the same seed."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1_run"))
from analyze_pilot import q8  # noqa: E402

BLUE = "#2a78d6"
SHIFTS = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
PCN = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PCM = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ORIGIN = {"gpt-5.6|express-yourself|5": 0, "gemini-3.7-flash|express-yourself|9": 11,
          "fable-5|express-yourself|4": 0, "gpt-5.6|uniquely-you|18": 0,
          "gemini-3.7-flash|uniquely-you|13": 2, "fable-5|uniquely-you|17": 0,
          "gemini-3.7-flash|emotional-state|3": 11, "fable-5|emotional-state|16": 2,
          "gpt-5.6|emotional-state|6": 4}
rng = np.random.default_rng(20260917)
OUT = ROOT / "docs/analysis"

ck = json.loads((ROOT / "part1_run/caio_ckpt.json").read_text())
ana = {r["judge"]: r for r in json.loads((ROOT / "part1_run/caio_analysis.json").read_text())["judges"]}
grid = defaultdict(dict)
for key, v in ck.items():
    s = q8(v)
    if s is None:
        continue
    stem_shift, judge = key.rsplit("|", 1)
    stem, sh = stem_shift.rsplit("|", 1)
    grid[(stem, int(sh))][judge] = s
judges = sorted({j for c in grid.values() for j in c})
pieces = sorted({p for p, _ in grid})
tonic = {(p, sh): (ORIGIN["|".join([p.split("|")[1], p.split("|")[2], p.split("|")[4]])] + sh) % 12
         for (p, sh) in grid}
panel = {cell: np.mean(list(js.values())) for cell, js in grid.items()}
cent = {}
for p in pieces:
    mu = np.mean([panel[(p, sh)] for sh in SHIFTS])
    for sh in SHIFTS:
        cent[(p, sh)] = panel[(p, sh)] - mu
dev = defaultdict(dict)
for cell, js in grid.items():
    for j, s in js.items():
        dev[j][cell] = s - np.mean([x for k, x in js.items() if k != j])

# Q1 figure: shared profile by shift and by target tonic
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
y = [np.mean([cent[(p, sh)] for p in pieces]) for sh in SHIFTS]
e = [2 * np.std([cent[(p, sh)] for p in pieces]) / np.sqrt(len(pieces)) for sh in SHIFTS]
a1.axhline(0, color="#444", lw=0.8)
a1.fill_between(SHIFTS, np.array(y) - e, np.array(y) + e, color=BLUE, alpha=0.18, lw=0)
a1.plot(SHIFTS, y, color=BLUE, lw=2.2, marker="o", ms=6)
a1.set_xticks(SHIFTS, [f"{s:+d}" for s in SHIFTS], fontsize=11)
a1.set_xlabel("shift (semitones)", fontsize=11)
a1.set_ylabel("panel rating change (1–5 scale)", fontsize=11)
by_t = defaultdict(list)
for cell, v in cent.items():
    by_t[tonic[cell]].append(v)
ts = sorted(by_t)
a2.axhline(0, color="#444", lw=0.8)
a2.errorbar(range(len(ts)), [np.mean(by_t[t]) for t in ts],
            yerr=[2 * np.std(by_t[t]) / np.sqrt(len(by_t[t])) for t in ts],
            color=BLUE, lw=1.8, marker="o", ms=5, capsize=2)
a2.set_xticks(range(len(ts)), [PCN[t] for t in ts], fontsize=11)
a2.set_xlabel("target key", fontsize=11)
for ax in (a1, a2):
    ax.set_ylim(-0.3, 0.3)
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.suptitle("Q1: Do LLMs collectively rate keys differently?   Answer: no — every shift and every target key within ±0.05",
             fontsize=13, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "keyshift2_q1.png", dpi=150)
plt.close(fig)

# Q2 figure: per-judge deviation-profile spread vs permutation noise
rows2 = []
for j in judges:
    null = []
    for _ in range(2000):
        fake = defaultdict(list)
        for p in pieces:
            for s2, sh in zip(rng.permutation(SHIFTS), SHIFTS):
                fake[s2].append(dev[j][(p, sh)])
        null.append(np.std([np.mean(fake[sh]) for sh in SHIFTS]))
    rows2.append((j, ana[j]["spread"], float(np.percentile(null, 97.5)), ana[j]["p"]))
rows2.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9, 10))
ypos = range(len(rows2))
ax.barh(ypos, [r[2] for r in rows2], color="#e4e1d8", height=0.7, label="pure-noise range (95th pct)")
ax.plot([r[1] for r in rows2], ypos, "o", color=BLUE, ms=6, label="actual key-profile spread (vs panel)")
ax.set_yticks(ypos, [f"{r[0]}{' *' if r[3] < 0.05 else ''}" for r in rows2], fontsize=9)
ax.set_xlabel("spread of the judge's 11 key deviations from the panel", fontsize=11)
n_over = sum(1 for r in rows2 if r[3] < 0.05)
ax.set_title(f"Q2: Does any individual judge have its own key profile?\n"
             f"Answer: {n_over} of 40 non-flat at p<.05 (chance predicts ~2; none survive correction)",
             fontsize=13, loc="left")
ax.legend(loc="lower right", fontsize=10, frameon=False)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "keyshift2_q2.png", dpi=150)
plt.close(fig)

# Q3 table: favorite-key bonus on deviations, all 40 judges
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
rows3 = []
for j in judges:
    kc = keys_by_arm.get(j)
    if not kc:
        continue
    fav, nfav = kc.most_common(1)[0]
    own = np.array([dev[j][c] for c in dev[j] if tonic[c] == fav])
    oth = np.array([dev[j][c] for c in dev[j] if tonic[c] != fav])
    b = float(own.mean() - oth.mean())
    se = float(np.sqrt(own.var(ddof=1) / len(own) + oth.var(ddof=1) / len(oth)))
    rows3.append({"arm": j, "fav": PCN[fav], "share": nfav / sum(kc.values()),
                  "bonus": b, "ci": 2 * se, "n": int(len(own))})
rows3.sort(key=lambda r: -r["bonus"])
bs = np.array([r["bonus"] for r in rows3])
(OUT / "keyshift2.json").write_text(json.dumps({
    "n_pieces": len(pieces), "n_judges": len(judges),
    "n_ratings": sum(1 for v in ck.values() if v), "q2_sig": n_over,
    "pooled": [float(bs.mean()), float(2 * bs.std() / np.sqrt(len(bs)))],
    "q3_rows": rows3}, indent=1))
print(f"wrote keyshift2_q1.png, keyshift2_q2.png, keyshift2.json ({len(rows3)} q3 rows)")
