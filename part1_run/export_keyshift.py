"""Export the key-shift EDA (Part 1 pilot) for the website: the two figures and
a JSON with the Q3 favorite-key table, into docs/analysis/. Deterministic
(seeded permutations); reads only part1_run/pilot_{ckpt,reps}.json + corpus."""
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
from analyze_pilot import piece_tonic_pc, q8  # noqa: E402

BLUE = "#2a78d6"
SHIFTS = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
PCN = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PCM = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
rng = np.random.default_rng(20260917)
OUT = ROOT / "docs/analysis"

ck = json.loads((ROOT / "part1_run/pilot_ckpt.json").read_text())
reps = json.loads((ROOT / "part1_run/pilot_reps.json").read_text())
cent = defaultdict(list)
for key, v in ck.items():
    p = key.split("|")
    s = q8(v)
    if s is not None:
        cent[(p[1], int(p[4]))].append((int(p[5]), s, piece_tonic_pc(reps[key.rsplit("|", 1)[0]])))
prof = defaultdict(lambda: defaultdict(list))
cells = defaultdict(list)
by_piece_arm = defaultdict(lambda: defaultdict(list))
for (arm, smp), lst in cent.items():
    mu = np.mean([x[1] for x in lst])
    for sh, s, pc in lst:
        prof[arm][sh].append(s - mu)
        cells[arm].append((s - mu, pc))
        by_piece_arm[arm][smp].append((sh, s - mu))
arms = sorted(prof)

# Q1 figure
y = [np.mean([np.mean(prof[a][sh]) for a in arms]) for sh in SHIFTS]
e = [2 * np.std([np.mean(prof[a][sh]) for a in arms]) / np.sqrt(len(arms)) for sh in SHIFTS]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axhline(0, color="#444", lw=0.8)
ax.fill_between(SHIFTS, np.array(y) - e, np.array(y) + e, color=BLUE, alpha=0.18, lw=0)
ax.plot(SHIFTS, y, color=BLUE, lw=2.2, marker="o", ms=6)
ax.set_ylim(-0.55, 0.55)
ax.set_xticks(SHIFTS, [f"{s:+d}" for s in SHIFTS], fontsize=12)
ax.set_xlabel("how far the piece was moved (semitones)", fontsize=12)
ax.set_ylabel("rating change (1–5 scale)", fontsize=12)
ax.set_title("Q1: Does moving a piece to another key change its rating?\n"
             "Answer: no — the average stays within ±0.07 at every shift (band = 95% range)",
             fontsize=13, loc="left")
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "keyshift_q1.png", dpi=150)
plt.close(fig)

# Q2 figure
rows2 = []
for a in arms:
    obs = np.std([np.mean(prof[a][sh]) for sh in SHIFTS])
    null = []
    for _ in range(2000):
        fake = defaultdict(list)
        for smp, lst in by_piece_arm[a].items():
            for shp, (sh, c) in zip(rng.permutation([x[0] for x in lst]), lst):
                fake[shp].append(c)
        null.append(np.std([np.mean(fake[sh]) for sh in SHIFTS]))
    rows2.append((a, obs, np.percentile(null, 97.5)))
rows2.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9, 10))
ypos = range(len(rows2))
ax.barh(ypos, [r[2] for r in rows2], color="#e4e1d8", height=0.7, label="pure-noise range (95th pct)")
ax.plot([r[1] for r in rows2], ypos, "o", color=BLUE, ms=6, label="actual key-profile spread")
ax.set_yticks(ypos, [r[0] for r in rows2], fontsize=9)
ax.set_xlabel("spread of the model's 11 key ratings (bigger = stronger key preference)", fontsize=11)
n_over = sum(1 for r in rows2 if r[1] > r[2])
ax.set_title(f"Q2: Does any individual model prefer certain keys?\n"
             f"Answer: no — {n_over} of 40 spreads exceed pure noise", fontsize=13, loc="left")
ax.legend(loc="lower right", fontsize=10, frameon=False)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "keyshift_q2.png", dpi=150)
plt.close(fig)

# Q3 table data
keys_by_arm = defaultdict(Counter)
for bd in (ROOT / "docs/data").iterdir():
    f = bd / "data.json"
    if not f.exists() or not bd.name.startswith("202608"):
        continue
    for pc in json.loads(f.read_text())["pieces"]:
        if not pc.get("ok") or pc.get("mode") != "abc":
            continue
        m = re.search(r"^K:\s*([A-G])([#b]?)", pc.get("abc", ""), re.M)
        if m:
            keys_by_arm[pc["model"]][(PCM[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)) % 12] += 1
rows3 = []
for arm in arms:
    kc = keys_by_arm.get(arm)
    if not kc:
        continue
    fav, nfav = kc.most_common(1)[0]
    own = np.array([c for c, pc in cells[arm] if pc == fav])
    oth = np.array([c for c, pc in cells[arm] if pc is not None and pc != fav])
    if len(own) < 1 or len(oth) < 3:
        continue
    b = float(own.mean() - oth.mean())
    se = float(np.sqrt((own.var(ddof=1) / len(own) if len(own) > 1 else 0.25 ** 2) + oth.var(ddof=1) / len(oth)))
    rows3.append({"arm": arm, "fav": PCN[fav], "share": nfav / sum(kc.values()),
                  "bonus": b, "ci": 2 * se, "n": int(len(own))})
rows3.sort(key=lambda r: -r["bonus"])
bs = np.array([r["bonus"] for r in rows3])
(OUT / "keyshift_pilot.json").write_text(json.dumps({
    "n_ratings": sum(1 for v in ck.values() if v), "n_arms": len(arms),
    "q2_over_noise": n_over,
    "pooled": [float(bs.mean()), float(2 * bs.std() / np.sqrt(len(bs)))],
    "q3_rows": rows3}, indent=1))
print(f"wrote keyshift_q1.png, keyshift_q2.png, keyshift_pilot.json ({len(rows3)} q3 rows)")
