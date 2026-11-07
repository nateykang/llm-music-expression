#!/usr/bin/env python3
"""Aggregate the new-models run (fable-5.1, opus-5.5, gpt-6-astra, gemini-4-argon,
gpt-6.1-sol) for docs/v3/new_models.html: independent-panel quality for every arm,
self-preference against two baselines, and musical defaults per new arm.

    python scripts/new_models_analysis.py [raw_json] [selfpref_json]

selfpref_json comes from scripts/selfpref_analysis.py on the same raw file.
"""
import collections as C
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_music.judge import QUALITY_KEYS  # noqa: E402

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "part4_run/analysis/part4_selfpref_raw.json"
SP = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "part4_run/analysis/selfpref_part4.json"
OUT = ROOT / "docs/analysis/new_models_v3.json"
DATA = ROOT / "docs/data"
NEW_BATCHES = ["20261106_031646__models_8_prompts_3", "20261106_040840__models_8_prompts_3",
               "20261001_121052__models_2_prompts_3", "20261001_121202__models_2_prompts_3"]
V3_BATCHES = ["20260819_201530__models_40_prompts_3", "20260819_203512__models_2_prompts_3",
              "20260820_061907__models_42_prompts_3"]
FOCUS = [f"{m}{s}" for m in ("gpt-6-astra", "gpt-6.1-sol", "gemini-4-argon", "opus-5.5", "fable-5.1")
         for s in ("", "-thinking")]
B = 2000
np.random.seed(20261107)


def fam(a):
    return a.removesuffix("-thinking")


def q(v):
    s = [v[k]["score"] for k in QUALITY_KEYS if k in v]
    return mean(s) if len(s) == len(QUALITY_KEYS) else None


def boot_mean(vals):
    vals = np.array(vals)
    b = vals[np.random.randint(0, len(vals), (B, len(vals)))].mean(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def boot_diff(s, l):
    s, l = np.array(s), np.array(l)
    d = (s[np.random.randint(0, len(s), (B, len(s)))].mean(1)
         - l[np.random.randint(0, len(l), (B, len(l)))].mean(1))
    return float(s.mean() - l.mean()), [float(x) for x in np.percentile(d, [2.5, 97.5])]


raw = json.loads(RAW.read_text())
sp = json.loads(SP.read_text())
judges = sp["judges_order"]
pieces = []
for r in raw:
    qs = {j: x for j, x in ((j, q(v)) for j, v in r["panel"].items()) if x is not None}
    pieces.append({"a": r["model"], "m": r["mode"], "panel": r["panel"], "qs": qs, "new": r["model"] in FOCUS})


def indep(p):
    return mean(x for j, x in p["qs"].items() if fam(j) != fam(p["a"]))


authors = {}
for a in sorted({p["a"] for p in pieces}):
    ps = [p for p in pieces if p["a"] == a]
    vals = [indep(p) for p in ps]
    labels = C.Counter(v.get("emotion_label") for p in ps for j, v in p["panel"].items() if fam(j) != fam(a))
    tot = sum(labels.values())
    authors[a] = {
        "n": len(ps), "q": mean(vals), "ci": boot_mean(vals),
        "abc": mean(indep(p) for p in ps if p["m"] == "abc"),
        "codegen": mean(indep(p) for p in ps if p["m"] == "codegen"),
        "dims": {k: mean(mean(v[k]["score"] for j, v in p["panel"].items() if fam(j) != fam(a) and k in v)
                         for p in ps) for k in QUALITY_KEYS},
        "labels": [[l, c / tot] for l, c in labels.most_common(3)],
    }
order = sorted(authors, key=lambda a: -authors[a]["q"])
for i, a in enumerate(order):
    authors[a]["rank"] = i + 1

selfp = {}
for J in judges:
    own, recent, old = [], [], []
    for p in pieces:
        if J not in p["qs"]:
            continue
        g = p["qs"][J] - mean(x for j, x in p["qs"].items() if j != J)
        if p["a"] == J:
            own.append(g)
        elif fam(p["a"]) != fam(J):
            (recent if p["new"] else old).append(g)
    va, va_ci = boot_diff(own, recent + old)
    vr, vr_ci = boot_diff(own, recent)
    selfp[J] = {"n_own": len(own), "own_gap": mean(own), "recent_gap": mean(recent), "old_gap": mean(old),
                "vs_all": va, "vs_all_ci": va_ci, "vs_recent": vr, "vs_recent_ci": vr_ci,
                "mean_given": sp["judges"][J]["mean_given"], "competence": sp["judges"][J]["competence_r"]}

matrix = {J: {a: mean(p["qs"][J] for p in pieces if p["a"] == a and J in p["qs"]) for a in FOCUS} for J in FOCUS}


def read_features(batches):
    rows = []
    for b in batches:
        with open(DATA / b / "features.csv", newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def num(rows, k):
    out = []
    for r in rows:
        try:
            out.append(float(r[k]))
        except (KeyError, ValueError):
            pass
    return out


def feat_summary(rows):
    modes = [r.get("key_mode_best") for r in rows if r.get("key_mode_best")]
    keys = C.Counter(f"{r['key_declared_tonic']} {r['key_declared_mode']}" for r in rows if r.get("key_declared_tonic"))
    return {"n": len(rows), "minor": sum(m == "minor" for m in modes) / len(modes) if modes else None,
            "top_key": keys.most_common(1)[0] if keys else None,
            "tempo": median(num(rows, "tempo_bpm")) if num(rows, "tempo_bpm") else None,
            **{k: median(num(rows, k)) for k in ("n_notes", "length_seconds", "n_instruments", "n_voices",
                                                 "pitch_class_entropy", "polyphony", "dynamic_span",
                                                 "structureness", "rhythm_entropy", "valence", "arousal")}}


frows = read_features(NEW_BATCHES)
features = {a: feat_summary([r for r in frows if r["model"] == a]) for a in FOCUS}
v3rows = read_features(V3_BATCHES)
features["v3 corpus (42 arms)"] = feat_summary(v3rows)

gen = {}
for b in NEW_BATCHES:
    for pc in json.loads((DATA / b / "data.json").read_text())["pieces"]:
        g = gen.setdefault(pc["model"], {"abc": [0, 0, 0], "codegen": [0, 0, 0]})[pc["mode"]]
        g[0] += 1
        g[1] += bool(pc.get("ok"))
        g[2] += pc.get("attempts", 1)

total = sum(len(p["panel"]) for p in pieces)
OUT.write_text(json.dumps({
    "focus": FOCUS, "order": order, "authors": authors, "selfp": selfp, "matrix": matrix, "features": features,
    "generation": gen, "by_trait": {j: sp["by_trait"][j] for j in FOCUS},
    "n_pieces": len(pieces), "n_judges": len(judges), "n_verdicts": total, "expected": len(pieces) * len(judges),
}, indent=1))
print(f"wrote {OUT}: {len(pieces)} pieces, {len(judges)} judges, {total} verdicts")
