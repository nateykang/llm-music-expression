#!/usr/bin/env python3
"""Mode-bias analyses — does an LLM judge favor the tonal mode it composes in?

Continuation of the self-expression key bias (models write ~70% minor free-form)
into evaluation, tested where authorship can't confound it:

  1. GENERALIZATION (Bach). The 10-model panel blind-rates the 371 Bach chorales
     (human-composed, 195 major / 176 minor). Per judge, its major-vs-minor score
     deviation from the panel is correlated with its OWN major-writing rate on the
     generated corpus. r>0 = the disposition transfers to human music.

  2. MECHANISM (relative-key relabel). The same generated ABC pieces, note-
     identical but with K: swapped to the relative key, are re-judged. The paired
     score shift (relabeled - original) isolates how much of the mode bias is
     driven by READING the declared key vs perceiving mode from the notes.

    python scripts/analyze_mode_bias.py

Reads docs/analysis/judge_bach_raw.json, judge_relabel_raw.json,
judge_allmodels_raw.json, and the corpus features.csv files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ["coherence", "harmony", "rhythm", "structure", "melody",
           "emotion", "creativity", "naturalness"]


def qual(v):
    s = [v[k]["score"] for k in QUALITY if k in v]
    return sum(s) / len(s) if s else None


def own_major_rate():
    maj, tot = {}, {}
    for f in (ROOT / "docs/data").glob("*/features.csv"):
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("prompt") != "free-form":
                continue
            km = r.get("key_mode_best")
            if km in ("major", "minor"):
                m = r["model"]
                tot[m] = tot.get(m, 0) + 1
                maj[m] = maj.get(m, 0) + (km == "major")
    return {m: maj[m] / tot[m] for m in tot}


def perm_p(xs, ys, n=20000):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    r0 = np.corrcoef(xs, ys)[0, 1]
    rng = np.random.default_rng(0)
    hits = sum(abs(np.corrcoef(rng.permutation(xs), ys)[0, 1]) >= abs(r0) for _ in range(n))
    return r0, hits / n


def bach_generalization():
    raw = json.loads((ROOT / "docs/analysis/judge_bach_raw.json").read_text(encoding="utf-8"))
    om = own_major_rate()
    judges = sorted({j for p in raw for j in p["panel"]})
    print(f"1) GENERALIZATION — {len(raw)} Bach chorales, panel of {len(judges)}")
    print(f"   {'judge':20s} {'own maj%':>9s} {'favors-major on Bach':>21s}")
    pairs = []
    for J in judges:
        maj, minr = [], []
        for p in raw:
            if J not in p["panel"]:
                continue
            qJ = qual(p["panel"][J])
            peers = [qual(v) for k, v in p["panel"].items() if k != J]
            peers = [x for x in peers if x is not None]
            if qJ is None or not peers:
                continue
            (maj if p["key"].endswith("major") else minr).append(qJ - np.mean(peers))
        bias = np.mean(maj) - np.mean(minr)
        if J in om:
            pairs.append((om[J], bias))
        print(f"   {J:20s} {100 * om.get(J, float('nan')):>8.0f}% {bias:>+21.3f}")
    r, p = perm_p([a for a, _ in pairs], [b for _, b in pairs])
    print(f"   => r(own major-rate, favors-major) = {r:+.2f}  (perm p={p:.3f}, n={len(pairs)})\n")


def relabel_mechanism():
    relabel = json.loads((ROOT / "docs/analysis/judge_relabel_raw.json").read_text(encoding="utf-8"))
    orig = {}
    for p in json.loads((ROOT / "docs/analysis/judge_allmodels_raw.json").read_text(encoding="utf-8")):
        orig[(p["model"], p.get("mode"), p.get("title"),
              str(p.get("sample")), p.get("batch"))] = p["panel"]
    om = own_major_rate()
    judges = sorted({j for p in relabel for j in p["panel"]})
    print(f"2) MECHANISM — {len(relabel)} note-identical pieces, K: relabeled to the relative key")
    print(f"   {'judge':20s} {'own maj%':>9s} {'label->major effect':>20s}")
    pairs, allshift = [], []
    for J in judges:
        eff = []
        for p in relabel:
            k = (p["model"], p.get("mode"), p.get("title"), str(p.get("sample")), p.get("batch"))
            if k not in orig or J not in p["panel"] or J not in orig[k]:
                continue
            qr, qo = qual(p["panel"][J]), qual(orig[k][J])
            if qr is None or qo is None:
                continue
            toward_major = 1 if p["orig_key"].endswith("m") else -1
            eff.append((qr - qo) * toward_major)
        allshift += eff
        pairs.append((om.get(J, float("nan")), np.mean(eff)))
        print(f"   {J:20s} {100 * om.get(J, float('nan')):>8.0f}% {np.mean(eff):>+20.3f}")
    good = [(a, b) for a, b in pairs if a == a]
    r, p = perm_p([a for a, _ in good], [b for _, b in good])
    print(f"   => global label->major effect = {np.mean(allshift):+.3f} pts "
          f"(n={len(allshift)}); r(own major-rate, effect) = {r:+.2f} (perm p={p:.3f})")


if __name__ == "__main__":
    bach_generalization()
    relabel_mechanism()
