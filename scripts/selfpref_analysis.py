#!/usr/bin/env python3
"""Part 2 self-preference analysis on the v3 express-yourself matrix.

Every model judges every piece (self included). For judge J on a piece by author
A: gap = q_J - mean(q_O, O != J) where q is the mean of the 8 quality dims.
  leniency(J)        = mean gap over pieces by OTHER families
  strict self (J)    = mean gap over J's own pieces          - leniency
  sibling (J)        = mean gap over the +-thinking sibling's - leniency
  family self (J)    = pooled own + sibling                   - leniency
Bootstrap (pieces) 95% CIs; competence = r(q_J, leave-J-out panel mean).
Writes docs/analysis/selfpref_v3.json (+ matrix PNG).

    python scripts/selfpref_analysis.py [raw_json]
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_music.judge import QUALITY_KEYS  # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "part2_run/analysis/part2_selfpref_raw.json"
OUT = ROOT / "docs/analysis/selfpref_v3.json"
random.seed(20260901); np.random.seed(20260901)
B = 1000


def family(arm: str) -> str:
    return arm[:-9] if arm.endswith("-thinking") else arm


def q(v: dict) -> float | None:
    s = [v[k]["score"] for k in QUALITY_KEYS if k in v]
    return mean(s) if len(s) == len(QUALITY_KEYS) else None


def ci(vals: np.ndarray, lo=2.5, hi=97.5):
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


raw = json.loads(SRC.read_text())
judges = sorted({j for p in raw for j in p["panel"]})
# per piece: author, mode, {judge: q}
pieces = []
for p in raw:
    qs = {j: q(v) for j, v in p["panel"].items()}
    qs = {j: x for j, x in qs.items() if x is not None}
    if len(qs) >= 30:
        pieces.append({"author": p["model"], "mode": "codegen" if p.get("mode") == "codegen" else "abc",
                       "key": f"{p['batch']}|{p['model']}|{p.get('mode','')}|{p.get('sample',0)}", "q": qs})
print(f"{len(pieces)} pieces, {len(judges)} judges", flush=True)

# gaps: gap[J][piece_idx] = q_J - mean(others)
gaps = {J: {} for J in judges}
for i, pc in enumerate(pieces):
    tot = sum(pc["q"].values()); n = len(pc["q"])
    for J, x in pc["q"].items():
        gaps[J][i] = x - (tot - x) / (n - 1)

authors_by_piece = np.array([pc["author"] for pc in pieces])
modes_by_piece = np.array([pc["mode"] for pc in pieces])
results, matrix = {}, {}
panel_mean_by_author = defaultdict(list)
for pc in pieces:
    panel_mean_by_author[pc["author"]].append(mean(pc["q"].values()))

for J in judges:
    fam = family(J)
    idx = np.array(sorted(gaps[J])); g = np.array([gaps[J][i] for i in idx])
    auth = authors_by_piece[idx]; mode = modes_by_piece[idx]
    own = auth == J
    sib = np.array([family(a) == fam for a in auth]) & ~own
    oth = np.array([family(a) != fam for a in auth])
    def stats(mask_self, mask_len=None):
        mask_len = oth if mask_len is None else mask_len
        if mask_self.sum() < 3 or mask_len.sum() < 10:
            return None
        s, l = g[mask_self], g[mask_len]
        est = s.mean() - l.mean()
        boot = np.array([np.random.choice(s, len(s)).mean() - np.random.choice(l, len(l)).mean() for _ in range(B)])
        return {"n_self": int(mask_self.sum()), "raw_gap": float(s.mean()), "leniency": float(l.mean()),
                "corrected": float(est), "ci95": ci(boot)}
    r = {"family": fam, "thinking": J.endswith("-thinking"), "n_pieces": int(len(g)),
         "strict_self": stats(own), "sibling": stats(sib), "family_self": stats(own | sib),
         "by_mode": {m: stats(own & (mode == m), oth & (mode == m)) for m in ("abc", "codegen")},
         "mean_given": float(np.mean([pieces[i]["q"][J] for i in idx]))}
    # competence: r(q_J, leave-J-out mean)
    xs = np.array([pieces[i]["q"][J] for i in idx])
    ys = np.array([(sum(pieces[i]["q"].values()) - pieces[i]["q"][J]) / (len(pieces[i]["q"]) - 1) for i in idx])
    r["competence_r"] = float(np.corrcoef(xs, ys)[0, 1])
    results[J] = r
    # judge x author matrix of mean q
    row = defaultdict(list)
    for i in idx: row[authors_by_piece[i]].append(pieces[i]["q"][J])
    matrix[J] = {a: float(mean(v)) for a, v in row.items()}

# thinking effect within family (corrected strict-self, thinking - base)
fam_pairs = {}
for J, r in results.items():
    if r["strict_self"]:
        fam_pairs.setdefault(r["family"], {})["thinking" if r["thinking"] else "base"] = r["strict_self"]["corrected"]
deltas = [v["thinking"] - v["base"] for v in fam_pairs.values() if "thinking" in v and "base" in v]
authors_quality = {a: float(mean(v)) for a, v in panel_mean_by_author.items()}
summary = {
    "n_pieces": len(pieces), "n_judges": len(judges), "n_verdicts": sum(len(g) for g in gaps.values()),
    "mean_corrected_strict_self": float(mean(r["strict_self"]["corrected"] for r in results.values() if r["strict_self"])),
    "n_positive_strict_self": sum(1 for r in results.values() if r["strict_self"] and r["strict_self"]["ci95"][0] > 0),
    "n_negative_strict_self": sum(1 for r in results.values() if r["strict_self"] and r["strict_self"]["ci95"][1] < 0),
    "thinking_minus_base_strict_self": {"mean": float(mean(deltas)) if deltas else None, "n_families": len(deltas),
                                        "deltas": {f: v["thinking"] - v["base"] for f, v in fam_pairs.items() if "thinking" in v and "base" in v}},
}
# cross-judge correlates of self-preference (over arms with a strict-self estimate)
_arms = [J for J in judges if results[J]["strict_self"]]
_sb = np.array([results[J]["strict_self"]["corrected"] for J in _arms])
_r = lambda xs: float(np.corrcoef(np.array(xs, dtype=float), _sb)[0, 1])
_pairs = [(v["base"], v["thinking"]) for v in fam_pairs.values() if "thinking" in v and "base" in v]
summary.update({
    "competence_vs_selfbias_r": _r([results[J]["competence_r"] for J in _arms]),
    "leniency_vs_selfbias_r": _r([results[J]["strict_self"]["leniency"] for J in _arms]),
    "own_quality_vs_selfbias_r": _r([authors_quality[J] for J in _arms]),
    "base_vs_thinking_selfbias_r": float(np.corrcoef(*zip(*_pairs))[0, 1]) if len(_pairs) > 2 else None,
    "low_competence_judges": {J: round(results[J]["competence_r"], 2) for J in _arms if results[J]["competence_r"] < 0.75},
})
OUT.write_text(json.dumps({"summary": summary, "judges": results, "matrix": matrix,
                           "author_quality": authors_quality, "judges_order": judges}, indent=1))
print(json.dumps(summary, indent=1))
print("\ncorrected strict-self bias (top/bottom 6):")
rk = sorted(((r["strict_self"]["corrected"], J) for J, r in results.items() if r["strict_self"]), reverse=True)
for c, J in rk[:6] + [("...", "...")] + rk[-6:]:
    if J == "...": print("   ..."); continue
    s = results[J]["strict_self"]; print(f"   {J:26} {c:+.3f}  CI [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]  competence r={results[J]['competence_r']:.2f}")
print(f"\nwrote {OUT}")
