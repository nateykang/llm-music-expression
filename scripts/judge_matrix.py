#!/usr/bin/env python3
"""All-judges self-bias study: judge × author matrix + competence + corrected self-bias.

With every model judging every piece (judge_allmodels_raw.json, run with
exclude_self=False), this reports:
  1. the judge × author mean-quality matrix (diagonal = self),
  2. each judge's COMPETENCE = how well its scores track the leave-it-out consensus
     (a model that doesn't track the panel is an unreliable critic; its self-bias
     is uninterpretable),
  3. each model's leniency-corrected SELF-BIAS (favoring itself beyond its general
     tendency), and
  4. the key question: does self-bias track (in)competence — do weaker judges
     favor themselves more?

Usage:  python scripts/judge_matrix.py [raw_json]   (default judge_allmodels_raw.json)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_music.judge import QUALITY_KEYS  # noqa: E402


def q(verdict: dict) -> float | None:
    vs = [verdict[k]["score"] for k in QUALITY_KEYS if k in verdict]
    return mean(vs) if vs else None


def pearson(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = mean(a), mean(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def main():
    src = ROOT / "docs/analysis" / (sys.argv[1] if len(sys.argv) > 1 else "judge_allmodels_raw.json")
    raw = [p for p in json.loads(src.read_text(encoding="utf-8")) if p["prompt"] == "free-form"]
    print(f"{len(raw)} free-form pieces from {src.name}\n")

    judges = sorted({j for p in raw for j in p["panel"]})
    authors_seen = defaultdict(int)
    for p in raw:
        authors_seen[p["model"]] += 1
    authors = sorted(authors_seen, key=lambda a: -authors_seen[a])

    # piece-level per-judge quality
    pieces = []  # (author, {judge: q})
    for p in raw:
        qd = {j: q(v) for j, v in p["panel"].items() if q(v) is not None}
        if qd:
            pieces.append((p["model"], qd))

    M = defaultdict(lambda: defaultdict(list))
    for author, qd in pieces:
        for j, qv in qd.items():
            M[j][author].append(qv)

    # ---- 1. matrix ----
    print("=== JUDGE × AUTHOR mean-quality ([self] on diagonal) ===")
    hdr = "judge \\ author".ljust(16) + "".join(f"{a.split('-')[0][:7]:>8}" for a in authors)
    print(hdr)
    for j in judges:
        row = j.ljust(16)
        for a in authors:
            vals = M[j].get(a, [])
            if not vals:
                row += f"{'·':>8}"
            else:
                s = f"{mean(vals):.2f}"
                row += f"{('['+s+']') if a == j else s:>8}"
        print(row)

    # ---- 2. competence: corr(J, leave-J-out consensus) ----
    print("\n=== JUDGE COMPETENCE (how well it tracks the panel) ===")
    print(f"{'judge':16} {'corr':>6} {'mean':>6} {'n':>4}")
    comp = {}
    for j in judges:
        xs, ys = [], []
        for author, qd in pieces:
            if j not in qd:
                continue
            others = [v for jj, v in qd.items() if jj != j]
            if others:
                xs.append(qd[j])
                ys.append(mean(others))
        r = pearson(xs, ys)
        comp[j] = r
        lvl = mean(xs) if xs else float("nan")
        print(f"{j:16} {r:>6.2f} {lvl:>6.2f} {len(xs):>4}")
    print("  corr = Pearson(judge's score, mean of all OTHER judges) — higher = more reliable critic")

    # ---- 3. leniency-corrected self-bias ----
    print("\n=== SELF-BIAS, leniency-corrected ===")
    print(f"{'model':16} {'n':>4} {'raw gap':>8} {'leniency':>9} {'corrected':>10}")
    selfbias = {}
    for m in judges:
        own_gaps, other_gaps = [], []
        for author, qd in pieces:
            if m not in qd:
                continue
            others = [v for jj, v in qd.items() if jj != m]
            if not others:
                continue
            gap = qd[m] - mean(others)
            (own_gaps if author == m else other_gaps).append(gap)
        if not own_gaps:
            continue
        raw_gap, leniency = mean(own_gaps), (mean(other_gaps) if other_gaps else 0.0)
        corrected = raw_gap - leniency
        selfbias[m] = corrected
        print(f"{m:16} {len(own_gaps):>4} {raw_gap:>+8.2f} {leniency:>+9.2f} {corrected:>+10.2f}")
    print("  corrected > 0 = favors its OWN pieces beyond how it treats everyone else")

    # ---- 4. does self-bias track competence? ----
    common = [m for m in selfbias if m in comp and comp[m] == comp[m]]
    if len(common) >= 3:
        cs = [comp[m] for m in common]
        sb = [selfbias[m] for m in common]
        print(f"\n=== competence vs self-bias: Pearson r = {pearson(cs, sb):+.2f} (n={len(common)}) ===")
        print("  (negative r = LESS competent judges favor themselves MORE)")
        print(f"  {'model':16} {'competence':>11} {'self-bias':>10}")
        for m in sorted(common, key=lambda x: comp[x]):
            print(f"  {m:16} {comp[m]:>11.2f} {selfbias[m]:>+10.2f}")


if __name__ == "__main__":
    main()
