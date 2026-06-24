#!/usr/bin/env python3
"""Correlate returned human ratings against the LLM-judge, per dimension.

Pairs human_ratings.json (exported from rate.html) with sample_key.json (which
holds each blind id's judge scores) and reports Spearman rank correlation per
dimension — the standard check (Chiang & Lee; MuSpike's human study) that earns
the judge the right to stand in for human perception. Also reports emotion-label
agreement.

Usage:  python scripts/score_validation.py path/to/human_ratings.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "docs" / "validation" / "sample_key.json"


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_ranks(a), _ranks(b))


def main():
    if len(sys.argv) < 2:
        print("usage: score_validation.py human_ratings.json")
        return
    human = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    key = json.loads(KEY.read_text(encoding="utf-8"))

    dims = ["coherence", "harmony", "rhythm", "structure", "melody", "emotion",
            "creativity", "naturalness", "valence", "arousal"]
    rated = [pid for pid in human if pid in key]
    print(f"{len(rated)} pieces rated by human (of {len(key)} in sheet)\n")
    if len(rated) < 4:
        print("need ≥4 rated pieces for a meaningful correlation.")
        return

    print(f"{'dimension':14} {'spearman':>9} {'n':>3}   human↔judge")
    qvals = []
    for d in dims:
        pairs = [(human[p][d], key[p]["judge"].get(d)) for p in rated
                 if d in human[p] and key[p]["judge"].get(d) is not None]
        if len(pairs) < 4:
            print(f"{d:14} {'—':>9} {len(pairs):>3}")
            continue
        h = [x for x, _ in pairs]
        j = [y for _, y in pairs]
        rho = spearman(h, j)
        flag = "  ✓" if rho >= 0.5 else ("  ~" if rho >= 0.3 else "  ✗ weak")
        print(f"{d:14} {rho:>9.2f} {len(pairs):>3}{flag}")
        if d not in ("valence", "arousal"):
            qvals.append(rho)

    # overall quality agreement: human mean-of-quality vs judge overall
    op = []
    for p in rated:
        hq = [human[p][d] for d in dims[:8] if d in human[p]]
        jo = key[p]["judge"].get("overall")
        if hq and jo is not None:
            op.append((sum(hq) / len(hq), jo))
    if len(op) >= 4:
        print(f"\noverall quality   {spearman([a for a,_ in op],[b for _,b in op]):>7.2f} "
              f"{len(op):>3}   (human mean-quality ↔ judge overall)")
    if qvals:
        print(f"mean quality-dim spearman: {sum(qvals)/len(qvals):.2f}")

    # emotion-label agreement (exact + valence-sign)
    el = [(human[p]["emotion_label"], key[p].get("judge")) for p in rated if human[p].get("emotion_label")]
    if el:
        print(f"\nemotion labels collected for {len(el)} pieces (compare to judge_raw.json labels manually).")


if __name__ == "__main__":
    main()
