#!/usr/bin/env python3
"""Does each judge rate its OWN music higher? — rigorous self-preference analysis.

Builds the judge × author mean-quality matrix from the blind run (judge_raw.json,
off-diagonal) plus the self-judgments (self_judge.json, the diagonal), then reports
self-preference CORRECTED for each judge's general leniency. The raw self-gap is
confounded: a judge that's harsh on everyone will look "self-critical" even if it
treats itself exactly like everyone else. The corrected measure isolates whether a
judge is EXTRA favorable (or harsh) to itself beyond its overall tendency.

  leniency(J)      = mean over non-self pieces of [ J's score - co-judges' score ]
  raw self-gap(J)  = mean over J's own pieces of  [ J's score - co-judges' score ]
  self-preference  = raw self-gap(J) - leniency(J)     (>0 = favors itself specifically)

Usage:  python scripts/self_preference.py
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

PANEL = ["gpt-5.5", "gemini-2.5-pro", "opus-4.8"]


def q(verdict: dict) -> float | None:
    vs = [verdict[k]["score"] for k in QUALITY_KEYS if k in verdict]
    return mean(vs) if vs else None


def main():
    raw = json.loads((ROOT / "docs/analysis/judge_raw.json").read_text(encoding="utf-8"))
    raw = [p for p in raw if p["prompt"] == "free-form"]
    self_rows = json.loads((ROOT / "docs/analysis/self_judge.json").read_text(encoding="utf-8"))

    # score_by[judge][author] = list of per-piece mean-quality
    score_by = defaultdict(lambda: defaultdict(list))
    raw_by_piece = {}
    for p in raw:
        author = p["model"]
        raw_by_piece[(author, p["title"])] = p["panel"]
        for j, verdict in p["panel"].items():
            qv = q(verdict)
            if qv is not None:
                score_by[j][author].append(qv)
    for r in self_rows:  # diagonal
        score_by[r["model"]][r["model"]].append(r["self"])

    authors = ["gpt-5.5", "gemini-2.5-pro", "opus-4.8", "opus-4.8-thinking", "sonnet-4.6",
               "deepseek-v4-pro", "gpt-4.1", "grok-4.3", "qwen3-max", "llama-4-maverick"]

    # ---- judge × author matrix ----
    print("=== JUDGE × AUTHOR mean-quality matrix (diagonal = self, in [brackets]) ===")
    print(f"{'judge \\\\ author':16}" + "".join(f"{a[:8]:>9}" for a in authors))
    for j in PANEL:
        cells = []
        for a in authors:
            vals = score_by[j].get(a, [])
            if not vals:
                cells.append(f"{'·':>9}")
            else:
                s = f"{mean(vals):.2f}"
                cells.append(f"{('['+s+']') if a == j else s:>9}")
        print(f"{j:16}" + "".join(cells))

    # ---- leniency + raw self-gap + corrected self-preference ----
    # leniency(J): J's deviation from co-judges on non-self pieces (per piece)
    leniency = {}
    for j in PANEL:
        devs = []
        for p in raw:
            panel = p["panel"]
            if j not in panel:
                continue
            others = [q(v) for jj, v in panel.items() if jj != j]
            others = [x for x in others if x is not None]
            sj = q(panel[j])
            if sj is not None and others:
                devs.append(sj - mean(others))
        leniency[j] = mean(devs) if devs else 0.0

    # raw self-gap from self_judge (self - peers on own pieces)
    raw_gap = {}
    for j in PANEL:
        gs = [r["self"] - r["others"] for r in self_rows if r["model"] == j]
        raw_gap[j] = mean(gs) if gs else None

    print("\n=== SELF-PREFERENCE, corrected for general leniency ===")
    print(f"{'judge':16} {'n':>3} {'raw self-gap':>13} {'leniency':>10} {'corrected':>11}")
    for j in PANEL:
        n = sum(1 for r in self_rows if r["model"] == j)
        if raw_gap[j] is None:
            continue
        corr = raw_gap[j] - leniency[j]
        print(f"{j:16} {n:>3} {raw_gap[j]:>+13.2f} {leniency[j]:>+10.2f} {corr:>+11.2f}")
    print("  raw self-gap = self - co-judges on own pieces")
    print("  leniency     = how the judge deviates from co-judges on EVERYONE ELSE")
    print("  corrected    = self-specific bias (>0 favors itself beyond its general tendency)")

    # ---- per-dimension self-gap ----
    print("\n=== PER-DIMENSION self-gap (self - peers on own pieces) ===")
    print(f"{'dimension':13}" + "".join(f"{j[:8]:>10}" for j in PANEL))
    perdim = {j: defaultdict(list) for j in PANEL}
    for r in self_rows:
        author = r["model"]
        panel = raw_by_piece.get((author, r["title"]), {})
        peers = {jj: v for jj, v in panel.items() if jj != author}
        for d in QUALITY_KEYS:
            sd = (r.get("self_dims") or {}).get(d)
            pv = [v[d]["score"] for v in peers.values() if d in v]
            if sd is not None and pv:
                perdim[author][d].append(sd - mean(pv))
    for d in QUALITY_KEYS:
        row = f"{d:13}"
        for j in PANEL:
            ds = perdim[j].get(d, [])
            row += f"{(mean(ds) if ds else 0):>+10.2f}"
        print(row)


if __name__ == "__main__":
    main()
