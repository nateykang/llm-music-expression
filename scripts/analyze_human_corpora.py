#!/usr/bin/env python3
"""First analyses of the human-corpora genre experiment (judge_human_raw.json).

A. Bach comparability: the 150 bach-arm pieces were already judged in the
   original 371-chorale run with the ORIGINAL rubric. Same pieces, same
   judges, same dimensions -> did adding enjoyment/interest/beauty/familiarity
   and the origin guess shift the original dimensions?
B. Familiarity mediation: does self-reported familiarity predict quality
   within arms, and does controlling it shrink the between-arm gaps?
C. Per-judge x arm profiles (leniency-corrected).
D. Mode bias per arm: does own major-writing rate predict favors-major bias
   inside each corpus?
E. Sample origin guesses for the misread arms.

    python scripts/analyze_human_corpora.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_music.judge import QUALITY_KEYS  # noqa: E402
from llm_music.report_common import SHORT  # noqa: E402

np.random.seed(0)
QUAL = QUALITY_KEYS
NEW = ["enjoyment", "interest", "beauty", "familiarity"]


def qual(v):
    s = [v[k]["score"] for k in QUAL if k in v]
    return mean(s) if s else None


def main():
    analysis = ROOT / "docs/analysis"
    rows = json.loads((analysis / "judge_human_raw.json").read_text(encoding="utf-8"))
    old_bach = json.loads((analysis / "judge_bach_raw.json").read_text(encoding="utf-8"))
    arms = sorted({r["arm"] for r in rows})
    judges = sorted({j for r in rows for j in r["panel"]})

    # ---- A. Bach comparability ---------------------------------------------------
    print("== A · Bach: original rubric vs extended rubric (same pieces/judges) ==")
    okey = Counter((r["title"], r["key"]) for r in old_bach)
    nkey = Counter((r["title"], r["key"]) for r in rows if r["arm"] == "bach")
    unique = {k for k in nkey if nkey[k] == 1 and okey.get(k) == 1}
    oldby = {(r["title"], r["key"]): r for r in old_bach}
    pairs, per_dim = [], defaultdict(list)
    old_dev, new_dev = defaultdict(lambda: ([], [])), None
    for r in rows:
        k = (r["title"], r["key"])
        if r["arm"] != "bach" or k not in unique:
            continue
        o = oldby[k]
        for J in judges:
            vo, vn = o["panel"].get(J), r["panel"].get(J)
            if not vo or not vn:
                continue
            qo, qn = qual(vo), qual(vn)
            if qo is None or qn is None:
                continue
            pairs.append((qo, qn))
            for d in QUAL:
                if d in vo and d in vn:
                    per_dim[d].append(vn[d]["score"] - vo[d]["score"])
    qo = np.array([a for a, _ in pairs])
    qn = np.array([b for _, b in pairs])
    print(f"{len(unique)} unambiguous (title,key) pieces, {len(pairs)} paired verdicts")
    print(f"mean quality old {qo.mean():.3f} -> new {qn.mean():.3f} "
          f"(shift {qn.mean() - qo.mean():+.3f}), corr r = {np.corrcoef(qo, qn)[0, 1]:.2f}")
    print("per-dimension shift (new - old): "
          + ", ".join(f"{d} {mean(v):+.2f}" for d, v in sorted(per_dim.items())))

    # per-judge favors-major bias, old vs new run (same matched pieces)
    def major_bias(source, panel_of, key_of):
        out = {}
        for J in judges:
            mj, mn = [], []
            for r in source:
                panel = panel_of(r)
                if J not in panel:
                    continue
                qJ = qual(panel[J])
                peers = [x for x in (qual(v) for j, v in panel.items() if j != J)
                         if x is not None]
                if qJ is None or not peers:
                    continue
                (mj if key_of(r).endswith("major") else mn).append(qJ - mean(peers))
            if mj and mn:
                out[J] = mean(mj) - mean(mn)
        return out

    nb = [r for r in rows if r["arm"] == "bach"]
    b_old = major_bias(old_bach, lambda r: r["panel"], lambda r: r["key"])
    b_new = major_bias(nb, lambda r: r["panel"], lambda r: r["key"])
    common = [J for J in judges if J in b_old and J in b_new]
    r_bias = np.corrcoef([b_old[J] for J in common], [b_new[J] for J in common])[0, 1]
    print(f"per-judge favors-major bias, old vs new run: r = {r_bias:+.2f} "
          f"(n = {len(common)} judges)")
    print("  " + "  ".join(f"{SHORT.get(J, J)} {b_old[J]:+.2f}->{b_new[J]:+.2f}"
                           for J in common))

    # ---- B. familiarity mediation -------------------------------------------------
    print("\n== B · Familiarity: within-arm correlation and arm-gap mediation ==")
    # verdict-level records
    recs = []  # (arm, judge, familiarity, quality)
    for r in rows:
        for J, v in r["panel"].items():
            q = qual(v)
            if q is None or "familiarity" not in v:
                continue
            recs.append((r["arm"], J, v["familiarity"]["score"], q))
    print(f"{'arm':<12}{'r(familiarity, quality) within arm':>36}")
    for a in arms:
        # judge-centered within the arm so leniency doesn't inflate the corr
        xs, ys = [], []
        for J in judges:
            sub = [(f, q) for a2, j, f, q in recs if a2 == a and j == J]
            if len(sub) < 10:
                continue
            mf = mean(f for f, _ in sub)
            mq = mean(q for _, q in sub)
            xs += [f - mf for f, _ in sub]
            ys += [q - mq for _, q in sub]
        print(f"{a:<12}{np.corrcoef(xs, ys)[0, 1]:>36.2f}")

    # OLS: quality ~ arm dummies + judge dummies (+ familiarity); bach = baseline
    arm_i = {a: i for i, a in enumerate(arms)}
    jud_i = {j: i for i, j in enumerate(judges)}
    base = arm_i["bach"]
    A_cols = len(arms) - 1

    def fit(with_fam):
        ncol = A_cols + len(judges) + (1 if with_fam else 0)
        Xm = np.zeros((len(recs), ncol))
        y = np.zeros(len(recs))
        for i, (a, j, f, q) in enumerate(recs):
            ai = arm_i[a]
            if ai != base:
                Xm[i, ai - (1 if ai > base else 0)] = 1
            Xm[i, A_cols + jud_i[j]] = 1
            if with_fam:
                Xm[i, -1] = f
            y[i] = q
        coef = np.linalg.lstsq(Xm, y, rcond=None)[0]
        gaps = {}
        for a in arms:
            if a == "bach":
                continue
            ai = arm_i[a]
            gaps[a] = coef[ai - (1 if ai > base else 0)]
        return gaps, (coef[-1] if with_fam else None)

    g0, _ = fit(False)
    g1, slope = fit(True)
    print(f"\narm gap vs bach (points), raw -> familiarity-adjusted "
          f"(familiarity slope {slope:+.2f}/point):")
    for a in sorted(g0, key=g0.get):
        print(f"  {a:<12}{g0[a]:+.3f} -> {g1[a]:+.3f}")

    # ---- C. per-judge x arm profiles ----------------------------------------------
    print("\n== C · Per-judge arm profiles (judge-centered mean quality) ==")
    print(f"{'judge':<10}" + "".join(f"{a[:9]:>11}" for a in arms))
    for J in judges:
        per = {}
        allq = [q for a, j, f, q in recs if j == J]
        gm = mean(allq)
        for a in arms:
            per[a] = mean(q for a2, j, f, q in recs if a2 == a and j == J) - gm
        print(f"{SHORT.get(J, J):<10}" + "".join(f"{per[a]:>11.2f}" for a in arms))

    # ---- D. mode bias per arm ------------------------------------------------------
    print("\n== D · Mode bias inside each corpus (own major rate vs favors-major) ==")
    maj, tot = {}, {}
    for f in (ROOT / "docs/data").glob("*/features.csv"):
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("prompt") == "free-form" and r.get("key_mode_best") in ("major", "minor"):
                m = r["model"]
                tot[m] = tot.get(m, 0) + 1
                maj[m] = maj.get(m, 0) + (r["key_mode_best"] == "major")
    own_major = {m: maj[m] / tot[m] for m in tot}
    print(f"{'arm':<12}{'r across judges':>16}{'n maj/min':>12}")
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        bias = major_bias(sub, lambda r: r["panel"], lambda r: r["mode"] or "")
        xs = [own_major[J] for J in bias if J in own_major]
        ys = [bias[J] for J in bias if J in own_major]
        nmaj = sum(1 for r in sub if r["mode"] == "major")
        r_ = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 else float("nan")
        print(f"{a:<12}{r_:>16.2f}{nmaj:>7}/{len(sub) - nmaj}")

    # ---- E. sample origin guesses --------------------------------------------------
    print("\n== E · Sample origin guesses ==")
    rng = np.random.default_rng(0)
    for a in ("chinese_han", "arab_and", "irish_folk"):
        gs = sorted({v["origin_guess"] for r in rows if r["arm"] == a
                     for v in r["panel"].values() if "origin_guess" in v})
        pick = rng.choice(len(gs), size=min(8, len(gs)), replace=False)
        print(f"\n{a}:")
        for i in pick:
            print(f"  - {gs[i]}")


if __name__ == "__main__":
    main()
