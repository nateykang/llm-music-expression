#!/usr/bin/env python3
"""Three follow-ups that put the PCA axes to work.

1. Taste vectors: regress each judge's leniency-corrected deviation on the
   piece's PC scores (own pieces EXCLUDED, so self-preference can't leak in).
   Does the taste vector point toward the judge's own location in style space?
2. Signature location: on which axes do a judge's own pieces differ most from
   its 50 lookalikes? (candidate carriers of the authorship residual)
3. Length compliance: judges are told "do not reward length" — do they anyway?
   Within-author contrasts so authorial style can't confound.

    python scripts/taste_vectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_music.embed_report import (  # noqa: E402
    _load_embeddings, _load_features, _qual,
)
from llm_music.report_common import SHORT  # noqa: E402

np.random.seed(0)
N_PC = 10


def main():
    analysis = ROOT / "docs/analysis"
    ids, En = _load_embeddings(analysis)
    models = np.array([i[0] for i in ids])
    feats = _load_features(ROOT / "docs/data")
    raw = json.loads((analysis / "judge_allmodels_raw.json").read_text(encoding="utf-8"))
    key2row = {i: j for j, i in enumerate(ids)}
    judges = sorted({j for p in raw for j in p["panel"]})

    from sklearn.decomposition import PCA
    X = PCA(n_components=N_PC, random_state=0).fit_transform(En)
    Xz = X / X.std(axis=0)  # unit-variance axes so betas are comparable

    # deviations for every (judge, piece) with embedding row + author
    dev_rows = {J: [] for J in judges}  # (row, author, dev)
    for p in raw:
        k = (p["model"], p.get("mode"), p.get("title"), str(p.get("sample")))
        row = key2row.get(k)
        if row is None:
            continue
        quals = {j: _qual(v) for j, v in p["panel"].items()}
        quals = {j: q for j, q in quals.items() if q is not None}
        for J, qJ in quals.items():
            peers = [q for j, q in quals.items() if j != J]
            if peers:
                dev_rows[J].append((row, p["model"], qJ - mean(peers)))

    # ---- 1. taste vectors --------------------------------------------------------
    pos, beta = {}, {}
    for J in judges:
        own_rows = np.where(models == J)[0]
        pos[J] = Xz[own_rows].mean(0)                      # location in style space
        oth = [(r, d) for r, a, d in dev_rows[J] if a != J]  # own pieces excluded
        A = np.column_stack([Xz[[r for r, _ in oth]], np.ones(len(oth))])
        y = np.array([d for _, d in oth])
        beta[J] = np.linalg.lstsq(A, y, rcond=None)[0][:N_PC]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    align = {J: cos(beta[J], pos[J]) for J in judges}
    obs = mean(align.values())
    # permutation: shuffle which position goes with which taste vector
    rng = np.random.default_rng(0)
    P_ = [pos[J] for J in judges]
    B_ = [beta[J] for J in judges]
    n_perm, hits = 20000, 0
    for _ in range(n_perm):
        perm = rng.permutation(len(judges))
        if mean(cos(B_[i], P_[perm[i]]) for i in range(len(judges))) >= obs:
            hits += 1
    print("== 1 · Taste vectors: does taste point toward own style position? ==")
    print(f"{'judge':<10}{'cos(taste, position)':>22}")
    for J in sorted(judges, key=lambda j: -align[j]):
        print(f"{SHORT.get(J, J):<10}{align[J]:>22.2f}")
    print(f"mean alignment = {obs:+.2f}   permutation p = {hits / n_perm:.4f} "
          f"(shuffling positions across judges)")

    # per-axis view: across judges, corr(own position on PC k, taste beta on PC k)
    print(f"\n{'axis':<6}{'corr(position, taste) across judges':>38}")
    for k in range(N_PC):
        pk = [pos[J][k] for J in judges]
        bk = [beta[J][k] for J in judges]
        r = np.corrcoef(pk, bk)[0, 1]
        print(f"PC{k + 1:<4}{r:>38.2f}")

    # ---- 2. where does the authorship signature live? ---------------------------
    print("\n== 2 · Signature location: own pieces vs 50 lookalikes, per axis ==")
    cent = {J: En[models == J].mean(0) for J in judges}
    for J in cent:
        cent[J] /= np.linalg.norm(cent[J])
    diffs = []
    for J in judges:
        sims = En @ cent[J]
        oth_rows = np.where(models != J)[0]
        kin_rows = oth_rows[np.argsort(-sims[oth_rows])[:50]]
        diffs.append(Xz[models == J].mean(0) - Xz[kin_rows].mean(0))
    D = np.array(diffs)
    order = np.argsort(-np.abs(D).mean(0))
    print(f"{'axis':<6}{'mean |own - kin| (z)':>22}{'sign consistency':>18}")
    for k in order:
        signs = np.sign(D[:, k])
        cons = max((signs > 0).sum(), (signs < 0).sum())
        print(f"PC{k + 1:<4}{np.abs(D[:, k]).mean():>22.2f}{cons:>15}/10")

    # ---- 3. do judges reward length despite the instruction? --------------------
    print("\n== 3 · Length compliance (within-author, so style can't confound) ==")
    lens = np.full(len(ids), np.nan)
    for i, k in enumerate(ids):
        fr = feats.get((k[0], k[1], k[2], k[3] if k[3] != "None" else "0"))
        try:
            lens[i] = float(fr["length_seconds"]) if fr and fr["length_seconds"] else np.nan
        except (KeyError, TypeError, ValueError):
            pass
    print(f"{'judge':<10}{'r(dev, length)':>16}{'pts/min':>10}{'n':>6}")
    panel_x, panel_y = [], []
    for J in judges:
        xs, ys = [], []
        by_author = {}
        for r, a, d in dev_rows[J]:
            if a != J and not np.isnan(lens[r]):
                by_author.setdefault(a, []).append((lens[r], d))
        for a, pts in by_author.items():
            if len(pts) < 5:
                continue
            ml = mean(x for x, _ in pts)
            md = mean(y for _, y in pts)
            xs += [x - ml for x, _ in pts]
            ys += [y - md for _, y in pts]
        r_ = np.corrcoef(xs, ys)[0, 1]
        slope = np.polyfit(xs, ys, 1)[0] * 60
        panel_x += xs
        panel_y += ys
        print(f"{SHORT.get(J, J):<10}{r_:>16.2f}{slope:>10.3f}{len(xs):>6}")
    r_all = np.corrcoef(panel_x, panel_y)[0, 1]
    slope_all = np.polyfit(panel_x, panel_y, 1)[0] * 60
    # permutation within the pooled demeaned data
    rng = np.random.default_rng(0)
    px = np.array(panel_x)
    py = np.array(panel_y)
    hits = sum(abs(np.corrcoef(rng.permutation(px), py)[0, 1]) >= abs(r_all)
               for _ in range(2000))
    print(f"panel pooled: r = {r_all:+.3f}, {slope_all:+.3f} pts per extra minute "
          f"(perm p = {hits / 2000:.3f}, n = {len(px)})")


if __name__ == "__main__":
    main()
