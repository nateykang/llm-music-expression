#!/usr/bin/env python3
"""How many dimensions do the music and its description share?

Three complementary estimators, each run for three music representations
(CLAP audio, MERT audio, symbolic features) against the text-embedding block
(text-embedding-3-large), per generation mode and pooled:

  1. Intrinsic-dimension arithmetic (TwoNN, Facco et al. 2017):
     shared = d_music + d_text - d_joint. Nonlinear; if one modality is a
     function of the other, the joint manifold collapses onto the larger.
  2. AJIVE-style joint rank (after Feng et al. 2018): pick each block's PCA
     rank by parallel analysis, then count principal angles between the two
     score subspaces that beat a random-subspace null -> joint rank r_J and
     individual ranks r_music - r_J, r_text - r_J.
  3. Cross-predictability (5-fold ridge): mean R^2 predicting each block's
     retained PCs from the other block - the "does music determine text"
     fraction, per direction.

    python scripts/joint_dimensionality.py

Writes docs/analysis/joint_dimensionality.json and prints the summary.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from description_corpus import load_pieces, piece_id  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
SEED = 20260723
N_NULL = 200
RNG = np.random.default_rng(SEED)


def twonn(X):
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12)
    mu = mu[mu > 1]
    return len(mu) / np.log(mu).sum()


def norm_block(X):
    X = X - X.mean(0)
    return X / np.sqrt(max((X ** 2).sum() / len(X), 1e-12))


def parallel_rank(X, n_null=30, q=0.95, max_rank=60):
    """PCA rank by parallel analysis: eigenvalues beating column-shuffled null."""
    X = X - X.mean(0)
    k = min(max_rank, min(X.shape) - 1)
    ev = PCA(k).fit(X).explained_variance_
    null = np.empty((n_null, k))
    for i in range(n_null):
        Xp = np.column_stack([RNG.permutation(X[:, j])
                              for j in range(X.shape[1])])
        null[i] = PCA(k).fit(Xp).explained_variance_
    thr = np.quantile(null, q, axis=0)
    keep = ev > thr
    return int(np.argmin(keep)) if not keep.all() else k


def joint_rank(X, Y, rx, ry):
    """Principal angles between PCA score subspaces vs random-subspace null."""
    n = len(X)
    Ux = np.linalg.qr(PCA(rx).fit_transform(X - X.mean(0)))[0]
    Uy = np.linalg.qr(PCA(ry).fit_transform(Y - Y.mean(0)))[0]
    s = np.linalg.svd(Ux.T @ Uy, compute_uv=False)
    kmax = min(rx, ry)
    null = np.empty((N_NULL, kmax))
    for i in range(N_NULL):
        A = np.linalg.qr(RNG.standard_normal((n, rx)))[0]
        B = np.linalg.qr(RNG.standard_normal((n, ry)))[0]
        null[i] = np.linalg.svd(A.T @ B, compute_uv=False)[:kmax]
    thr = np.quantile(null, 0.95, axis=0)
    keep = s[:kmax] > thr
    rj = int(np.argmin(keep)) if not keep.all() else kmax
    return rj, s[:kmax].round(3).tolist(), thr.round(3).tolist()


def cross_r2(X, Y, rank_y, n_folds=5):
    """Mean 5-fold CV R^2 predicting Y's top-`rank_y` PCs from X."""
    Yp = PCA(rank_y).fit_transform(Y - Y.mean(0))
    Yp = Yp / Yp.std(0)
    Xs = PCA(min(60, min(X.shape) - 1)).fit_transform(X - X.mean(0))
    r2 = []
    for tr, te in KFold(n_folds, shuffle=True, random_state=SEED).split(Xs):
        m = Ridge(alpha=10.0).fit(Xs[tr], Yp[tr])
        pred = m.predict(Xs[te])
        ss_res = ((Yp[te] - pred) ** 2).sum(0)
        ss_tot = ((Yp[te] - Yp[te].mean(0)) ** 2).sum(0)
        r2.append(np.mean(1 - ss_res / ss_tot))
    return float(np.mean(r2))


def main():
    pieces = load_pieces(ROOT)
    z = np.load(ANALYSIS / "clap_embeddings.npz", allow_pickle=True)
    tz = np.load(ANALYSIS / "description_text_embeddings.npz", allow_pickle=True)
    mz = np.load(ANALYSIS / "music2emo_embeddings.npz", allow_pickle=True)
    cpos = {p: i for i, p in enumerate(z["ids"])}
    tpos = {p: i for i, p in enumerate(tz["ids"])}
    mmap = {tuple(s.split("|")): j for j, s in enumerate(mz["index"])}

    # symbolic feature block from each batch's features.csv
    sym, sym_ids = {}, []
    num_cols = None
    for batch in sorted({p["batch"] for p in pieces}):
        fpath = ROOT / "docs/data" / batch / "features.csv"
        with fpath.open() as fh:
            for row in csv.DictReader(fh):
                pid = f"{batch}|{row['model']}|{row['mode']}|{row['sample']}"
                if num_cols is None:
                    num_cols = [c for c in row if c not in
                                ("model", "prompt", "mode", "sample", "batch",
                                 "title", "key_tonic", "key_mode",
                                 "key_declared_tonic", "key_declared_mode",
                                 "key_mode_best", "mode_match",
                                 "affect_quadrant", "features_version")]
                vals = []
                for c in num_cols:
                    try:
                        vals.append(float(row[c]))
                    except (TypeError, ValueError):
                        vals.append(np.nan)
                sym[pid] = vals
    result = {"seed": SEED, "n_null": N_NULL, "blocks": {}}

    def music_block(kind, sub):
        if kind == "clap":
            return np.stack([z["audio"][cpos[piece_id(p)]] for p in sub])
        if kind == "mert":
            return np.stack([mz["embeddings"][
                mmap[(p["model"], p["mode"], p["title"], str(p["sample"]))]]
                for p in sub])
        M = np.array([sym[piece_id(p)] for p in sub])
        col_mean = np.nanmean(M, axis=0)
        idx = np.where(np.isnan(M))
        M[idx] = np.take(col_mean, idx[1])
        return (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)

    def eligible(kind, p):
        if piece_id(p) not in tpos:
            return False
        if kind == "clap":
            return piece_id(p) in cpos
        if kind == "mert":
            return (p["model"], p["mode"], p["title"], str(p["sample"])) in mmap
        return piece_id(p) in sym

    for kind in ("clap", "mert", "symbolic"):
        for mode in ("abc", "codegen", "all"):
            sub = [p for p in pieces if (mode == "all" or p["mode"] == mode)
                   and eligible(kind, p)]
            M = music_block(kind, sub)
            T = np.stack([tz["embeddings"][tpos[piece_id(p)]] for p in sub])
            Mn, Tn = norm_block(M), norm_block(T)
            dm, dt = twonn(Mn), twonn(Tn)
            dj = twonn(np.hstack([Mn, Tn]))
            rm, rt = parallel_rank(M), parallel_rank(T)
            rj, angles, thr = joint_rank(M, T, rm, rt)
            entry = {
                "n": len(sub),
                "twonn": {"music": round(dm, 1), "text": round(dt, 1),
                          "joint": round(dj, 1),
                          "shared": round(dm + dt - dj, 1)},
                "ranks": {"music": rm, "text": rt, "joint": rj,
                          "music_only": rm - rj, "text_only": rt - rj},
                "principal_cosines": angles[:12],
                "null_95": thr[:12],
                "r2_music_to_text": round(cross_r2(M, T, rt), 3),
                "r2_text_to_music": round(cross_r2(T, M, rm), 3),
            }
            result["blocks"][f"{kind}|{mode}"] = entry
            e = entry
            print(f"[{kind:8s}|{mode:7s}] n={e['n']:3d}  "
                  f"ID m/t/joint={e['twonn']['music']}/{e['twonn']['text']}/"
                  f"{e['twonn']['joint']} shared={e['twonn']['shared']}  "
                  f"ranks m/t={rm}/{rt} joint={rj} "
                  f"(m-only={rm-rj}, t-only={rt-rj})  "
                  f"R2 m->t={e['r2_music_to_text']} t->m={e['r2_text_to_music']}",
                  flush=True)

    out = ANALYSIS / "joint_dimensionality.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
