#!/usr/bin/env python3
"""Joint dimensionality, rigorous pass. Addresses the v1 review:

  A. Synthetic recovery — simulate T=f(Z,S_T), M=g(Z,S_M) with KNOWN shared
     dimension d_Z, run the exact v1 pipeline (block-normalized TwoNN
     inclusion-exclusion; principal-angle joint rank), and report what each
     estimator recovers under linear/nonlinear maps and noise. Calibrates how
     much to trust the real-data numbers.
  B. CLAP-CLAP shared-space geometry — audio and text encoders trained into
     the SAME R^512, so subspace intersection is mathematically meaningful:
     modality gap, parallel-analysis ranks, principal angles vs a
     random-subspace null, union-rank inclusion-exclusion, per mode, with
     subsample stability. (Caveat carried in the output: CLAP was trained to
     align these, so this measures shared structure in CLAP's semantics.)
  C. Held-out CCA (independent encoders) — PCA+CCA fit inside training folds
     only, canonical correlations evaluated on held-out pairs; significance by
     permuting the pairing within mode strata and re-running the entire
     pipeline. Replaces v1's in-sample CCA.
  D. Subsample stability for the v1 TwoNN numbers (80% subsamples) so IDs are
     reported with uncertainty, not decimal precision.

    python scripts/joint_dimensionality_v2.py

Writes docs/analysis/joint_dimensionality_v2.json and prints the summary.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from description_corpus import load_pieces, piece_id  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
SEED = 20260724
RNG = np.random.default_rng(SEED)


# ------------------------------------------------------------------ shared
def twonn(X):
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12)
    mu = mu[mu > 1]
    return len(mu) / np.log(mu).sum()


def norm_block(X):
    X = X - X.mean(0)
    return X / np.sqrt(max((X ** 2).sum() / len(X), 1e-12))


def parallel_rank(X, n_null=25, q=0.95, max_rank=60, rng=None):
    rng = rng or RNG
    X = X - X.mean(0)
    k = min(max_rank, min(X.shape) - 1)
    ev = PCA(k).fit(X).explained_variance_
    null = np.empty((n_null, k))
    for i in range(n_null):
        Xp = np.column_stack([rng.permutation(X[:, j])
                              for j in range(X.shape[1])])
        null[i] = PCA(k).fit(Xp).explained_variance_
    keep = ev > np.quantile(null, q, axis=0)
    return int(np.argmin(keep)) if not keep.all() else k


def score_joint_rank(X, Y, rx, ry, n_null=100, rng=None):
    """v1 estimator: principal angles between n-dim SCORE subspaces."""
    rng = rng or RNG
    n = len(X)
    Ux = np.linalg.qr(PCA(rx).fit_transform(X - X.mean(0)))[0]
    Uy = np.linalg.qr(PCA(ry).fit_transform(Y - Y.mean(0)))[0]
    s = np.linalg.svd(Ux.T @ Uy, compute_uv=False)
    kmax = min(rx, ry)
    null = np.empty((n_null, kmax))
    for i in range(n_null):
        A = np.linalg.qr(rng.standard_normal((n, rx)))[0]
        B = np.linalg.qr(rng.standard_normal((n, ry)))[0]
        null[i] = np.linalg.svd(A.T @ B, compute_uv=False)[:kmax]
    keep = s[:kmax] > np.quantile(null, 0.95, axis=0)
    return int(np.argmin(keep)) if not keep.all() else kmax


# ------------------------------------------------------------- A. synthetic
def synth_case(n, dZ, dST, dSM, pT=200, pM=100, nonlinear=False, noise=0.3,
               rng=None):
    rng = rng or RNG
    Z = rng.standard_normal((n, dZ)) if dZ else np.zeros((n, 0))
    ST = rng.standard_normal((n, dST))
    SM = rng.standard_normal((n, dSM))
    latT, latM = np.hstack([Z, ST]), np.hstack([Z, SM])

    def emit(lat, p):
        if nonlinear:
            H = np.tanh(lat @ rng.standard_normal((lat.shape[1], 3 * max(1, lat.shape[1]))))
            X = H @ rng.standard_normal((H.shape[1], p))
        else:
            X = lat @ rng.standard_normal((lat.shape[1], p))
        return X + noise * rng.standard_normal((n, p))

    return emit(latT, pT), emit(latM, pM)


def run_synthetic():
    rows = []
    for dZ, dST, dSM in ((4, 15, 2), (0, 12, 4), (6, 6, 6)):
        for nonlinear in (False, True):
            for noise in (0.1, 0.5):
                T, M = synth_case(651, dZ, dST, dSM, nonlinear=nonlinear,
                                  noise=noise)
                Tn, Mn = norm_block(T), norm_block(M)
                dt, dm = twonn(Tn), twonn(Mn)
                dj = twonn(np.hstack([Tn, Mn]))
                rt, rm = parallel_rank(T), parallel_rank(M)
                rj = score_joint_rank(T, M, rt, rm)
                rows.append({"true": [dZ, dST, dSM],
                             "nonlinear": nonlinear, "noise": noise,
                             "twonn_shared": round(dt + dm - dj, 1),
                             "twonn": [round(dt, 1), round(dm, 1), round(dj, 1)],
                             "rank_joint": rj, "ranks": [rt, rm]})
                r = rows[-1]
                print(f"  true dZ={dZ:2d} (dST={dST},dSM={dSM}) "
                      f"{'nl' if nonlinear else 'ln'} noise={noise}: "
                      f"twonn_shared={r['twonn_shared']:5.1f}  "
                      f"rank_joint={rj}", flush=True)
    return rows


# ------------------------------------------------------- B. CLAP-CLAP space
def clap_clap(A, T, label, n_sub=30):
    out = {"label": label, "n": len(A)}
    # modality gap
    ga, gt = A.mean(0), T.mean(0)
    gap = ga - gt
    cos_gap = float(np.linalg.norm(gap) /
                    np.sqrt(np.linalg.norm(ga) * np.linalg.norm(gt) + 1e-12))
    Ac, Tc = A - ga, T - gt  # per-modality centering removes the gap direction
    out["modality_gap_norm"] = round(float(np.linalg.norm(gap)), 3)

    def geometry(Ac, Tc, rng):
        ra, rt = parallel_rank(Ac, rng=rng), parallel_rank(Tc, rng=rng)
        # loading subspaces live in the SAME R^512 -> legitimate intersection
        Va = PCA(ra).fit(Ac).components_.T          # 512 x ra
        Vt = PCA(rt).fit(Tc).components_.T
        s = np.linalg.svd(Va.T @ Vt, compute_uv=False)
        kmax = min(ra, rt)
        null = np.empty((100, kmax))
        p = Ac.shape[1]
        for i in range(100):
            Ra = np.linalg.qr(rng.standard_normal((p, ra)))[0]
            Rt = np.linalg.qr(rng.standard_normal((p, rt)))[0]
            null[i] = np.linalg.svd(Ra.T @ Rt, compute_uv=False)[:kmax]
        thr = np.quantile(null, 0.95, axis=0)
        aligned = int(np.argmin(s[:kmax] > thr)) if not (s[:kmax] > thr).all() \
            else kmax
        strong = int((s[:kmax] > 0.7).sum())
        r_union = parallel_rank(np.vstack([Ac, Tc]), rng=rng)
        return ra, rt, aligned, strong, ra + rt - r_union, s[:kmax]

    ra, rt, aligned, strong, incl_excl, s = geometry(Ac, Tc, RNG)
    out.update({"rank_audio": ra, "rank_text": rt,
                "aligned_directions": aligned,
                "strong_directions_cos>0.7": strong,
                "shared_incl_excl": incl_excl,
                "principal_cosines": [round(float(v), 2) for v in s[:12]]})
    # subsample stability (80%)
    stats = {"aligned": [], "incl_excl": []}
    for i in range(n_sub):
        rng = np.random.default_rng(SEED + i + 1)
        idx = rng.choice(len(A), int(0.8 * len(A)), replace=False)
        _, _, al, _, ie, _ = geometry(Ac[idx], Tc[idx], rng)
        stats["aligned"].append(al)
        stats["incl_excl"].append(ie)
    for k, v in stats.items():
        lo, hi = np.percentile(v, [5, 95])
        out[f"{k}_sub90"] = [int(lo), int(hi)]
    print(f"  [{label}] gap|Δμ|={out['modality_gap_norm']}  "
          f"ranks A/T={ra}/{rt}  aligned={aligned} "
          f"(90% sub {out['aligned_sub90']})  strong(cos>.7)={strong}  "
          f"incl-excl shared={incl_excl} (90% sub {out['incl_excl_sub90']})",
          flush=True)
    return out


# ---------------------------------------------------------- C. held-out CCA
def cv_cca(X, Y, strata, n_comp=10, pca_k=40, n_perm=60):
    """CCA fit on train folds only; canonical corrs on held-out pairs.
    Null: permute pairing within strata, re-run the whole pipeline."""
    def held_out_corrs(Yp):
        corrs = np.zeros(n_comp)
        kf = KFold(5, shuffle=True, random_state=SEED)
        for tr, te in kf.split(X):
            px = PCA(pca_k).fit(X[tr] - X[tr].mean(0))
            py = PCA(pca_k).fit(Yp[tr] - Yp[tr].mean(0))
            Xtr, Xte = px.transform(X[tr]), px.transform(X[te])
            Ytr, Yte = py.transform(Yp[tr]), py.transform(Yp[te])
            cca = CCA(n_components=n_comp, max_iter=500).fit(Xtr, Ytr)
            U, V = cca.transform(Xte, Yte)
            corrs += np.array([np.corrcoef(U[:, i], V[:, i])[0, 1]
                               for i in range(n_comp)])
        return corrs / 5

    obs = held_out_corrs(Y)
    null = np.empty((n_perm, n_comp))
    strata = np.asarray(strata)
    for i in range(n_perm):
        rng = np.random.default_rng(SEED + 1000 + i)
        perm = np.arange(len(Y))
        for s in np.unique(strata):
            m = np.where(strata == s)[0]
            perm[m] = rng.permutation(m)
        null[i] = held_out_corrs(Y[perm])
    thr = np.quantile(null, 0.95, axis=0)
    return obs, thr, int((obs > thr).sum())


# ----------------------------------------------------- D. TwoNN subsampling
def id_stability(M, T, n_sub=40):
    vals = []
    for i in range(n_sub):
        rng = np.random.default_rng(SEED + 2000 + i)
        idx = rng.choice(len(M), int(0.8 * len(M)), replace=False)
        Mn, Tn = norm_block(M[idx]), norm_block(T[idx])
        dm, dt = twonn(Mn), twonn(Tn)
        dj = twonn(np.hstack([Mn, Tn]))
        vals.append((dm, dt, dm + dt - dj))
    v = np.array(vals)
    return {k: [round(float(a), 1), round(float(b), 1)]
            for k, (a, b) in zip(("id_music", "id_text", "shared"),
                                 zip(np.percentile(v, 5, 0),
                                     np.percentile(v, 95, 0)))}


def main():
    result = {"seed": SEED}

    print("=== A. synthetic recovery (n=651, pipeline identical to v1) ===")
    result["synthetic"] = run_synthetic()

    pieces = load_pieces(ROOT)
    z = np.load(ANALYSIS / "clap_embeddings.npz", allow_pickle=True)
    tz = np.load(ANALYSIS / "description_text_embeddings.npz", allow_pickle=True)
    cpos = {p: i for i, p in enumerate(z["ids"])}
    tpos = {p: i for i, p in enumerate(tz["ids"])}
    sub = [p for p in pieces if piece_id(p) in cpos and piece_id(p) in tpos]

    print("\n=== B. CLAP-CLAP shared-space geometry (same R^512) ===")
    result["clap_clap"] = []
    for mode in ("abc", "codegen", "all"):
        pp = [p for p in sub if mode == "all" or p["mode"] == mode]
        A = np.stack([z["audio"][cpos[piece_id(p)]] for p in pp])
        T = np.stack([z["text_long"][cpos[piece_id(p)]] for p in pp])
        result["clap_clap"].append(clap_clap(A, T, mode))

    print("\n=== C. held-out CCA, independent encoders (CLAP-audio vs OAI-text) ===")
    A = np.stack([z["audio"][cpos[piece_id(p)]] for p in sub])
    T = np.stack([tz["embeddings"][tpos[piece_id(p)]] for p in sub])
    strata = [p["mode"] for p in sub]
    obs, thr, k_sig = cv_cca(A, T, strata)
    result["cv_cca"] = {"held_out_corrs": [round(float(v), 3) for v in obs],
                        "null_95": [round(float(v), 3) for v in thr],
                        "significant": k_sig}
    print(f"  held-out canonical corrs: "
          + " ".join(f"{v:.2f}" for v in obs))
    print(f"  null 95th:                "
          + " ".join(f"{v:.2f}" for v in thr))
    print(f"  -> significant held-out shared directions: {k_sig}")

    print("\n=== D. TwoNN subsample stability (CLAP-audio vs OAI-text, 90% CIs) ===")
    result["id_stability"] = id_stability(A, T)
    print(" ", result["id_stability"])

    out = ANALYSIS / "joint_dimensionality_v2.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
