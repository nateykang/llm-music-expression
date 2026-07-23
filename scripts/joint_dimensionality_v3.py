#!/usr/bin/env python3
"""Joint dimensionality v3 — the finishing pass prescribed by review of v2.

  A. CLAP-CLAP geometry redone with (i) projection overlap trace(Pa Pt) as the
     primary graded statistic and (ii) a spectrum-preserving rotation null:
     each null draw applies independent random orthogonal rotations to the two
     centered blocks, exactly preserving both covariance spectra while
     randomizing relative subspace orientation.
  B. Held-out CCA (independent encoders: CLAP-audio vs OAI-text) with 999
     composer-x-mode restricted permutations and step-down max-statistic
     (Westfall-Young) familywise-adjusted p-values per component.
  C. Canonical-subspace stability: principal angles between the leading
     canonical subspaces (ambient space) learned in different CV folds.
  D. Leave-one-composer-out generalization: CCA trained on 10 composers,
     held-out correlations on the unseen composer, null by permuting the
     held-out pairing under the frozen model.
  E. Synthetic validation of the inference design itself: paired blocks with
     piece-level shared latents PLUS composer and mode nuisance factors at the
     empirical stratum sizes; verify that unrestricted / mode-stratified /
     composer-x-mode-stratified permutation nulls behave as claimed.

    python scripts/joint_dimensionality_v3.py

Writes docs/analysis/joint_dimensionality_v3.json.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from description_corpus import load_pieces, piece_id  # noqa: E402
from joint_dimensionality_v2 import parallel_rank  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
SEED = 20260725
N_PERM = 999
N_COMP = 10
PCA_K = 40


# ---------------------------------------------------------------- A. CLAP-CLAP
def rotation_null_geometry(A, T, n_null=200):
    """Projection overlap + aligned counts vs a spectrum-preserving null."""
    rng = np.random.default_rng(SEED)
    Ac, Tc = A - A.mean(0), T - T.mean(0)
    ra, rt = parallel_rank(Ac, rng=rng), parallel_rank(Tc, rng=rng)
    p = Ac.shape[1]

    def cosines(X, Y):
        Va = PCA(ra).fit(X).components_.T
        Vt = PCA(rt).fit(Y).components_.T
        return np.linalg.svd(Va.T @ Vt, compute_uv=False)[:min(ra, rt)]

    s_obs = cosines(Ac, Tc)
    d_obs = float((s_obs ** 2).sum())
    null_d = np.empty(n_null)
    null_s = np.empty((n_null, len(s_obs)))
    for i in range(n_null):
        Qa = np.linalg.qr(rng.standard_normal((p, p)))[0]
        Qt = np.linalg.qr(rng.standard_normal((p, p)))[0]
        s = cosines(Ac @ Qa, Tc @ Qt)
        null_s[i] = s
        null_d[i] = (s ** 2).sum()
    thr = np.quantile(null_s, 0.95, axis=0)
    keep = s_obs > thr
    aligned = int(np.argmin(keep)) if not keep.all() else len(s_obs)
    return {"rank_audio": int(ra), "rank_text": int(rt),
            "d_overlap": round(d_obs, 1),
            "d_overlap_null_mean": round(float(null_d.mean()), 2),
            "d_overlap_null_95": round(float(np.quantile(null_d, 0.95)), 2),
            "d_overlap_p": round(float(((null_d >= d_obs).sum() + 1)
                                       / (n_null + 1)), 4),
            "aligned_vs_rotation_null": aligned,
            "strong_cos_gt_0.7": int((s_obs > 0.7).sum()),
            "principal_cosines": [round(float(v), 2) for v in s_obs[:12]]}


# ------------------------------------------------- B. step-down permutation CCA
def fold_fits(X, Y, seed=SEED):
    """Per-fold PCA+CCA fits; returns held-out corrs and ambient loadings."""
    corrs = np.zeros(N_COMP)
    loads_x, loads_y = [], []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        px = PCA(PCA_K).fit(X[tr] - X[tr].mean(0))
        py = PCA(PCA_K).fit(Y[tr] - Y[tr].mean(0))
        cca = CCA(n_components=N_COMP, max_iter=500).fit(
            px.transform(X[tr]), py.transform(Y[tr]))
        U, V = cca.transform(px.transform(X[te]), py.transform(Y[te]))
        corrs += np.array([np.corrcoef(U[:, i], V[:, i])[0, 1]
                           for i in range(N_COMP)])
        loads_x.append(px.components_.T @ cca.x_rotations_)
        loads_y.append(py.components_.T @ cca.y_rotations_)
    return corrs / 5, loads_x, loads_y


def stepdown_cca(X, Y, strata):
    obs, loads_x, loads_y = fold_fits(X, Y)
    strata = np.asarray(strata)
    null = np.empty((N_PERM, N_COMP))
    for i in range(N_PERM):
        rng = np.random.default_rng(SEED + 10_000 + i)
        perm = np.arange(len(Y))
        for s in np.unique(strata):
            m = np.where(strata == s)[0]
            perm[m] = rng.permutation(m)
        null[i] = fold_fits(X, Y[perm])[0]
        if (i + 1) % 100 == 0:
            print(f"    perm [{i + 1}/{N_PERM}]", flush=True)
    # Westfall-Young step-down max-statistic on the observed ordering
    order = np.argsort(-obs)
    p_adj = np.empty(N_COMP)
    for j, k in enumerate(order):
        succ_max = null[:, order[j:]].max(axis=1)
        p_adj[k] = ((succ_max >= obs[k]).sum() + 1) / (N_PERM + 1)
    for j in range(1, N_COMP):  # enforce monotonicity down the ordering
        p_adj[order[j]] = max(p_adj[order[j]], p_adj[order[j - 1]])
    return obs, p_adj, loads_x, loads_y


def subspace_stability(loads, k=5):
    """Mean principal cosines between k-dim canonical subspaces across folds."""
    Qs = [np.linalg.qr(L[:, :k])[0] for L in loads]
    cos = []
    for i in range(len(Qs)):
        for j in range(i + 1, len(Qs)):
            cos.append(np.linalg.svd(Qs[i].T @ Qs[j], compute_uv=False))
    return np.array(cos).mean(axis=0)


# ------------------------------------------------------------------- D. LOCO
def loco(X, Y, composers, n_perm=500):
    comps = sorted(set(composers))
    composers = np.asarray(composers)
    mean_corrs = np.zeros(N_COMP)
    mean_null95 = np.zeros(N_COMP)
    for ci, c in enumerate(comps):
        te = np.where(composers == c)[0]
        tr = np.where(composers != c)[0]
        px = PCA(PCA_K).fit(X[tr] - X[tr].mean(0))
        py = PCA(PCA_K).fit(Y[tr] - Y[tr].mean(0))
        cca = CCA(n_components=N_COMP, max_iter=500).fit(
            px.transform(X[tr]), py.transform(Y[tr]))
        U, V = cca.transform(px.transform(X[te]), py.transform(Y[te]))
        corr = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1]
                         for i in range(N_COMP)])
        rng = np.random.default_rng(SEED + 500 + ci)
        null = np.empty((n_perm, N_COMP))
        for i in range(n_perm):
            pr = rng.permutation(len(te))
            null[i] = [np.corrcoef(U[:, j], V[pr, j])[0, 1]
                       for j in range(N_COMP)]
        mean_corrs += corr
        mean_null95 += np.quantile(null, 0.95, axis=0)
    return mean_corrs / len(comps), mean_null95 / len(comps)


# ------------------------------------------------------------- E. synthetic
def synth_nuisance(n_cells, dZ=3, dST=8, dSM=3, alpha_comp=1.0, alpha_mode=1.0,
                   noise=0.3, pT=200, pM=100):
    """Paired blocks with piece-level shared Z plus composer & mode nuisance."""
    rng = np.random.default_rng(SEED + 77)
    rows_T, rows_M, comp_lab, mode_lab = [], [], [], []
    WT = rng.standard_normal((dZ + dST, pT))
    WM = rng.standard_normal((dZ + dSM, pM))
    comps = list(n_cells)
    comp_vecs_T = {c: rng.standard_normal(pT) for c, _ in n_cells}
    comp_vecs_M = {c: rng.standard_normal(pM) for c, _ in n_cells}
    mode_vec_T, mode_vec_M = rng.standard_normal(pT), rng.standard_normal(pM)
    for (c, mode), n in n_cells.items():
        Z = rng.standard_normal((n, dZ))
        T = np.hstack([Z, rng.standard_normal((n, dST))]) @ WT
        M = np.hstack([Z, rng.standard_normal((n, dSM))]) @ WM
        g = 1.0 if mode == "codegen" else -1.0
        T += alpha_comp * comp_vecs_T[c] + alpha_mode * g * mode_vec_T
        M += alpha_comp * comp_vecs_M[c] + alpha_mode * g * mode_vec_M
        T += noise * rng.standard_normal(T.shape)
        M += noise * rng.standard_normal(M.shape)
        rows_T.append(T); rows_M.append(M)
        comp_lab += [c] * n; mode_lab += [mode] * n
    return (np.vstack(rows_T), np.vstack(rows_M),
            np.array(comp_lab), np.array(mode_lab))


def stepdown_count(X, Y, strata, n_perm=199):
    global N_PERM
    saved, N_PERM = N_PERM, n_perm
    try:
        obs, p_adj, _, _ = stepdown_cca(X, Y, strata)
    finally:
        N_PERM = saved
    return int((p_adj < 0.05).sum()), obs, p_adj


def main():
    result = {"seed": SEED, "n_perm": N_PERM}
    pieces = load_pieces(ROOT)
    z = np.load(ANALYSIS / "clap_embeddings.npz", allow_pickle=True)
    tz = np.load(ANALYSIS / "description_text_embeddings.npz", allow_pickle=True)
    cpos = {p: i for i, p in enumerate(z["ids"])}
    tpos = {p: i for i, p in enumerate(tz["ids"])}
    sub = [p for p in pieces if piece_id(p) in cpos and piece_id(p) in tpos]

    print("=== A. CLAP-CLAP: projection overlap vs rotation null ===", flush=True)
    result["clap_clap_v3"] = {}
    for mode in ("abc", "codegen", "all"):
        pp = [p for p in sub if mode == "all" or p["mode"] == mode]
        A = np.stack([z["audio"][cpos[piece_id(p)]] for p in pp])
        T = np.stack([z["text_long"][cpos[piece_id(p)]] for p in pp])
        g = rotation_null_geometry(A, T)
        result["clap_clap_v3"][mode] = g
        print(f"  [{mode}] d_overlap={g['d_overlap']} "
              f"(rot-null mean {g['d_overlap_null_mean']}, "
              f"95th {g['d_overlap_null_95']}, p={g['d_overlap_p']})  "
              f"aligned={g['aligned_vs_rotation_null']}  "
              f"strong={g['strong_cos_gt_0.7']}", flush=True)

    A = np.stack([z["audio"][cpos[piece_id(p)]] for p in sub])
    T = np.stack([tz["embeddings"][tpos[piece_id(p)]] for p in sub])
    strata = [f"{p['mode']}|{p['model']}" for p in sub]
    composers = [p["model"] for p in sub]

    print(f"\n=== B. step-down CCA, {N_PERM} composer-x-mode permutations ===",
          flush=True)
    obs, p_adj, loads_x, loads_y = stepdown_cca(A, T, strata)
    result["stepdown_cca"] = {
        "held_out_corrs": [round(float(v), 3) for v in obs],
        "p_adjusted": [round(float(v), 4) for v in p_adj],
        "significant_fwer_05": int((p_adj < 0.05).sum())}
    print("  corrs:", " ".join(f"{v:.2f}" for v in obs), flush=True)
    print("  p_adj:", " ".join(f"{v:.3f}" for v in p_adj), flush=True)
    print(f"  -> FWER<.05 significant components: {(p_adj < .05).sum()}",
          flush=True)

    print("\n=== C. canonical-subspace stability (top-5, fold-to-fold) ===",
          flush=True)
    sx = subspace_stability(loads_x)
    sy = subspace_stability(loads_y)
    result["subspace_stability"] = {
        "audio_side_cosines": [round(float(v), 2) for v in sx],
        "text_side_cosines": [round(float(v), 2) for v in sy]}
    print(f"  audio side: {' '.join(f'{v:.2f}' for v in sx)}", flush=True)
    print(f"  text  side: {' '.join(f'{v:.2f}' for v in sy)}", flush=True)

    print("\n=== D. leave-one-composer-out generalization ===", flush=True)
    corr, null95 = loco(A, T, composers)
    result["loco"] = {"mean_held_out_corrs": [round(float(v), 3) for v in corr],
                      "mean_null_95": [round(float(v), 3) for v in null95],
                      "components_above_null": int((corr > null95).sum())}
    print("  corrs :", " ".join(f"{v:.2f}" for v in corr), flush=True)
    print("  null95:", " ".join(f"{v:.2f}" for v in null95), flush=True)

    print("\n=== E. synthetic: does the stratified null isolate piece-level Z? ===",
          flush=True)
    from collections import Counter
    n_cells = Counter((p["model"], p["mode"]) for p in sub)
    Ts, Ms, cl, ml = synth_nuisance(dict(n_cells))
    schemes = {"unrestricted": np.zeros(len(Ts), dtype=int),
               "mode": ml, "composer_x_mode": np.char.add(cl, ml)}
    result["synthetic_nuisance"] = {"true_piece_level_dZ": 3}
    for name, st in schemes.items():
        k, so, sp = stepdown_count(Ms, Ts, st)
        result["synthetic_nuisance"][name] = {
            "significant_fwer_05": k,
            "corrs": [round(float(v), 2) for v in so[:6]]}
        print(f"  {name:16s}: FWER-significant = {k} (true piece-level dZ=3)",
              flush=True)

    out = ANALYSIS / "joint_dimensionality_v3.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
