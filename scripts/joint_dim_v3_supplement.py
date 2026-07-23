#!/usr/bin/env python3
"""v3 supplement: (i) full-pipeline subsample uncertainty for the CLAP-CLAP
projection overlap — centering, rank selection, PCA, and trace(Pa Pt) all
recomputed per 80% subsample — reported with the ratio over the rotation-null
expectation; (ii) the 2x2 synthetic regime grid (piece-level sharing x
composer/mode nuisance), each tested under unrestricted and composer-x-mode
restricted step-down permutation inference. The critical success criterion:
in the nuisance-only regime, unrestricted permutations reject while
composer-x-mode-restricted inference does not.

    python scripts/joint_dim_v3_supplement.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from description_corpus import load_pieces, piece_id  # noqa: E402
from joint_dimensionality_v2 import parallel_rank  # noqa: E402
import joint_dimensionality_v3 as v3  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
SEED = 20260726


def overlap_full_pipeline(A, T, rng):
    Ac, Tc = A - A.mean(0), T - T.mean(0)
    ra = parallel_rank(Ac, rng=rng)
    rt = parallel_rank(Tc, rng=rng)
    Va = PCA(ra).fit(Ac).components_.T
    Vt = PCA(rt).fit(Tc).components_.T
    s = np.linalg.svd(Va.T @ Vt, compute_uv=False)
    return float((s ** 2).sum())


def main():
    result = {"seed": SEED}
    pieces = load_pieces(ROOT)
    z = np.load(ANALYSIS / "clap_embeddings.npz", allow_pickle=True)
    cpos = {p: i for i, p in enumerate(z["ids"])}
    sub = [p for p in pieces if piece_id(p) in cpos]

    print("=== (i) projection-overlap uncertainty, full pipeline per subsample ===",
          flush=True)
    result["overlap_uncertainty"] = {}
    for mode in ("abc", "codegen", "all"):
        pp = [p for p in sub if mode == "all" or p["mode"] == mode]
        A = np.stack([z["audio"][cpos[piece_id(p)]] for p in pp])
        T = np.stack([z["text_long"][cpos[piece_id(p)]] for p in pp])
        vals = []
        for i in range(40):
            rng = np.random.default_rng(SEED + i)
            idx = rng.choice(len(A), int(0.8 * len(A)), replace=False)
            vals.append(overlap_full_pipeline(A[idx], T[idx], rng))
        lo, hi = np.percentile(vals, [5, 95])
        point = overlap_full_pipeline(A, T, np.random.default_rng(SEED))
        result["overlap_uncertainty"][mode] = {
            "d_overlap": round(point, 1),
            "sub90": [round(float(lo), 1), round(float(hi), 1)]}
        print(f"  [{mode}] d_overlap={point:.1f}  90% subsample [{lo:.1f}, {hi:.1f}]",
              flush=True)

    print("\n=== (ii) synthetic 2x2 regime grid (199-perm step-down each) ===",
          flush=True)
    n_cells = dict(Counter((p["model"], p["mode"]) for p in sub))
    regimes = {"nuisance_only": dict(dZ=0, alpha_comp=1.0, alpha_mode=1.0),
               "piece_only": dict(dZ=3, alpha_comp=0.0, alpha_mode=0.0),
               "both": dict(dZ=3, alpha_comp=1.0, alpha_mode=1.0),
               "neither": dict(dZ=0, alpha_comp=0.0, alpha_mode=0.0)}
    result["synthetic_grid"] = {}
    for rname, kw in regimes.items():
        Ts, Ms, cl, ml = v3.synth_nuisance(n_cells, **kw)
        row = {}
        for sname, st in (("unrestricted", np.zeros(len(Ts), dtype=int)),
                          ("composer_x_mode", np.char.add(cl, ml))):
            k, so, sp = v3.stepdown_count(Ms, Ts, st)
            row[sname] = {"significant_fwer_05": int(k),
                          "top_corrs": [round(float(v), 2) for v in so[:4]]}
            print(f"  {rname:14s} | {sname:16s}: FWER-sig = {k}", flush=True)
        result["synthetic_grid"][rname] = row

    ok = (result["synthetic_grid"]["nuisance_only"]["unrestricted"]
          ["significant_fwer_05"] > 0
          and result["synthetic_grid"]["nuisance_only"]["composer_x_mode"]
          ["significant_fwer_05"] == 0)
    result["critical_success_nuisance_only"] = bool(ok)
    print(f"\n  critical success (nuisance-only: unrestricted rejects, "
          f"restricted does not): {ok}", flush=True)

    out = ANALYSIS / "joint_dimensionality_v3_supplement.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
