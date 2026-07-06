#!/usr/bin/env python3
"""Embedding-space analysis of the MERT audio embeddings (+ mood/symbolic spaces).

Reproduces, deterministically (seed 0), the working-note numbers:
  1. PCA effective dimensionality of the three feature spaces.
  2. 5-NN model/method identification (5-fold CV) + silhouette — local
     fingerprints vs global clusters.
  3. Cross-representation fingerprint transfer (train ABC -> test codegen).
  4. Within-model diversity + near-duplicate pairs (sampling mode collapse).
  5. t-SNE / PCA maps colored by method, model, and Music2Emo valence, plus
     identification of detached islands (DBSCAN on the t-SNE plane) and their
     instrumentation profile.

    python scripts/embedding_analysis.py [--out docs/analysis/embedding_figs]

Requires the dev extras (scikit-learn, matplotlib): pip install -e ".[dev]".
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore")
np.random.seed(0)

NUMERIC = ["scale_consistency", "pitch_class_entropy", "pitch_entropy", "pitch_in_scale_rate",
           "consonance_rate", "chord_tone_rate", "chord_tonal_distance", "structureness",
           "polyphony", "n_voices", "note_density", "tempo_bpm", "pitch_range", "n_pitches_used",
           "rhythm_entropy", "pitch_interval", "ioi", "dynamic_span", "dynamic_changes",
           "length_seconds"]

BG, INK, MUTED = "#faf9f7", "#1d1b18", "#6b665d"
PAL = ["#7a5a3a", "#b5651d", "#3a6b5a", "#8a3a4a", "#4a5a7a", "#9a7a3a",
       "#5a7a4a", "#7a4a6a", "#3a7a7a", "#aa5a3a", "#37648a", "#a08030"]
MODE_C = {"abc": "#37648a", "smt-abc": "#7a9ac0", "codegen": "#BA7517"}


def parse_idx(s: str):
    model, mode, rest = s.split("|", 2)
    title, sample = rest.rsplit("|", 1)
    return model, mode, title, sample


def load_spaces():
    z = np.load(ROOT / "docs/analysis/music2emo_embeddings.npz", allow_pickle=True)
    ids = [parse_idx(s) for s in z["index"].tolist()]
    E = z["embeddings"].astype(np.float64)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)

    m2e = json.loads((ROOT / "docs/analysis/music2emo_full.json").read_text(encoding="utf-8"))
    bykey = {(e["model"], e.get("mode"), e.get("title"), str(e.get("sample"))): e for e in m2e}
    moods = sorted(m2e[0]["mood_probs"].keys())
    M = np.array([[bykey[i]["mood_probs"][m] for m in moods] for i in ids])
    val = np.array([bykey[i]["valence"] for i in ids])

    feats = {}
    for f in (ROOT / "docs/data").glob("*/features.csv"):
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("prompt") == "free-form":
                feats[(r["model"], r.get("mode"), r.get("title"), str(r.get("sample") or 0))] = r
    S_rows, keep = [], []
    for j, i in enumerate(ids):
        fr = feats.get((i[0], i[1], i[2], i[3] if i[3] != "None" else "0"))
        if fr is None:
            continue
        try:
            S_rows.append([float(fr[k]) for k in NUMERIC])
            keep.append(j)
        except (ValueError, TypeError, KeyError):
            pass
    return ids, En, M, np.array(S_rows), keep, val, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/analysis/embedding_figs"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    ids, En, M, S, keep, val, feats = load_spaces()
    models = np.array([i[0] for i in ids])
    modes = np.array([i[1] for i in ids])
    Ss = StandardScaler().fit_transform(S)
    print(f"spaces: MERT {En.shape}, moods {M.shape}, symbolic {Ss.shape}")

    # 1. dimensionality
    print("\n== PCA effective dimensionality ==")
    for name, X in [("MERT", En), ("moods", M), ("symbolic", Ss)]:
        p = PCA(random_state=0).fit(X)
        cum = np.cumsum(p.explained_variance_ratio_)
        print(f"  {name:9s} PCs for 50/80/95%: {int(np.searchsorted(cum, .5)) + 1}/"
              f"{int(np.searchsorted(cum, .8)) + 1}/{int(np.searchsorted(cum, .95)) + 1}"
              f"   PC1={p.explained_variance_ratio_[0] * 100:.0f}%")

    # 2. fingerprints
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    print("\n== 5-NN label identification (5-fold CV) + silhouette ==")
    for name, X, kj in [("MERT", En, None), ("moods", M, None), ("symbolic", Ss, keep)]:
        ml = models[kj] if kj is not None else models
        mo = modes[kj] if kj is not None else modes
        am = cross_val_score(KNeighborsClassifier(5, metric="cosine"), X, ml, cv=cv).mean()
        ao = cross_val_score(KNeighborsClassifier(5, metric="cosine"), X, mo, cv=cv).mean()
        print(f"  {name:9s} model {am * 100:3.0f}% (chance 8%)   method {ao * 100:3.0f}%"
              f"   silhouette(model) {silhouette_score(X, ml, metric='cosine'):+.3f}")

    # 3. transfer
    abc_m = np.isin(modes, ["abc", "smt-abc"])
    cg_m = modes == "codegen"
    common = sorted(set(models[abc_m]) & set(models[cg_m]))
    ka = abc_m & np.isin(models, common)
    kc = cg_m & np.isin(models, common)
    f1 = KNeighborsClassifier(5, metric="cosine").fit(En[ka], models[ka])
    f2 = KNeighborsClassifier(5, metric="cosine").fit(En[kc], models[kc])
    print(f"\n== fingerprint transfer ({len(common)} models, chance {100 / len(common):.0f}%) ==")
    print(f"  ABC->codegen {100 * (f1.predict(En[kc]) == models[kc]).mean():.0f}%"
          f"   codegen->ABC {100 * (f2.predict(En[ka]) == models[ka]).mean():.0f}%")

    # 4. diversity + near-duplicates
    D = 1 - En @ En.T
    print("\n== within-model diversity (mean pairwise cosine distance) ==")
    div = {}
    for m in sorted(set(models)):
        mask = models == m
        tri = D[np.ix_(mask, mask)][np.triu_indices(mask.sum(), 1)]
        div[m] = tri.mean()
        print(f"  {m:22s} n={mask.sum():3d}  diversity={tri.mean():.3f}  closest={tri.min():.4f}")
    corpus_mean = D[np.triu_indices(len(En), 1)].mean()
    print(f"  corpus-wide mean pair distance: {corpus_mean:.3f}")
    Dd = D.copy()
    np.fill_diagonal(Dd, 9)
    dup_pairs = {(min(a, b), max(a, b)) for a, b in np.argwhere(Dd < 0.005)}
    print(f"  near-identical audio pairs (<0.005): {len(dup_pairs)}")
    for a, b in sorted(dup_pairs):
        print(f"    {ids[a][0]} s{ids[a][3]} '{ids[a][2][:26]}' <-> s{ids[b][3]} '{ids[b][2][:26]}'  d={Dd[a, b]:.5f}")

    # 5. maps + island
    Xt = TSNE(n_components=2, perplexity=30, random_state=0, init="pca",
              metric="cosine").fit_transform(En)
    lab = DBSCAN(eps=4.0, min_samples=8).fit_predict(Xt)
    sizes = Counter(lab[lab >= 0])
    if len(sizes) > 1:
        main_c = sizes.most_common(1)[0][0]
        for c, n in sizes.items():
            if c == main_c:
                continue
            m = lab == c
            print(f"\n== detached island: n={n} ==")
            print(f"  models: {Counter(models[m]).most_common(5)}")
            print(f"  modes:  {Counter(modes[m]).most_common()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def scatter(X, colorer, title, fname, cvals=None):
        f, ax = plt.subplots(figsize=(7.4, 6), dpi=130)
        f.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]), ax.set_yticks([])
        if cvals is not None:
            sc = ax.scatter(X[:, 0], X[:, 1], c=cvals, cmap="coolwarm_r", s=16, alpha=.85,
                            edgecolors="none")
            f.colorbar(sc, ax=ax, shrink=.7, label="Music2Emo valence (1–9)")
        else:
            for lab_, col in colorer:
                m = np.array([lab_ == x for x in (modes if lab_ in MODE_C else models)])
                ax.scatter(X[m, 0], X[m, 1], color=col, s=16, alpha=.85, edgecolors="none", label=lab_)
            ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, .5))
        ax.set_title(title, color=INK, fontsize=11)
        f.tight_layout()
        f.savefig(out / fname, facecolor=BG)
        plt.close(f)

    mlist = sorted(set(models))
    scatter(Xt, [(m, MODE_C[m]) for m in ["abc", "smt-abc", "codegen"]],
            "t-SNE of MERT audio embeddings — by generation method", "tsne_mode.png")
    scatter(Xt, [(m, PAL[i % len(PAL)]) for i, m in enumerate(mlist)],
            "t-SNE — by model", "tsne_model.png")
    scatter(Xt, None, "t-SNE — by Music2Emo valence", "tsne_valence.png", cvals=val)

    f, ax = plt.subplots(figsize=(7.4, 4.4), dpi=130)
    f.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    order = sorted(mlist, key=lambda m: div[m])
    ax.barh(range(len(order)), [div[m] for m in order], color="#7a5c3e")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.axvline(corpus_mean, color=MUTED, ls=":", lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("mean pairwise cosine distance between a model's own pieces", fontsize=8, color=MUTED)
    ax.set_title("Within-model diversity of the audio embeddings", color=INK, fontsize=11)
    f.tight_layout()
    f.savefig(out / "diversity.png", facecolor=BG)
    plt.close(f)
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
