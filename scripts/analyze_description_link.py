#!/usr/bin/env python3
"""Free analyses tying descriptions to judged quality and to music geometry —
no new model calls; everything reads existing docs/analysis artifacts.

  A. Which description traits predict the BLIND panel score? (plan idea 3)
     Cheap lexical traits (length, hedging, self-praise, technical vocabulary,
     first-person rate) + the extracted claims (claimed valence/arousal),
     correlated with judge_allmodels `overall` — pooled AND within-model
     (model-demeaned), since composer identity confounds pooled correlations.
  B. Calibration / overconfidence per composer (plan idea 5a): does a model's
     descriptive self-assurance track how good its music actually is — across
     models, and across its own pieces?
  C. Persuasion (plan idea 5b): on the 188 corpus pieces judged both blind and
     with the composer's note, does the note shift scores, and do confident
     notes shift them more?
  D. Representational similarity (plan idea 4): Mantel tests between music
     space (CLAP-audio; MERT) and description space (CLAP-text) — do similar
     pieces get similar descriptions?
  E. Description fingerprint: leave-one-out nearest-centroid composer ID from
     text embeddings alone (chance 1/11) — the text side of the style-space
     signature question.

    python scripts/analyze_description_link.py

Writes docs/analysis/description_link_analysis.json and prints the summary.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from description_corpus import load_pieces, piece_id  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
RNG = np.random.default_rng(20260710)
N_PERM = 1000

HEDGES = ("perhaps|maybe|might|hope(?:fully|d|s)?|attempt(?:s|ed)?|tr(?:y|ies|ied) to|"
          "tentative(?:ly)?|modest(?:ly)?|humble|imperfect(?:ly)?|uncertain(?:ty)?|"
          "somewhat|slightly|aim(?:s|ed)? to|mean(?:t|s)? to|intend(?:s|ed)? to|"
          "roughly|a bit|not (?:quite|sure|certain)")
PRAISE = ("deeply|profound(?:ly)?|powerful(?:ly)?|striking(?:ly)?|vivid(?:ly)?|"
          "masterful(?:ly)?|beautiful(?:ly)?|elegant(?:ly)?|compelling|evocative|"
          "soaring|sublime|luminous|exquisite|breathtaking|perfect(?:ly)?|"
          "triumphant(?:ly)?|rich(?:ly)?|stunning|remarkable|extraordinary")
TECH = ("counterpoint|contrapuntal|cadence|cadential|motif|motiv(?:e|ic)|harmon(?:y|ic|ies)|"
        "chromatic(?:ism)?|modulat(?:e|es|ion|ing)|arpeggi(?:o|os|ated)|ostinato|"
        "syncopat(?:ed|ion)|inversion|pedal (?:point|tone)|voice[- ]leading|suspension|"
        "hemiola|canon(?:ic)?|fug(?:ue|al)|imitation|sequence|augment(?:ed|ation)|"
        "diminish(?:ed|ution)|tritone|dissonan(?:ce|t)|resolution|tonic|dominant|"
        "subdominant|polyrhythm|cross[- ]rhythm|appoggiatura|retrograde")

_PATS = {"hedge": re.compile(rf"\b(?:{HEDGES})\b", re.I),
         "praise": re.compile(rf"\b(?:{PRAISE})\b", re.I),
         "tech": re.compile(rf"\b(?:{TECH})\b", re.I),
         "first_person": re.compile(r"\b(?:I|my|me|myself)\b")}


def text_traits(text: str) -> dict:
    words = max(1, len(text.split()))
    out = {"n_words": float(words)}
    for name, pat in _PATS.items():
        out[f"{name}_per100w"] = 100.0 * len(pat.findall(text)) / words
    out["confidence_net"] = out["praise_per100w"] - out["hedge_per100w"]
    return out


def within_model_spearman(df: pd.DataFrame, x: str, y: str) -> tuple:
    """Spearman of x vs y after removing each model's mean from both."""
    d = df[[x, y, "model"]].dropna()
    if len(d) < 30:
        return None, None, len(d)
    xr = d[x] - d.groupby("model")[x].transform("mean")
    yr = d[y] - d.groupby("model")[y].transform("mean")
    rho, p = spearmanr(xr, yr)
    return round(float(rho), 3), round(float(p), 5), len(d)


def mantel(D1: np.ndarray, D2: np.ndarray, n_perm: int = N_PERM) -> dict:
    """Spearman Mantel test between two square distance matrices."""
    n = D1.shape[0]
    iu = np.triu_indices(n, 1)
    r1, r2 = rankdata(D1[iu]), rankdata(D2[iu])
    obs = float(np.corrcoef(r1, r2)[0, 1])
    hits = 0
    for _ in range(n_perm):
        perm = RNG.permutation(n)
        rp = rankdata(D2[perm][:, perm][iu])
        if np.corrcoef(r1, rp)[0, 1] >= obs:
            hits += 1
    return {"rho": round(obs, 4), "p": round((hits + 1) / (n_perm + 1), 4), "n": n}


def cos_dist(X: np.ndarray) -> np.ndarray:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 - Xn @ Xn.T


def loo_centroid_acc(X: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out nearest-centroid classification accuracy."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    classes = sorted(set(labels))
    sums = {c: Xn[labels == c].sum(0) for c in classes}
    counts = {c: int((labels == c).sum()) for c in classes}
    correct = 0
    for i in range(len(Xn)):
        best, best_s = None, -2.0
        for c in classes:
            cnt = counts[c] - (1 if labels[i] == c else 0)
            if cnt == 0:
                continue
            cen = sums[c] - (Xn[i] if labels[i] == c else 0)
            s = float(Xn[i] @ (cen / np.linalg.norm(cen)))
            if s > best_s:
                best, best_s = c, s
        correct += int(best == labels[i])
    return round(correct / len(Xn), 3)


def main():
    pieces = load_pieces(ROOT)
    rows = []
    claims = {r["id"]: r["claims"] for r in json.loads(
        (ANALYSIS / "description_claims.json").read_text(encoding="utf-8"))}
    for p in pieces:
        text = f"{p['short_description']} {p['long_description']}".strip()
        c = claims.get(piece_id(p), {})
        rows.append({"id": piece_id(p), "batch": p["batch"], "model": p["model"],
                     "mode": p["mode"], "sample": p["sample"], "title": p["title"],
                     "claimed_valence": c.get("valence"),
                     "claimed_arousal": c.get("arousal"), **text_traits(text)})
    df = pd.DataFrame(rows)

    ratings_path = ANALYSIS / "description_trait_ratings.json"
    rated_cols = []
    if ratings_path.exists():
        rr = json.loads(ratings_path.read_text(encoding="utf-8"))
        rdf = pd.DataFrame([{"id": r["id"],
                             **{f"rated_{k}": v for k, v in r["ratings"].items()}}
                            for r in rr])
        rated_cols = [c for c in rdf.columns if c != "id"]
        df = df.merge(rdf, on="id", how="left")

    ja = pd.read_csv(ANALYSIS / "judge_allmodels.csv")
    df = df.merge(ja[["batch", "model", "mode", "sample", "overall", "valence",
                      "arousal", "emotion", "creativity", "naturalness"]],
                  on=["batch", "model", "mode", "sample"], how="left")
    result = {}

    # ---- A. traits -> blind overall -----------------------------------
    traits = ["n_words", "hedge_per100w", "praise_per100w", "tech_per100w",
              "first_person_per100w", "confidence_net",
              "claimed_valence", "claimed_arousal"] + rated_cols
    a = {}
    for t in traits:
        d = df[[t, "overall"]].dropna()
        rho, p = spearmanr(d[t], d["overall"])
        wr, wp, wn = within_model_spearman(df, t, "overall")
        a[t] = {"pooled_rho": round(float(rho), 3), "pooled_p": round(float(p), 5),
                "within_model_rho": wr, "within_model_p": wp, "n": len(d)}
    result["A_traits_vs_blind_overall"] = a

    # ---- B. calibration across and within composers -------------------
    per_model = df.groupby("model").agg(
        confidence_net=("confidence_net", "mean"),
        hedge=("hedge_per100w", "mean"), praise=("praise_per100w", "mean"),
        blind_overall=("overall", "mean")).round(3)
    across = spearmanr(per_model["confidence_net"], per_model["blind_overall"])
    within = {m: round(float(spearmanr(g["confidence_net"], g["overall"])[0]), 3)
              for m, g in df.dropna(subset=["overall"]).groupby("model")
              if len(g) >= 20}
    zc = (per_model["confidence_net"] - per_model["confidence_net"].mean()) \
        / per_model["confidence_net"].std()
    zq = (per_model["blind_overall"] - per_model["blind_overall"].mean()) \
        / per_model["blind_overall"].std()
    per_model["overconfidence_z"] = (zc - zq).round(2)
    if rated_cols:
        rated_pm = df.groupby("model")[["rated_implied_quality",
                                        "rated_confidence"]].mean().round(3)
        per_model = per_model.join(rated_pm)
        zr = (per_model["rated_implied_quality"]
              - per_model["rated_implied_quality"].mean()) \
            / per_model["rated_implied_quality"].std()
        per_model["overconfidence_z_rated"] = (zr - zq).round(2)
    result["B_calibration"] = {
        "per_model": json.loads(per_model.sort_values(
            "overconfidence_z", ascending=False).to_json(orient="index")),
        "across_models_rho": round(float(across.statistic), 3),
        "across_models_p": round(float(across.pvalue), 4),
        "within_model_conf_vs_score": within,
    }

    # ---- C. persuasion: noted - blind on the paired subset ------------
    jb = pd.read_csv(ANALYSIS / "judge.csv")
    jn = pd.read_csv(ANALYSIS / "judge_noted.csv")
    keys = ["batch", "model", "mode", "sample"]
    pair = jb[keys + ["overall"]].merge(
        jn[keys + ["overall", "intent"]], on=keys, suffixes=("_blind", "_noted"))
    pair = pair.merge(df[keys + ["confidence_net", "hedge_per100w",
                                 "praise_per100w", "n_words"] + rated_cols],
                      on=keys)
    pair["delta"] = pair["overall_noted"] - pair["overall_blind"]
    c_traits = ["confidence_net", "praise_per100w", "hedge_per100w",
                "n_words"] + rated_cols
    c_rho = {t: tuple(round(float(v), 4) for v in
                      spearmanr(pair[t], pair["delta"], nan_policy="omit"))
             for t in c_traits}
    result["C_persuasion"] = {
        "n_paired": len(pair),
        "mean_delta_noted_minus_blind": round(float(pair["delta"].mean()), 3),
        "delta_sd": round(float(pair["delta"].std()), 3),
        "frac_inflated": round(float((pair["delta"] > 0).mean()), 3),
        "delta_by_model": {m: round(float(g["delta"].mean()), 3)
                           for m, g in pair.groupby("model")},
        "trait_vs_delta_spearman": {k: {"rho": v[0], "p": v[1]}
                                    for k, v in c_rho.items()},
        "intent_vs_delta": tuple(round(float(v), 4) for v in
                                 spearmanr(pair["intent"], pair["delta"])),
    }

    # ---- D. RSA: music geometry vs description geometry ---------------
    z = np.load(ANALYSIS / "clap_embeddings.npz", allow_pickle=True)
    pos = {pid: i for i, pid in enumerate(z["ids"])}
    mz = np.load(ANALYSIS / "music2emo_embeddings.npz", allow_pickle=True)
    mert_map = {}
    for j, s in enumerate(mz["index"]):
        model, mode, title, sample = s.split("|")
        mert_map[(model, mode, title, sample)] = j
    text_spaces = {"clap": None}  # None -> sliced from z["text_long"]
    tz_path = ANALYSIS / "description_text_embeddings.npz"
    if tz_path.exists():
        tz = np.load(tz_path, allow_pickle=True)
        tpos = {pid: i for i, pid in enumerate(tz["ids"])}
        text_spaces[str(tz["model"])] = (tz["embeddings"], tpos)
    d_rsa = {}
    for mode in ("abc", "codegen", "all"):
        sub = df if mode == "all" else df[df["mode"] == mode]
        keep = [r for r in sub.to_dict("records") if r["id"] in pos]
        ci = [pos[r["id"]] for r in keep]
        A = z["audio"][ci]
        Dm = cos_dist(A)
        iu = np.triu_indices(len(ci), 1)
        thr = np.quantile(Dm[iu], 0.1)
        mert_loc = [(j, mert_map[k]) for j, r in enumerate(keep)
                    if (k := (r["model"], r["mode"], r["title"],
                              str(r["sample"]))) in mert_map]
        d_rsa[mode] = {}
        for sname, space in text_spaces.items():
            if space is None:
                T = z["text_long"][ci]
            else:
                E, tpos_ = space
                if any(r["id"] not in tpos_ for r in keep):
                    continue
                T = E[[tpos_[r["id"]] for r in keep]]
            Dt = cos_dist(T)
            entry = {"audio_vs_text": mantel(Dm, Dt),
                     "text_dist_close_music_pairs":
                         round(float(Dt[iu][Dm[iu] <= thr].mean()), 4),
                     "text_dist_all_pairs": round(float(Dt[iu].mean()), 4)}
            if len(mert_loc) >= 50:
                lj = [j for j, _ in mert_loc]
                mj = [m for _, m in mert_loc]
                entry["mert_vs_text"] = mantel(
                    cos_dist(mz["embeddings"][mj]), cos_dist(T[lj]))
                entry["mert_join_n"] = len(mj)
            d_rsa[mode][sname] = entry
    result["D_rsa"] = d_rsa

    # ---- E. composer fingerprint from descriptions alone --------------
    e = {}
    for mode in ("abc", "codegen"):
        sub = df[df["mode"] == mode]
        keep = [r for r in sub.to_dict("records") if r["id"] in pos]
        ci = [pos[r["id"]] for r in keep]
        labels = np.array([r["model"] for r in keep])
        e[mode] = {"audio_loo_acc": loo_centroid_acc(z["audio"][ci], labels),
                   "chance": round(1.0 / len(set(labels)), 3), "n": len(ci)}
        for sname, space in text_spaces.items():
            if space is None:
                T = z["text_long"][ci]
            else:
                E_, tpos_ = space
                if any(r["id"] not in tpos_ for r in keep):
                    continue
                T = E_[[tpos_[r["id"]] for r in keep]]
            e[mode][f"text_loo_acc_{sname}"] = loo_centroid_acc(T, labels)
    result["E_fingerprint"] = e

    out = ANALYSIS / "description_link_analysis.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\n=== A. traits vs blind overall (pooled / within-model rho) ===")
    for t, v in result["A_traits_vs_blind_overall"].items():
        print(f"  {t:22s} {v['pooled_rho']:+.3f} (p={v['pooled_p']})   "
              f"within {v['within_model_rho']:+.3f} (p={v['within_model_p']})")
    print("\n=== B. calibration (overconfidence_z = z(conf) - z(quality)) ===")
    for m, v in result["B_calibration"]["per_model"].items():
        extra = ""
        if "rated_implied_quality" in v and v["rated_implied_quality"] is not None:
            extra = (f"  implied_q={v['rated_implied_quality']:.2f}"
                     f"  overconf_z_rated={v['overconfidence_z_rated']:+.2f}")
        print(f"  {m:22s} conf={v['confidence_net']:+.2f}  "
              f"overall={v['blind_overall']:.2f}  "
              f"overconf_z={v['overconfidence_z']:+.2f}{extra}")
    print(f"  across-model rho={result['B_calibration']['across_models_rho']} "
          f"(p={result['B_calibration']['across_models_p']})")
    print("\n=== C. persuasion (noted - blind, n=%d) ===" % result["C_persuasion"]["n_paired"])
    C = result["C_persuasion"]
    print(f"  mean delta={C['mean_delta_noted_minus_blind']:+.3f} "
          f"(sd {C['delta_sd']}), {C['frac_inflated']:.0%} inflated")
    for k, v in C["trait_vs_delta_spearman"].items():
        print(f"  {k:18s} vs delta: rho={v['rho']:+.3f} (p={v['p']})")
    print("\n=== D. RSA (Mantel) ===")
    for mode, spaces in result["D_rsa"].items():
        for sname, v in spaces.items():
            line = (f"  [{mode}][{sname}] audio~text rho={v['audio_vs_text']['rho']}"
                    f" (p={v['audio_vs_text']['p']})")
            if "mert_vs_text" in v:
                line += (f"   mert~text rho={v['mert_vs_text']['rho']}"
                         f" (p={v['mert_vs_text']['p']}, n={v['mert_join_n']})")
            print(line)
            print(f"        text-dist: closest-music-decile "
                  f"{v['text_dist_close_music_pairs']} vs all {v['text_dist_all_pairs']}")
    print("\n=== E. composer fingerprint (LOO nearest-centroid) ===")
    for mode, v in result["E_fingerprint"].items():
        txt = "  ".join(f"{k.removeprefix('text_loo_acc_')}={x:.1%}"
                        for k, x in v.items() if k.startswith("text_loo_acc_"))
        print(f"  [{mode}] text: {txt}  audio={v['audio_loo_acc']:.1%}"
              f"  (chance {v['chance']:.1%}, n={v['n']})")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
