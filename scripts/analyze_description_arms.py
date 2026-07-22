#!/usr/bin/env python3
"""Stats for the en-route vs post-hoc description comparison (Q1: valence).

Primary endpoint: within-piece paired delta (en route − post hoc) in
evaluative_positivity on the SHORT text (length-matched by construction),
averaged over available raters, Wilcoxon signed-rank. Long-text deltas are
reported raw and with a log-length-ratio adjustment (regression intercept).
Everything stratified by mode; sparse-toolkit pieces never pool with
free-form. Writes docs/analysis/description_arms_summary.json.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "docs/analysis"
RATERS = ["fable-5", "gpt-5.6-thinking"]
SCALES = ["evaluative_positivity", "weakness_admission", "affect_valence"]


def rater_mean(row: dict, arm: str, text: str, scale: str) -> float | None:
    vals = [row[arm][r][f"{text}.{scale}"] for r in RATERS
            if r in row[arm] and f"{text}.{scale}" in row[arm][r]]
    return float(np.mean(vals)) if vals else None


def wilcoxon(deltas: list[float]) -> dict:
    arr = np.array(deltas)
    nz = arr[arr != 0]
    out = {"n": len(arr), "mean_delta": round(float(arr.mean()), 3),
           "sd": round(float(arr.std(ddof=1)), 3) if len(arr) > 1 else None}
    if len(nz) >= 10:
        w = stats.wilcoxon(arr)
        out["wilcoxon_p"] = float(f"{w.pvalue:.2e}")
    return out


def main():
    rows = json.loads((ANALYSIS / "valence_comparison.json").read_text())
    summary = {"raters": RATERS, "n_pieces": len(rows), "scales": {}}

    for scale in SCALES:
        block = {}
        for text in ("short", "long"):
            deltas_all, by_mode, by_model = [], defaultdict(list), defaultdict(list)
            lenratio = []
            floor = {"enroute": 0, "posthoc": 0, "n": 0}
            for r in rows:
                er = rater_mean(r, "enroute", text, scale)
                ph = rater_mean(r, "posthoc", text, scale)
                if er is None or ph is None:
                    continue
                d = er - ph
                deltas_all.append(d)
                by_mode[r["mode"]].append(d)
                by_model[r["model"]].append(d)
                lw_er = r["enroute"]["measures"][f"{text}.words"]
                lw_ph = r["posthoc"]["measures"][f"{text}.words"]
                lenratio.append(math.log(max(1, lw_er) / max(1, lw_ph)))
                if scale == "weakness_admission":
                    floor["n"] += 1
                    floor["enroute"] += er == 1.0
                    floor["posthoc"] += ph == 1.0
            entry = {"overall": wilcoxon(deltas_all),
                     "by_mode": {m: wilcoxon(v) for m, v in sorted(by_mode.items())},
                     "by_model": {m: wilcoxon(v) for m, v in sorted(by_model.items())}}
            if text == "long":
                # length control: delta ~ log(words_er/words_ph); intercept =
                # expected delta at equal lengths
                X = np.column_stack([np.ones(len(lenratio)), lenratio])
                beta, *_ = np.linalg.lstsq(X, np.array(deltas_all), rcond=None)
                entry["length_adjusted_mean_delta"] = round(float(beta[0]), 3)
                entry["length_slope"] = round(float(beta[1]), 3)
            if scale == "weakness_admission" and floor["n"]:
                entry["floor_rate"] = {
                    "enroute": round(floor["enroute"] / floor["n"], 3),
                    "posthoc": round(floor["posthoc"] / floor["n"], 3)}
            block[text] = entry
        summary["scales"][scale] = block

    # deterministic measures
    det = {}
    for key in ("vader", "first_person_per_100w", "words"):
        for text in ("short", "long"):
            ds = [r["enroute"]["measures"][f"{text}.{key}"]
                  - r["posthoc"]["measures"][f"{text}.{key}"] for r in rows]
            er = [r["enroute"]["measures"][f"{text}.{key}"] for r in rows]
            ph = [r["posthoc"]["measures"][f"{text}.{key}"] for r in rows]
            det[f"{text}.{key}"] = {
                "enroute_mean": round(float(np.mean(er)), 3),
                "posthoc_mean": round(float(np.mean(ph)), 3),
                **wilcoxon(ds)}
    summary["deterministic"] = det

    # rater agreement (pearson across all arm×text cells)
    agree = {}
    for scale in SCALES:
        a, b = [], []
        for r in rows:
            for arm in ("enroute", "posthoc"):
                for text in ("short", "long"):
                    key = f"{text}.{scale}"
                    if all(rt in r[arm] and key in r[arm][rt] for rt in RATERS):
                        a.append(r[arm][RATERS[0]][key])
                        b.append(r[arm][RATERS[1]][key])
        agree[scale] = round(float(np.corrcoef(a, b)[0, 1]), 3)
    summary["rater_agreement_r"] = agree

    # contrast category rates (per piece, per arm)
    con = json.loads((ANALYSIS / "description_contrast.json").read_text())
    n = len(con["pieces"])
    summary["contrast_per_piece"] = {
        side: {cat: round(cnt / n, 2) for cat, cnt in cats.items()}
        for side, cats in con["category_totals"].items()}
    summary["contrast_n"] = n

    out = ANALYSIS / "description_arms_summary.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1)[:4000])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
