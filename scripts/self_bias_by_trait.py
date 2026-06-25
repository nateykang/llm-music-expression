#!/usr/bin/env python3
"""Per-trait self-bias for every model — which dimensions does each model judge
its OWN music differently on, beyond its general tendency.

Cell = leniency-corrected self-bias on a dimension:
    mean over OWN pieces of [self(dim) - peers(dim)]
  - mean over OTHER pieces of [self(dim) - peers(dim)]
>0 = the model is kinder to itself on this trait than it is to everyone else;
<0 = harder on itself. Reads from judge_allmodels_raw.json (exclude_self=False).

Usage:  python scripts/self_bias_by_trait.py
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

DIMS = QUALITY_KEYS + ["valence", "arousal"]
SHORT = {"gpt-5.5": "gpt5.5", "gemini-2.5-pro": "gemini", "opus-4.8": "opus",
         "sonnet-4.6": "sonnet", "deepseek-v4-pro": "deepsk", "gpt-4.1": "gpt4.1",
         "grok-4.3": "grok", "qwen3-max": "qwen", "llama-4-maverick": "llama"}
# columns ordered by judge competence (most reliable critics first)
ORDER = ["opus-4.8", "sonnet-4.6", "gpt-5.5", "qwen3-max", "deepseek-v4-pro",
         "grok-4.3", "gpt-4.1", "llama-4-maverick", "gemini-2.5-pro"]


def main():
    raw = [p for p in json.loads((ROOT / "docs/analysis/judge_allmodels_raw.json")
                                 .read_text(encoding="utf-8")) if p["prompt"] == "free-form"]

    # per (model, dim): own-piece gaps and other-piece gaps vs peers
    own = defaultdict(lambda: defaultdict(list))
    oth = defaultdict(lambda: defaultdict(list))
    for p in raw:
        author = p["model"]
        panel = p["panel"]
        for j in panel:
            for d in DIMS:
                sj = (panel[j].get(d) or {}).get("score")
                peers = [(panel[k].get(d) or {}).get("score") for k in panel if k != j]
                peers = [x for x in peers if x is not None]
                if sj is None or not peers:
                    continue
                gap = sj - mean(peers)
                (own if author == j else oth)[j][d].append(gap)

    models = [m for m in ORDER if own.get(m)]
    corr = {m: {} for m in models}
    for m in models:
        for d in DIMS:
            if own[m].get(d) and oth[m].get(d):
                corr[m][d] = mean(own[m][d]) - mean(oth[m][d])

    print("=== PER-TRAIT self-bias (leniency-corrected) — + favors self, - harder on self ===")
    print(f"{'dimension':12}" + "".join(f"{SHORT[m]:>8}" for m in models))
    for d in DIMS:
        row = f"{d:12}"
        for m in models:
            v = corr[m].get(d)
            row += f"{v:>+8.2f}" if v is not None else f"{'·':>8}"
        print(row)
    print(f"\n{'n own pieces':12}" + "".join(f"{len(own[m].get('harmony', [])):>8}" for m in models))

    # standouts
    cells = [(corr[m][d], m, d) for m in models for d in DIMS if d in corr[m]]
    cells.sort()
    print("\nmost SELF-CRITICAL (model, trait):")
    for v, m, d in cells[:5]:
        print(f"  {SHORT[m]:8} {d:12} {v:+.2f}")
    print("most SELF-FAVORING (model, trait):")
    for v, m, d in cells[-5:][::-1]:
        print(f"  {SHORT[m]:8} {d:12} {v:+.2f}")


if __name__ == "__main__":
    main()
