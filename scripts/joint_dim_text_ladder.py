#!/usr/bin/env python3
"""Geometric informativeness ladder: rerun the certified joint-structure
analyses (TwoNN IDs, step-down permutation CCA vs CLAP-audio, LOCO,
subspace stability) with the text block at three granularities — title only,
short description only, full text — mirroring the behavioral matching ladder.

Embeds titles and shorts with text-embedding-3-large (cached npz per
granularity), then runs the v3 inference per granularity.

    python scripts/joint_dim_text_ladder.py

Writes docs/analysis/joint_dim_text_ladder.json.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from description_corpus import load_pieces, piece_id  # noqa: E402
from joint_dimensionality_v2 import twonn, norm_block  # noqa: E402
import joint_dimensionality_v3 as v3  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
MODEL = "text-embedding-3-large"


def embed_variant(pieces, texts, cache_name):
    path = ANALYSIS / cache_name
    if path.exists():
        return np.load(path, allow_pickle=True)["embeddings"]
    from openai import OpenAI

    client = OpenAI()
    vecs = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=MODEL, input=texts[i:i + 100])
        vecs.extend(d.embedding for d in resp.data)
    E = np.asarray(vecs, dtype=np.float32)
    np.savez_compressed(path, ids=np.array([piece_id(p) for p in pieces]),
                        embeddings=E, model=np.array(MODEL))
    print(f"  embedded {len(E)} -> {path.name}", flush=True)
    return E


def main():
    pieces = load_pieces(ROOT)
    z = np.load(ANALYSIS / "clap_embeddings.npz", allow_pickle=True)
    tz = np.load(ANALYSIS / "description_text_embeddings.npz", allow_pickle=True)
    cpos = {p: i for i, p in enumerate(z["ids"])}
    tpos = {p: i for i, p in enumerate(tz["ids"])}
    sub = [p for p in pieces if piece_id(p) in cpos and piece_id(p) in tpos]
    A = np.stack([z["audio"][cpos[piece_id(p)]] for p in sub])
    strata = [f"{p['mode']}|{p['model']}" for p in sub]
    composers = [p["model"] for p in sub]

    variants = {
        "title": embed_variant(sub, [(p["title"] or "(untitled)").strip()
                                     for p in sub],
                               "description_text_embeddings_title.npz"),
        "short": embed_variant(sub, [p["short_description"].strip() or "(none)"
                                     for p in sub],
                               "description_text_embeddings_short.npz"),
        "full": np.stack([tz["embeddings"][tpos[piece_id(p)]] for p in sub]),
    }

    result = {"model": MODEL, "n": len(sub), "variants": {}}
    for name, T in variants.items():
        print(f"\n=== [{name}] ===", flush=True)
        Tn, An = norm_block(T), norm_block(A)
        ids = {"id_text": round(twonn(Tn), 1),
               "id_joint": round(twonn(np.hstack([An, Tn])), 1)}
        print(f"  TwoNN text={ids['id_text']} joint={ids['id_joint']}",
              flush=True)
        obs, p_adj, loads_x, _ = v3.stepdown_cca(A, T, strata)
        sx = v3.subspace_stability(loads_x)
        corr, null95 = v3.loco(A, T, composers)
        entry = {
            **ids,
            "held_out_corrs": [round(float(v), 3) for v in obs],
            "p_adjusted": [round(float(v), 4) for v in p_adj],
            "significant_fwer_05": int((p_adj < 0.05).sum()),
            "stability_cosines_audio_side": [round(float(v), 2) for v in sx],
            "loco_corrs": [round(float(v), 3) for v in corr],
            "loco_null95": [round(float(v), 3) for v in null95],
            "loco_components_above_null": int((corr > null95).sum()),
        }
        result["variants"][name] = entry
        print(f"  corrs: {' '.join(f'{v:.2f}' for v in obs[:6])}  "
              f"FWER-sig={entry['significant_fwer_05']}  "
              f"LOCO>null={entry['loco_components_above_null']}  "
              f"stability: {' '.join(f'{v:.2f}' for v in sx)}", flush=True)

    out = ANALYSIS / "joint_dim_text_ladder.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
