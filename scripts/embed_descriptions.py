#!/usr/bin/env python3
"""Embed every corpus description with a dedicated text-embedding model
(default OpenAI text-embedding-3-large) — a proper semantic text space for the
RSA and fingerprint analyses, replacing CLAP's audio-caption text encoder,
which is optimized for predicting audio pairings rather than prose semantics.

Embeds the full self-description (short + long, no title).

    python scripts/embed_descriptions.py

Writes docs/analysis/description_text_embeddings.npz {ids, embeddings, model}.
"""

from __future__ import annotations

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

MODEL = "text-embedding-3-large"
BATCH = 100


def main():
    from openai import OpenAI

    client = OpenAI()
    pieces = load_pieces(ROOT)
    ids = [piece_id(p) for p in pieces]
    texts = [f"{p['short_description']} {p['long_description']}".strip()
             or "(empty)" for p in pieces]
    vecs = []
    for i in range(0, len(texts), BATCH):
        resp = client.embeddings.create(model=MODEL, input=texts[i:i + BATCH])
        vecs.extend(d.embedding for d in resp.data)
        print(f"  [{min(i + BATCH, len(texts))}/{len(texts)}]", flush=True)
    E = np.asarray(vecs, dtype=np.float32)
    out = ROOT / "docs/analysis/description_text_embeddings.npz"
    np.savez_compressed(out, ids=np.array(ids), embeddings=E,
                        model=np.array(MODEL))
    print(f"Wrote {E.shape} -> {out}")


if __name__ == "__main__":
    main()
