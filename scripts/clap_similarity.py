#!/usr/bin/env python3
"""Experiment 2.3 — description<->audio correspondence in CLAP's joint space.

Embeds every piece's rendered audio and its own description text with a
music-trained CLAP (contrastive language-audio) model, then asks: is a piece's
audio closer to ITS description than to the other pieces' descriptions?

Corpus: the free-form 30-samples-per-composer batches (651 pieces, all with
audio) — see description_corpus.py. Text is used VERBATIM (faithfulness, not a
de-identified matching test). Two text variants: "short" (title + short
description) and "long" (the full reflection).

    python scripts/clap_similarity.py --limit 5     # smoke test
    python scripts/clap_similarity.py               # full run

Resumable: audio embeddings cache per piece under docs/analysis/embedding/clap/.
Writes docs/analysis/clap_embeddings.npz and docs/analysis/clap_similarity.json:
  matched vs mismatched cosine, and within-composer 30-way retrieval
  (top-1 / MRR / median rank; chance top-1 = 1/30) per composer x mode.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from description_corpus import load_pieces, piece_id  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
CACHE = ANALYSIS / "embedding" / "clap"
SR = 48_000
CHUNK = 10 * SR
# NOT laion/larger_clap_music: its HF-hub conversion is broken (every text and
# every audio input embeds to the same direction, cos~1). larger_clap_general
# and clap-htsat-unfused verified healthy on unrelated-caption probes.
MODEL_ID = "laion/larger_clap_general"


def _device():
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _cache_path(pid: str) -> Path:
    return CACHE / (re.sub(r"[^A-Za-z0-9_.-]", "_", pid) + ".npy")


def embed_audio(model, processor, device, path: Path) -> np.ndarray:
    import librosa
    import torch

    y, _ = librosa.load(path, sr=SR, mono=True)
    chunks = [y[i:i + CHUNK] for i in range(0, len(y), CHUNK)]
    chunks = [c for c in chunks if len(c) >= SR // 2] or [y]
    inputs = processor(audio=chunks, sampling_rate=SR, return_tensors="pt",
                       padding=True)
    with torch.no_grad():
        e = model.get_audio_features(
            **{k: v.to(device) for k, v in inputs.items()})
    e = getattr(e, "pooler_output", e)  # transformers >=5 returns an output object
    e = torch.nn.functional.normalize(e, dim=-1).mean(0)
    return torch.nn.functional.normalize(e, dim=0).cpu().numpy()


def embed_texts(model, processor, device, texts: list[str],
                batch_size: int = 64) -> np.ndarray:
    import torch

    out = []
    for i in range(0, len(texts), batch_size):
        inputs = processor(text=texts[i:i + batch_size], return_tensors="pt",
                           padding=True, truncation=True)
        with torch.no_grad():
            e = model.get_text_features(
                **{k: v.to(device) for k, v in inputs.items()})
        e = getattr(e, "pooler_output", e)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out)


def retrieval_stats(sim: np.ndarray) -> dict:
    """sim[i, j] = cos(audio_i, text_j); the diagonal is the true pairing.
    Rank 1 = the piece's own description is its audio's nearest text."""
    n = sim.shape[0]
    ranks = 1 + (sim > np.diag(sim)[:, None]).sum(axis=1)
    return {"n": n,
            "top1": round(float((ranks == 1).mean()), 3),
            "mrr": round(float((1.0 / ranks).mean()), 3),
            "median_rank": float(np.median(ranks)),
            "chance_top1": round(1.0 / n, 3)}


def summarize(pieces: list[dict], A: np.ndarray, T: dict) -> dict:
    result = {"model": MODEL_ID, "text_variants": {}}
    groups = defaultdict(list)
    for i, p in enumerate(pieces):
        groups[(p["model"], p["mode"])].append(i)
    for variant, X in T.items():
        sim = A @ X.T
        diag = np.diag(sim)
        off = sim[~np.eye(len(sim), dtype=bool)]
        per = {}
        for (model, mode), idx in sorted(groups.items()):
            sub = sim[np.ix_(idx, idx)]
            per[f"{model}|{mode}"] = {
                **retrieval_stats(sub),
                "matched_cos": round(float(np.diag(sub).mean()), 4)}
        result["text_variants"][variant] = {
            "matched_cos": round(float(diag.mean()), 4),
            "mismatched_cos": round(float(off.mean()), 4),
            "gap": round(float(diag.mean() - off.mean()), 4),
            "global_retrieval": retrieval_stats(sim),
            "within_composer_retrieval": per,
        }
    return result


def print_summary(result: dict) -> None:
    for variant, r in result["text_variants"].items():
        g = r["global_retrieval"]
        print(f"\n[{variant}] matched={r['matched_cos']:.3f} "
              f"mismatched={r['mismatched_cos']:.3f} gap={r['gap']:+.3f}   "
              f"global {g['n']}-way: top1={g['top1']:.1%} mrr={g['mrr']:.3f} "
              f"med_rank={g['median_rank']:.0f}")
        print(f"  within-composer 30-way (chance top1=3.3%):")
        for key, s in r["within_composer_retrieval"].items():
            print(f"  {key:34s} top1={s['top1']:6.1%}  mrr={s['mrr']:.3f}  "
                  f"med_rank={s['median_rank']:4.0f}  matched_cos={s['matched_cos']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N pieces")
    args = ap.parse_args()

    import torch
    from transformers import ClapModel, ClapProcessor

    torch.manual_seed(0)
    device = _device()
    pieces = [p for p in load_pieces(ROOT) if p["audio"]]
    if args.limit:
        pieces = pieces[: args.limit]
    print(f"{len(pieces)} pieces, device={device}, model={MODEL_ID}", flush=True)

    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)

    CACHE.mkdir(parents=True, exist_ok=True)
    A = np.zeros((len(pieces), model.config.projection_dim), dtype=np.float32)
    todo = [(i, p) for i, p in enumerate(pieces)
            if not _cache_path(piece_id(p)).exists()]
    print(f"audio: {len(pieces) - len(todo)} cached, {len(todo)} to embed", flush=True)
    for k, (i, p) in enumerate(todo):
        np.save(_cache_path(piece_id(p)),
                embed_audio(model, processor, device, p["audio"]))
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            print(f"  [{k + 1}/{len(todo)}]", flush=True)
    for i, p in enumerate(pieces):
        A[i] = np.load(_cache_path(piece_id(p)))

    texts_short = [f"{p['title']}. {p['short_description']}" for p in pieces]
    texts_long = [p["long_description"] or p["short_description"] for p in pieces]
    T = {"short": embed_texts(model, processor, device, texts_short),
         "long": embed_texts(model, processor, device, texts_long)}

    ids = np.array([piece_id(p) for p in pieces])
    np.savez_compressed(ANALYSIS / "clap_embeddings.npz",
                        ids=ids, audio=A, text_short=T["short"],
                        text_long=T["long"])
    result = summarize(pieces, A, T)
    (ANALYSIS / "clap_similarity.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    print_summary(result)
    print(f"\nWrote clap_embeddings.npz + clap_similarity.json -> {ANALYSIS}")


if __name__ == "__main__":
    main()
