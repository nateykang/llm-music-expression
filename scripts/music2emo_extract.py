#!/usr/bin/env python3
"""Extract the audio-emotion + acoustic feature suite for every free-form piece, using
Music2Emo (MERT) + librosa. Writes docs/analysis/music2emo_full.json and the MERT
embeddings npz (music2emo_embeddings.npz). Run on a CUDA GPU pod after
setup_music2emo_gpu.sh, from inside ~/Music2Emotion (so `import music2emo` resolves):

    python /path/to/repo/scripts/music2emo_extract.py                 # all pieces, overwrite
    python /path/to/repo/scripts/music2emo_extract.py --model fable-5 --append   # add one model

Per piece: Music2Emo gives valence/arousal (1-9), mood tags + a 56-mood probability
vector, an audio-derived key + chord sequence, and a 1536-dim MERT embedding (the last
four exposed by scripts/music2emo_patch.py). librosa adds a spectral / dynamics / MFCC /
chroma suite. Everything is measured on FluidSynth-rendered MIDI — out-of-distribution
for MERT, so a cross-check rather than ground truth.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

# Music2Emo checkpoints predate torch's weights_only default flip; allow full unpickle.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

import numpy as np  # noqa: E402
import librosa  # noqa: E402
from music2emo import Music2emo  # noqa: E402

# MERT weights are frozen, but seed everything anyway so any stochastic corner
# (dropout left on, kernel nondeterminism fallbacks) can't drift between runs.
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

ROOT = Path(__file__).resolve().parents[1]


def lib_feats(path: str) -> dict:
    y, sr = librosa.load(path, sr=22050, mono=True)
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))
    f = {}
    try:
        f["lib_tempo"] = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    except Exception as e:
        print(f"  beat_track failed on {path}: {e}", flush=True)
        f["lib_tempo"] = None
    f["spec_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    f["spec_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    f["spec_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    f["spec_flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    f["spec_contrast"] = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    rms = librosa.feature.rms(y=y)
    f["rms_mean"] = float(np.mean(rms))
    f["rms_std"] = float(np.std(rms))
    f["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        f["mfcc%d" % (i + 1)] = float(np.mean(mfcc[i]))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        f["chroma%d" % (i + 1)] = float(np.mean(chroma[i]))
    yh, _ = librosa.effects.hpss(y)
    f["harmonic_ratio"] = float(np.sum(yh ** 2) / (np.sum(y ** 2) + 1e-9))
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    f["onset_rate"] = float(len(onsets) / (len(y) / sr + 1e-9))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "docs/data"))
    ap.add_argument("--out", default=str(ROOT / "docs/analysis/music2emo_full.json"))
    ap.add_argument("--emb-out", default=str(ROOT / "docs/analysis/music2emo_embeddings.npz"))
    ap.add_argument("--model", default=None, help="only process this generating model")
    ap.add_argument("--prompt", default="free-form",
                    help="prompt to process, or 'all' for every prompt (the multi-prompt "
                         "sweeps enable prompt-vs-PC analyses in embedding space)")
    ap.add_argument("--append", action="store_true",
                    help="merge into existing json/npz instead of overwriting")
    args = ap.parse_args()

    pieces = []
    for b in sorted(glob.glob(str(Path(args.data_dir) / "2026*"))):
        bd = Path(b)
        dj = bd / "data.json"
        if not dj.exists():
            continue
        for p in json.load(open(dj))["pieces"]:
            if not p.get("ok"):
                continue
            if args.prompt != "all" and p.get("prompt") != args.prompt:
                continue
            if args.model and p.get("model") != args.model:
                continue
            ar = p.get("audio")
            if ar and (bd / ar).exists():
                pieces.append((p, bd, bd / ar))
    print("%d pieces to process" % len(pieces), flush=True)

    m = Music2emo()
    results, embs, idx = [], [], []
    seen_keys = set()
    for i, (p, bd, ap_) in enumerate(pieces):
        # sample + batch make the identity unique: models reuse titles across
        # samples, and the same cell can recur in different batches.
        rec = {"model": p["model"], "prompt": p.get("prompt"), "mode": p.get("mode"),
               "title": p.get("title"), "sample": p.get("sample"), "batch": bd.name}
        ikey = "%s|%s|%s|%s" % (rec["model"], rec["mode"], rec["title"], rec["sample"])
        if ikey in seen_keys:
            raise SystemExit(f"npz index collision for {ikey!r} (same model/mode/title/sample "
                             "under two prompts or batches) — the embedding index key needs a "
                             "revision before this corpus can be extracted")
        seen_keys.add(ikey)
        try:
            out = m.predict(str(ap_))
            rec["valence"] = float(out["valence"])
            rec["arousal"] = float(out["arousal"])
            rec["moods"] = out["predicted_moods"]
            rec["mood_probs"] = out["mood_probs"]
            rec["audio_key"] = out["key"]
            ch = out["chords"]
            nn = [c for c in ch if c not in ("N", "X")]
            rec["chord_n"] = len(ch)
            rec["chord_distinct"] = len(set(nn))
            rec["chord_changes"] = sum(1 for j in range(1, len(ch)) if ch[j] != ch[j - 1])
            embs.append(out["mert_embedding"])
            idx.append("%s|%s|%s|%s" % (rec["model"], rec["mode"], rec["title"], rec["sample"]))
        except Exception as e:
            print(f"  music2emo failed on {ap_}: {str(e)[:100]}", flush=True)
            rec["error"] = str(e)[:150]
        try:
            rec.update(lib_feats(str(ap_)))
        except Exception as e:
            rec["librosa_error"] = str(e)[:100]
        results.append(rec)
        if (i + 1) % 20 == 0:
            print("%d/%d" % (i + 1, len(pieces)), flush=True)

    # merge or overwrite
    if args.append and Path(args.out).exists():
        existing = json.load(open(args.out))
        key = lambda r: (r["model"], r.get("mode"), r.get("title"), str(r.get("sample")))
        have = {key(r) for r in existing}
        results = existing + [r for r in results if key(r) not in have]
    json.dump(results, open(args.out, "w"), indent=1)

    if embs:
        new_emb = np.array(embs, dtype=np.float32)
        new_idx = np.array(idx)
        if args.append and Path(args.emb_out).exists():
            z = np.load(args.emb_out, allow_pickle=True)
            keep = [j for j, x in enumerate(new_idx.tolist()) if x not in set(z["index"].tolist())]
            new_emb = np.concatenate([z["embeddings"], new_emb[keep]], axis=0)
            new_idx = np.concatenate([z["index"], new_idx[keep]])
        np.savez_compressed(args.emb_out, embeddings=new_emb, index=new_idx)

    ok = sum(1 for r in results if "valence" in r)
    print("=== M2E DONE: %d entries, %d with valence ===" % (len(results), ok), flush=True)


if __name__ == "__main__":
    main()
