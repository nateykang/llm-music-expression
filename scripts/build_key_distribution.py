#!/usr/bin/env python3
"""Build a reference KEY distribution from the GigaMIDI corpus — an approximation of
the key prior in the data LLMs are trained on, to compare against the keys the models
actually generate.

Streams a sample of GigaMIDI, and for each file estimates the key with the
Krumhansl-Schmuckler algorithm (duration-weighted pitch-class histogram correlated
against the 24 rotated major/minor key profiles) — the SAME method music21 uses for
our generated pieces' detected key, so the two distributions are comparable. Saves
docs/analysis/key_distribution.json (per-key probability + major fraction + tonic
distribution).

Usage:  python scripts/build_key_distribution.py [--n 6000]
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Krumhansl-Kessler key profiles.
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def key_of(hist):
    h = np.asarray(hist, float)
    if h.sum() <= 0:
        return None
    h = h - h.mean()
    hn = np.linalg.norm(h)
    if hn == 0:
        return None
    best = None
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        p = np.asarray(prof) - np.mean(prof)
        pn = np.linalg.norm(p)
        for t in range(12):
            r = float(np.dot(h, np.roll(p, t)) / (hn * pn))
            if best is None or r > best[0]:
                best = (r, NAMES[t], mode)
    return (best[1], best[2]) if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args()

    import pretty_midi
    from datasets import load_dataset

    ds = load_dataset("Metacreation/GigaMIDI", "v1.0.0", split="train", streaming=True)
    keys = Counter()
    n = 0
    print(f"streaming GigaMIDI, target {args.n} keyable files…", flush=True)
    for rec in ds:
        try:
            pm = pretty_midi.PrettyMIDI(io.BytesIO(rec["music"]))
        except Exception:
            continue
        hist = [0.0] * 12
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                hist[note.pitch % 12] += max(0.0, note.end - note.start)
        k = key_of(hist)
        if k is None:
            continue
        n += 1
        keys[f"{k[0]} {k[1]}"] += 1
        if n % 500 == 0:
            print(f"  {n}/{args.n} keyed", flush=True)
        if n >= args.n:
            break

    total = sum(keys.values())
    major = sum(c for kk, c in keys.items() if kk.endswith("major")) / total
    tonic = Counter()
    for kk, c in keys.items():
        tonic[kk.split()[0]] += c
    out = {
        "n_files": n,
        "major_frac": round(major, 4),
        "keys": {kk: round(c / total, 5) for kk, c in keys.items()},
        "tonic": {t: round(c / total, 5) for t, c in tonic.items()},
    }
    (ROOT / "docs/analysis/key_distribution.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nWrote key_distribution.json  (n={n})")
    print(f"major fraction: {major:.3f}")
    print("top keys:", keys.most_common(8))
    print("=== KEY DIST DONE ===")


if __name__ == "__main__":
    main()
