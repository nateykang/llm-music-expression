#!/usr/bin/env python3
"""Build an instrument-rarity (IDF) weight table from the GigaMIDI corpus.

Streams a sample of GigaMIDI, parses each file's General-MIDI instrument programs,
counts how often each program appears across files, and converts to inverse-document-
frequency weights: weight = -log(fraction of files containing the program). Common
instruments (piano, guitar) → low weight; rare ones (koto, sitar, organ) → high.
Saves docs/analysis/instrument_idf.json (program -> weight), used by analyze.py to
score each generated piece's instrument-rarity.

Usage:  python scripts/build_instrument_idf.py [--n 6000]
"""

from __future__ import annotations

import argparse
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="number of corpus files to sample")
    args = ap.parse_args()

    import pretty_midi
    from datasets import load_dataset

    ds = load_dataset("Metacreation/GigaMIDI", "v1.0.0", split="train", streaming=True)
    prog_files = Counter()
    drum_files = 0
    n = 0
    print(f"streaming GigaMIDI, target {args.n} parseable files…", flush=True)
    for rec in ds:
        try:
            pm = pretty_midi.PrettyMIDI(io.BytesIO(rec["music"]))
        except Exception:
            continue
        progs, has_drum = set(), False
        for inst in pm.instruments:
            if inst.is_drum:
                has_drum = True
            else:
                progs.add(int(inst.program))
        if not progs and not has_drum:
            continue
        n += 1
        for p in progs:
            prog_files[p] += 1
        drum_files += has_drum
        if n % 500 == 0:
            print(f"  {n}/{args.n} parsed", flush=True)
        if n >= args.n:
            break

    # IDF weight per program: -log(smoothed fraction of files containing it)
    weights = {}
    for p in range(128):
        freq = (prog_files.get(p, 0) + 1) / (n + 2)
        weights[str(p)] = round(-math.log(freq), 3)
    weights["drum"] = round(-math.log((drum_files + 1) / (n + 2)), 3)

    out = ROOT / "docs/analysis/instrument_idf.json"
    out.write_text(json.dumps({"n_files": n, "weights": weights}, indent=1), encoding="utf-8")
    print(f"\nWrote {out}  (n={n} files)")
    print("most common programs (low weight):",
          [(p, weights[str(p)]) for p, _ in prog_files.most_common(6)])
    rare = sorted(((weights[str(p)], p) for p in range(128) if prog_files.get(p, 0) > 0), reverse=True)[:6]
    print("rarest used programs (high weight):", [(p, w) for w, p in rare])
    print("=== IDF BUILD DONE ===")


if __name__ == "__main__":
    main()
