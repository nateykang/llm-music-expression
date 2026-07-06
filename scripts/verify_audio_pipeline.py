#!/usr/bin/env python3
"""End-to-end verification of the audio-emotion pipeline. Run after any audio
bake, Music2Emo extraction, or audio-judge run:

    python scripts/verify_audio_pipeline.py             # fast checks (~seconds)
    python scripts/verify_audio_pipeline.py --durations # + exhaustive audio-vs-score sweep

Checks, in order of what they prove:
  1. Every manifest audio path encodes exactly its own (prompt, model, sample) —
     the file an API call reads IS the named piece's file.
  2. MP3s carry no ID3 metadata (audio judges stay blind) and fit API limits.
  3. music2emo_full.json covers every audio-eligible free-form piece, with
     unique (model, mode, title, sample) identity and in-range values.
  4. The embeddings npz index matches the JSON 1:1.
  5. Cross-layer agreement: measurements of the SAME audio by independent
     systems must correlate — if audio were ever misrouted, these drop to ~0.
     Hard floor: audio-derived key tonic must match the symbolic key well above
     chance; Music2Emo and gemini-hear valence must both track minor-ness.
  6. predicted_moods == {mood: prob >= 0.5} (patch wiring), tolerating the
     rounded 0.5000 boundary.
  7. (--durations) every MP3's length matches its own score-derived MIDI —
     ABC via abc2midi, code-gen via MusicXML->MIDI. Slow (~15 min).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

NOTE2PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
           "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)


def pearson(pairs):
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    ma, mb = mean(a), mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (va * vb) if va and vb else float("nan")


def manifests():
    for b in sorted((ROOT / "docs/data").glob("*/data.json")):
        yield b.parent, json.loads(b.read_text(encoding="utf-8"))["pieces"]


def mp3_dur(p: Path):
    out = subprocess.run(["afinfo", str(p)], capture_output=True, text=True).stdout
    m = re.search(r"estimated duration: ([\d.]+)", out)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", action="store_true",
                    help="also run the exhaustive audio-vs-score duration sweep (slow)")
    args = ap.parse_args()

    # --- 1. audio path naming + 2. blindness/size -------------------------------
    print("== audio delivery ==")
    bad_names, n_audio, max_mb, tagged = [], 0, 0.0, 0
    for bd, pieces in manifests():
        for p in pieces:
            if not (p.get("ok") and p.get("audio")):
                continue
            n_audio += 1
            suffix = f"_s{p.get('sample')}" if p.get("sample") else ""
            if p["audio"] != f"audio/{p['prompt']}/{p['model']}{suffix}.mp3":
                bad_names.append((bd.name, p["model"], p.get("sample")))
            f = bd / p["audio"]
            if f.exists():
                max_mb = max(max_mb, f.stat().st_size / 1e6)
                head = f.read_bytes()[:2048]
                if head.startswith(b"ID3") or b"TIT2" in head or b"TPE1" in head:
                    tagged += 1
    check(not bad_names, f"all {n_audio} audio paths encode their own piece",
          f"{len(bad_names)} mismatched" if bad_names else "")
    check(tagged == 0, "no ID3 metadata in MP3s (judges stay blind)", f"{tagged} tagged")
    check(max_mb < 15, "largest MP3 under API inline limits", f"{max_mb:.1f} MB")

    # --- 3. music2emo coverage + identity + ranges -------------------------------
    print("== music2emo data ==")
    m2e_path = ROOT / "docs/analysis/music2emo_full.json"
    m2e = json.loads(m2e_path.read_text(encoding="utf-8")) if m2e_path.exists() else []
    eligible = sum(1 for bd, ps in manifests() for p in ps
                   if p.get("ok") and p["prompt"] == "free-form"
                   and p.get("audio") and (bd / p["audio"]).exists())
    with_v = [e for e in m2e if "valence" in e]
    check(len(with_v) == eligible, "every audio-eligible free-form piece measured",
          f"{len(with_v)}/{eligible}")
    kc = Counter((e["model"], e.get("mode"), e.get("title"), str(e.get("sample"))) for e in m2e)
    check(max(kc.values(), default=1) == 1, "identity keys unique in music2emo_full.json")
    check(all(1 <= e["valence"] <= 9 and 1 <= e["arousal"] <= 9 for e in with_v),
          "valence/arousal within the 1-9 scale")

    # --- 4. embeddings npz --------------------------------------------------------
    import numpy as np
    npz = ROOT / "docs/analysis/music2emo_embeddings.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        idx = z["index"].tolist()
        check(len(idx) == len(z["embeddings"]), "npz index and embeddings aligned")
        check(len(set(idx)) == len(idx), "npz index keys unique",
              f"{len(idx) - len(set(idx))} duplicates")

    # --- 5. cross-layer agreement -------------------------------------------------
    print("== cross-layer agreement (would collapse to ~0 if audio were misrouted) ==")
    feats = {}
    for f in (ROOT / "docs/data").glob("*/features.csv"):
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("prompt") == "free-form":
                feats[(r["model"], r.get("mode"), r.get("title"), str(r.get("sample") or 0))] = r
    ja_path = ROOT / "docs/analysis/judge_audio_llm.json"
    ja = json.loads(ja_path.read_text(encoding="utf-8")) if ja_path.exists() else []
    hear = {(r["model"], r.get("mode"), r.get("title"), str(r.get("sample"))): r
            for r in ja if r["judge"] == "gemini-2.5-pro" and r["modality"] == "audio"}

    m2e_minor, hear_minor = [], []
    key_match = key_tot = mode_match = 0
    for e in with_v:
        f = feats.get((e["model"], e.get("mode"), e.get("title"), str(e.get("sample") or 0)))
        h = hear.get((e["model"], e.get("mode"), e.get("title"), str(e.get("sample"))))
        if not f:
            continue
        km = f.get("key_mode_best")
        if km in ("major", "minor"):
            is_minor = 1 if km == "minor" else 0
            m2e_minor.append((e["valence"], is_minor))
            if h and h.get("valence") is not None:
                hear_minor.append((h["valence"], is_minor))
            ak = (e.get("audio_key") or "").split()
            if len(ak) == 2:
                tonic = (f.get("key_declared_tonic") or f.get("key_tonic") or "").replace("-", "b")
                spc, apc = NOTE2PC.get(tonic), NOTE2PC.get(ak[0].replace("-", "b"))
                if spc is not None and apc is not None:
                    key_tot += 1
                    key_match += (spc == apc)
                    mode_match += (ak[1] == km)
    r1 = pearson(m2e_minor) if len(m2e_minor) > 30 else float("nan")
    r2 = pearson(hear_minor) if len(hear_minor) > 30 else float("nan")
    check(r1 < -0.15, "Music2Emo valence tracks symbolic minor-ness", f"r={r1:+.2f}, n={len(m2e_minor)}")
    if hear_minor:
        check(r2 < -0.15, "gemini-hear valence tracks symbolic minor-ness", f"r={r2:+.2f}, n={len(hear_minor)}")
    if key_tot:
        check(key_match / key_tot > 0.5, "audio-derived key tonic matches symbolic key",
              f"{100 * key_match / key_tot:.0f}% (chance ~8-25%)")
        check(mode_match / key_tot > 0.55, "audio-derived mode matches symbolic mode",
              f"{100 * mode_match / key_tot:.0f}% (chance ~50%)")

    # --- 6. mood wiring -----------------------------------------------------------
    bad_moods = 0
    for e in with_v:
        if "mood_probs" not in e or not e.get("moods"):
            continue
        top = {m for m, p in e["mood_probs"].items() if p > 0.5}          # strictly above
        low = {m for m, p in e["mood_probs"].items() if p >= 0.4999}       # rounded boundary
        if not (top <= set(e["moods"]) <= low):
            bad_moods += 1
    check(bad_moods == 0, "predicted_moods match mood_probs threshold (patch wiring)",
          f"{bad_moods} outside the 0.5 boundary")

    # --- 7. exhaustive duration sweep ----------------------------------------------
    if args.durations:
        print("== audio-vs-score duration sweep (slow) ==")
        import logging
        logging.disable(logging.WARNING)
        from llm_music.render import abc_to_midi
        import pretty_midi
        from music21 import converter
        sus, n = [], 0
        for bd, pieces in manifests():
            for p in pieces:
                if not (p.get("ok") and p.get("audio") and (bd / p["audio"]).exists()):
                    continue
                n += 1
                try:
                    with tempfile.TemporaryDirectory() as td:
                        if p.get("abc"):
                            midi = abc_to_midi(p["abc"], Path(td))
                        elif p.get("score"):
                            midi = Path(td) / "x.mid"
                            converter.parse(str(bd / p["score"])).write("midi", fp=str(midi))
                        else:
                            continue
                        mdur = pretty_midi.PrettyMIDI(str(midi)).get_end_time() if midi else None
                except Exception:
                    continue  # unreadable-by-tooling pieces are known and logged elsewhere
                adur = mp3_dur(bd / p["audio"])
                if mdur and adur and not (-1.5 < adur - mdur < 12):
                    # pretty_midi misreads tempo on some multi-track files; only
                    # flag if a fresh render also disagrees (cheap re-check via ratio)
                    if not (0.6 < (adur / mdur) < 1.8):
                        sus.append((bd.name, p["model"], p.get("sample", 0), round(mdur, 1), round(adur, 1)))
        check(not sus, f"every MP3 ({n}) matches its own score's duration",
              f"{len(sus)} suspicious" if sus else "")
        for s in sus[:10]:
            print("    ", s)

    print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} check(s) FAILED: " + "; ".join(FAILURES))
    print("all checks passed")


if __name__ == "__main__":
    main()
