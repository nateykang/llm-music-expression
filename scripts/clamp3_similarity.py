#!/usr/bin/env python3
"""Description<->score correspondence in CLaMP 3's joint space — the symbolic
twin of clap_similarity.py, with no audio synthesis anywhere in the loop.

Protocol follows Text2Score (arXiv:2605.13431), which scores prompt adherence
as CLaMP 3 similarity between verbatim text and the piece as MIDI: each score
is converted to MIDI (abc2midi for abc-mode with gchords off, matching the
feature-extraction convention; music21 for codegen MusicXML), then to MTF via
CLaMP 3's batch_midi2mtf.py --m3_compatible, and embedded with the symbolic
encoder. We use the C2 checkpoint (the variant CLaMP 3 documents for symbolic
retrieval). Unlike Text2Score's user prompts, our text is the model's OWN
description (faithfulness, not adherence), at three granularities: "title",
"short" (title + short description, matching clap_similarity), "long" (the
full reflection).

Corpus: the free-form 30-samples-per-composer batches (see description_corpus).

Requires a sibling checkout of https://github.com/sanderwood/clamp3 at
~/clamp3 with its own venv (.venv) and the C2 checkpoint in clamp3/code/
(config.py switched to "c2").

    python scripts/clamp3_similarity.py --limit 6    # smoke test
    python scripts/clamp3_similarity.py              # full run (resumable)

Writes docs/analysis/clamp3_embeddings.npz and docs/analysis/clamp3_similarity.json.
Extraction caches: docs/analysis/embedding/clamp3/{music,title,short,long}/*.npy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from clap_similarity import retrieval_stats, summarize, print_summary  # noqa: E402
from description_corpus import load_pieces, piece_id  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
FEAT = ANALYSIS / "embedding" / "clamp3"
CLAMP3 = Path.home() / "clamp3"
CLAMP3_PY = CLAMP3 / ".venv" / "bin" / "python"
STAGE = Path("/private/tmp/claude-501/-Users-nathanielkang-llm-music-expression"
             "/98513b28-b0b9-4476-9ed6-b2393dc80999/scratchpad/clamp3_stage")
TEXT_VARIANTS = ("title", "short", "long")


def safe_id(pid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", pid)


def texts_for(p: dict) -> dict:
    return {
        "title": p["title"] or p["short_description"],
        "short": f"{p['title']}. {p['short_description']}",
        "long": p["long_description"] or p["short_description"],
    }


def stage_midi(pieces: list[dict]) -> Path:
    """One MIDI file per piece under STAGE/midi (resumable)."""
    from llm_music.render import abc_to_midi

    midi_dir = STAGE / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    work = STAGE / "midi_work"
    failed = []
    todo = [p for p in pieces
            if not (midi_dir / f"{safe_id(piece_id(p))}.mid").exists()]
    print(f"midi: {len(pieces) - len(todo)} staged, {len(todo)} to convert",
          flush=True)
    for k, p in enumerate(todo):
        sid = safe_id(piece_id(p))
        try:
            if p["mode"] == "codegen":
                import music21

                score = music21.converter.parse(
                    ROOT / "docs/data" / p["batch"] / p["score"])
                score.write("midi", fp=midi_dir / f"{sid}.mid")
            else:
                if work.exists():
                    shutil.rmtree(work)
                out = abc_to_midi(p["abc"], work, gchords=False)
                if out is None:
                    raise RuntimeError("abc2midi produced no MIDI")
                shutil.move(out, midi_dir / f"{sid}.mid")
        except Exception as e:  # noqa: BLE001 — collect, report, move on
            failed.append((piece_id(p), str(e)[:200]))
        if (k + 1) % 50 == 0 or k + 1 == len(todo):
            print(f"  [{k + 1}/{len(todo)}]", flush=True)
    if failed:
        print(f"midi conversion failed for {len(failed)} pieces:", flush=True)
        for pid, err in failed[:10]:
            print(f"  {pid}: {err}", flush=True)
    return midi_dir


def midi_to_mtf(midi_dir: Path) -> Path:
    mtf_dir = STAGE / "mtf"
    if mtf_dir.exists():
        shutil.rmtree(mtf_dir)
    mtf_dir.mkdir(parents=True)
    r = subprocess.run([str(CLAMP3_PY), "batch_midi2mtf.py", str(midi_dir),
                        str(mtf_dir), "--m3_compatible"],
                       cwd=CLAMP3 / "preprocessing" / "midi",
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"batch_midi2mtf failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    n = len(list(mtf_dir.glob("*.mtf")))
    print(f"mtf: {n} files from {len(list(midi_dir.glob('*.mid')))} MIDI",
          flush=True)
    return mtf_dir


def stage_texts(pieces: list[dict]) -> dict:
    dirs = {}
    for variant in TEXT_VARIANTS:
        d = STAGE / f"text_{variant}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for p in pieces:
            text = " ".join(texts_for(p)[variant].split())  # single line
            (d / f"{safe_id(piece_id(p))}.txt").write_text(text, encoding="utf-8")
        dirs[variant] = d
    return dirs


def extract(in_dir: Path, out_dir: Path) -> None:
    """Run CLaMP 3 feature extraction for files missing from out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in out_dir.glob("*.npy")}
    todo_dir = STAGE / f"todo_{out_dir.name}"
    if todo_dir.exists():
        shutil.rmtree(todo_dir)
    todo_dir.mkdir(parents=True)
    todo = [f for f in sorted(in_dir.iterdir()) if f.stem not in done]
    print(f"[{out_dir.name}] {len(done)} cached, {len(todo)} to embed", flush=True)
    if not todo:
        return
    for f in todo:
        shutil.copy(f, todo_dir / f.name)
    # The machine's stored HF token fails signature checks; don't send it for
    # the public xlm-roberta-base download.
    env = {**os.environ, "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}
    r = subprocess.run([str(CLAMP3_PY), "extract_clamp3.py", str(todo_dir),
                        str(out_dir), "--get_global"],
                       cwd=CLAMP3 / "code", capture_output=True, text=True,
                       env=env)
    if r.returncode != 0:
        raise RuntimeError(f"extract_clamp3 failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    print(f"[{out_dir.name}] extraction done: "
          f"{len(list(out_dir.glob('*.npy')))} total", flush=True)


def load_matrix(out_dir: Path, sids: list[str]) -> np.ndarray:
    X = np.zeros((len(sids), 768), dtype=np.float32)
    for i, sid in enumerate(sids):
        v = np.load(out_dir / f"{sid}.npy").reshape(-1)
        X[i] = v / np.linalg.norm(v)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N pieces")
    args = ap.parse_args()

    pieces = load_pieces(ROOT)
    if args.limit:
        pieces = pieces[: args.limit]
    print(f"{len(pieces)} corpus pieces", flush=True)

    midi_dir = stage_midi(pieces)
    mtf_dir = midi_to_mtf(midi_dir)
    text_dirs = stage_texts(pieces)

    extract(mtf_dir, FEAT / "music")
    for variant in TEXT_VARIANTS:
        extract(text_dirs[variant], FEAT / variant)

    # Keep only pieces whose score survived conversion + extraction.
    kept, dropped = [], []
    for p in pieces:
        sid = safe_id(piece_id(p))
        if (FEAT / "music" / f"{sid}.npy").exists():
            kept.append(p)
        else:
            dropped.append(piece_id(p))
    print(f"{len(kept)} pieces embedded, {len(dropped)} dropped in conversion",
          flush=True)

    sids = [safe_id(piece_id(p)) for p in kept]
    M = load_matrix(FEAT / "music", sids)
    T = {v: load_matrix(FEAT / v, sids) for v in TEXT_VARIANTS}

    # Health probe: a degenerate encoder maps everything to one direction
    # (seen with laion/larger_clap_music) — catch it before trusting results.
    off = (M @ M.T)[~np.eye(len(M), dtype=bool)]
    assert off.mean() < 0.95, f"music embeddings degenerate: mean off-diag cos={off.mean():.3f}"

    np.savez_compressed(ANALYSIS / "clamp3_embeddings.npz",
                        ids=np.array([piece_id(p) for p in kept]),
                        music=M, **{f"text_{v}": T[v] for v in TEXT_VARIANTS})

    result = summarize(kept, M, T)
    result["model"] = "clamp3-c2 (symbolic; MIDI->MTF per Text2Score protocol)"
    result["music_offdiag_cos_mean"] = round(float(off.mean()), 4)
    result["dropped"] = dropped
    (ANALYSIS / "clamp3_similarity.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    print_summary(result)
    print(f"\nWrote clamp3_embeddings.npz + clamp3_similarity.json -> {ANALYSIS}")


if __name__ == "__main__":
    main()
