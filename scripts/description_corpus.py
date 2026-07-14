"""Shared corpus for the description<->music experiments: the free-form
30-samples-per-composer batches (11 models x 30 in abc across three batches,
11 models x 30 in codegen), one row per ok piece, with its features.csv row
joined in when present.

    from description_corpus import load_pieces
    pieces = load_pieces(ROOT)   # [{batch, model, mode, sample, title,
                                 #   short_description, long_description,
                                 #   audio (abs Path|None), features (dict|None)}]
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

FREEFORM_BATCHES = [
    "20260622_164100__models_11_prompts_1",   # abc: opus-4.8(+thinking), sonnet-4.6
    "20260622_195241__models_7_prompts_1",    # abc: the other 7 composers
    "20260627_045417__models_1_prompts_1",    # abc: sonnet-4.6-thinking
    "20260623_105811__models_11_prompts_1",   # codegen: all 11
]


def piece_id(p: dict) -> str:
    """Stable content key: batch|model|mode|sample."""
    return f"{p['batch']}|{p['model']}|{p['mode']}|{p['sample']}"


def _features_by_key(batch_dir: Path) -> dict:
    fpath = batch_dir / "features.csv"
    if not fpath.exists():
        return {}
    out = {}
    with fpath.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["model"], row["mode"], int(row["sample"]))] = row
    return out


def load_pieces(root: Path, batches: list[str] | None = None) -> list[dict]:
    rows = []
    for batch in batches or FREEFORM_BATCHES:
        batch_dir = root / "docs/data" / batch
        data = json.loads((batch_dir / "data.json").read_text(encoding="utf-8"))
        feats = _features_by_key(batch_dir)
        for p in data["pieces"]:
            if not p.get("ok"):
                continue
            audio = batch_dir / p["audio"] if p.get("audio") else None
            rows.append({
                "batch": batch,
                "model": p["model"],
                "mode": p["mode"],
                "sample": p["sample"],
                "title": p.get("title", ""),
                "short_description": p.get("short_description", ""),
                "long_description": p.get("long_description", ""),
                "abc": p.get("abc"),
                "score": p.get("score"),
                "audio": audio if audio and audio.exists() else None,
                "features": feats.get((p["model"], p["mode"], p["sample"])),
            })
    return rows
