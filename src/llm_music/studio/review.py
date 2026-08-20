"""Listening sessions: a sampled slice of a generation batch served to the
composer for qualitative review.

A review session is a fixed queue of already-generated pieces pulled straight
from a batch under docs/data/ (the layout store.py writes) — no model is ever
called. The composer listens, writes per-piece thoughts and overall
comparison notes; everything lands in the session's events.jsonl. While the
session is blind, model identities hide behind anonymous per-model labels
("Model A") until an explicit, logged reveal — so notes written blind can be
told apart from notes written after.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

_BATCH_RE = re.compile(r"^\d{8}_\d{6}__models_\d+_prompts_\d+$")
_FILE_SUFFIXES = {".mp3", ".musicxml"}

MAX_PIECES = 30  # keep one session listenable in a sitting


def batch_dir(root: Path, name: str) -> Path:
    # The name doubles as a path component; the strict format check is what
    # makes traversal impossible.
    if not _BATCH_RE.match(name):
        raise KeyError(f"bad batch name: {name!r}")
    d = root / name
    if not (d / "data.json").exists():
        raise KeyError(f"no such batch: {name!r}")
    return d


def load_manifest(root: Path, name: str) -> dict:
    return json.loads((batch_dir(root, name) / "data.json").read_text(encoding="utf-8"))


def list_batches(root: Path) -> list[dict]:
    """Newest-first summaries for the picker: which models have ok pieces."""
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not (_BATCH_RE.match(d.name) and (d / "data.json").exists()):
            continue
        try:
            manifest = json.loads((d / "data.json").read_text(encoding="utf-8"))
        except Exception:
            continue  # half-written manifest (e.g. a batch mid-sync): skip, don't 500
        counts: dict[str, int] = {}
        modes = set()
        for p in manifest.get("pieces", []):
            if p.get("ok"):
                counts[p["model"]] = counts.get(p["model"], 0) + 1
                modes.add(p.get("mode", "?"))
        out.append({
            "name": d.name,
            "timestamp": manifest.get("timestamp", d.name.split("__")[0]),
            "prompts": manifest.get("prompts", []),
            "modes": sorted(modes),
            "model_counts": counts,
        })
    return out


def build_setup(root: Path, batches: list[str], models: list[str], per_cell: int,
                seed: int, blind: bool, prompts: list[str] | None = None) -> dict:
    """Sample the review queue: per_cell random ok pieces from every
    (model × prompt × writing method) cell found across the given batches —
    stratified, so a multi-prompt or two-method session comes out balanced.
    ``prompts`` narrows the session to those prompts (default: all found).
    Pieces are ordered prompt by prompt (shuffled within each prompt), and
    anonymous group labels are assigned per model. The returned dict is the
    session's review_setup event — the full unblinded record; what the browser
    may see is client_view()'s business."""
    rng = random.Random(seed)
    cells: dict[tuple, list[dict]] = {}  # (model, prompt, mode) -> pieces
    prompt_order: list[str] = []
    for batch in batches:
        manifest = load_manifest(root, batch)
        for prompt in manifest.get("prompts", []):
            if prompt not in prompt_order and (not prompts or prompt in prompts):
                prompt_order.append(prompt)
        for p in manifest.get("pieces", []):
            if prompts and p.get("prompt") not in prompts:
                continue
            if p.get("ok") and p.get("model") in models:
                key = (p["model"], p.get("prompt"), p.get("mode"))
                cells.setdefault(key, []).append({**p, "batch": batch})
    covered = {key[0] for key in cells}
    missing = [m for m in models if m not in covered]
    if missing:
        raise ValueError(
            f"no successful pieces for: {', '.join(missing)}")
    chosen: list[dict] = []
    for key in sorted(cells, key=lambda k: (models.index(k[0]),
                                            prompt_order.index(k[1]), k[2])):
        pool = cells[key]
        chosen.extend(rng.sample(pool, min(per_cell, len(pool))))
    if len(chosen) > MAX_PIECES:
        raise ValueError(f"{len(chosen)} pieces is too many for one sitting — "
                         f"the limit is {MAX_PIECES}")
    # Prompt-by-prompt listening order (same brief back to back), blind-shuffled
    # within each prompt.
    ordered: list[dict] = []
    for prompt in prompt_order:
        bucket = [p for p in chosen if p.get("prompt") == prompt]
        rng.shuffle(bucket)
        ordered.extend(bucket)
    order = sorted({p["model"] for p in ordered})
    if len(order) > 26:
        raise ValueError("too many models for one-letter labels")
    rng.shuffle(order)  # so "Model A" isn't just alphabetically first
    groups = {model: f"Model {chr(ord('A') + i)}" for i, model in enumerate(order)}
    pieces = [{
        "idx": i,
        "batch": p["batch"],
        "group": groups[p["model"]],
        "model": p["model"],
        "prompt": p.get("prompt"),
        "prompt_label": p.get("prompt_label") or p.get("prompt"),
        "mode": p.get("mode"),
        "sample": p.get("sample", 0),
        "title": p.get("title") or "",
        "audio": p.get("audio"),
        "score": p.get("score"),
        "abc": p.get("abc"),
    } for i, p in enumerate(ordered)]
    return {"batches": list(batches), "blind": blind, "seed": seed,
            "groups": {label: model for model, label in groups.items()},
            "pieces": pieces}


def client_view(setup: dict, revealed: bool) -> dict:
    """The browser-safe projection of a setup event. While blind, model
    identities never cross the wire — not even file paths, which embed model
    names; files are served by piece index instead."""
    pieces = []
    for p in setup["pieces"]:
        view = {"idx": p["idx"], "group": p["group"], "mode": p["mode"],
                "prompt_label": p["prompt_label"], "title": p["title"],
                "has_audio": bool(p["audio"]), "has_score": bool(p["score"]),
                "has_abc": bool(p["abc"])}
        if revealed:
            view["model"] = p["model"]
            view["sample"] = p["sample"]
        pieces.append(view)
    out = {"pieces": pieces, "revealed": revealed}
    if revealed:
        out["groups"] = setup["groups"]
    return out


def resolve_batch_file(root: Path, batch: str, rel: str) -> Path:
    """Turn a manifest-relative path (audio/..., scores/...) into a real file,
    refusing anything that escapes the batch folder or isn't a piece asset."""
    d = batch_dir(root, batch)
    path = (d / rel).resolve()
    if not path.is_relative_to(d.resolve()) or path.suffix not in _FILE_SUFFIXES:
        raise KeyError(f"bad piece file: {rel!r}")
    return path
