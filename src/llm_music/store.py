"""Bake generation results into the static-site layout + data.json manifest.

Layout (mirrors the original sara-fish site, with MusicXML instead of baked PDF):

  docs/data/<ts>__models_N_prompts_M/
    data.json
    audio/<prompt>/<model>.ogg
    scores/<prompt>/<model>.musicxml
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import DATA_DIR
from .generate import PieceResult


def batch_dir_name(timestamp: str, n_models: int, n_prompts: int) -> str:
    return f"{timestamp}__models_{n_models}_prompts_{n_prompts}"


def open_batch(
    timestamp: str,
    models: list[str],
    prompts: list[str],
    base_dir: Path | None = None,
) -> Path:
    """Create the batch folder (and an empty manifest) so results can be written
    incrementally. Returns the folder path."""
    base = base_dir or DATA_DIR
    batch = base / batch_dir_name(timestamp, len(models), len(prompts))
    (batch / "audio").mkdir(parents=True, exist_ok=True)
    (batch / "scores").mkdir(parents=True, exist_ok=True)
    write_manifest(batch, timestamp, models, prompts, [])
    return batch


def append_result(batch: Path, result: PieceResult, sample: int = 0) -> dict:
    """Copy one piece's outputs into the batch folder and return its manifest entry.

    `sample` is the repeat index for a (model, prompt) cell when generating many
    samples; it suffixes the stored filenames so samples don't overwrite each other.
    """
    r = result
    suffix = f"_s{sample}" if sample else ""
    entry = {
        "model": r.model,
        "prompt": r.prompt,
        "prompt_label": r.prompt_label,
        "mode": r.mode,
        "sample": sample,
        "ok": r.ok,
        "prompt_text": r.prompt_text,
        "system_prompt": r.system_prompt,
        "title": r.title,
        "short_description": r.short_description,
        "long_description": r.long_description,
        "attempts": r.attempts,
    }
    if r.abc:
        entry["abc"] = r.abc
    if r.ok and r.musicxml_path:
        score_rel = f"scores/{r.prompt}/{r.model}{suffix}.musicxml"
        _copy(r.musicxml_path, batch / score_rel)
        entry["score"] = score_rel
    if r.ok and r.audio_path:
        audio_rel = f"audio/{r.prompt}/{r.model}{suffix}.mp3"
        _copy(r.audio_path, batch / audio_rel)
        entry["audio"] = audio_rel
    if not r.ok:
        entry["error"] = r.error
    return entry


def write_manifest(
    batch: Path,
    timestamp: str,
    models: list[str],
    prompts: list[str],
    entries: list[dict],
) -> None:
    """(Re)write data.json for a batch and refresh the batch index. Safe to call
    after every piece so an interrupted run still leaves a valid partial batch."""
    manifest = {
        "timestamp": timestamp,
        "models": models,
        "prompts": prompts,
        "pieces": entries,
    }
    (batch / "data.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _update_index(batch.parent)


def write_results(
    results: list[PieceResult],
    timestamp: str,
    models: list[str],
    prompts: list[str],
    base_dir: Path | None = None,
) -> Path:
    """Copy outputs into a batch folder and write data.json in one shot. Returns
    the folder. (Incremental callers use open_batch/append_result/write_manifest.)"""
    batch = open_batch(timestamp, models, prompts, base_dir)
    entries = [append_result(batch, r) for r in results]
    write_manifest(batch, timestamp, models, prompts, entries)
    return batch


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _update_index(base: Path) -> None:
    """Write data/index.json listing available batches (newest first)."""
    batches = sorted(
        (p.name for p in base.iterdir() if p.is_dir() and (p / "data.json").exists()),
        reverse=True,
    )
    (base / "index.json").write_text(
        json.dumps({"batches": batches}, indent=2), encoding="utf-8"
    )
