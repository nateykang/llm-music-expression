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


def write_results(
    results: list[PieceResult],
    timestamp: str,
    models: list[str],
    prompts: list[str],
    base_dir: Path | None = None,
) -> Path:
    """Copy outputs into a batch folder and write data.json. Returns the folder."""
    base = base_dir or DATA_DIR
    batch = base / batch_dir_name(timestamp, len(models), len(prompts))
    (batch / "audio").mkdir(parents=True, exist_ok=True)
    (batch / "scores").mkdir(parents=True, exist_ok=True)

    entries = []
    for r in results:
        entry = {
            "model": r.model,
            "prompt": r.prompt,
            "mode": r.mode,
            "ok": r.ok,
            "title": r.title,
            "short_description": r.short_description,
            "long_description": r.long_description,
            "attempts": r.attempts,
        }
        if r.ok and r.musicxml_path:
            score_rel = f"scores/{r.prompt}/{r.model}.musicxml"
            _copy(r.musicxml_path, batch / score_rel)
            entry["score"] = score_rel
        if r.ok and r.audio_path:
            audio_rel = f"audio/{r.prompt}/{r.model}.ogg"
            _copy(r.audio_path, batch / audio_rel)
            entry["audio"] = audio_rel
        if not r.ok:
            entry["error"] = r.error
        entries.append(entry)

    manifest = {
        "timestamp": timestamp,
        "models": models,
        "prompts": prompts,
        "pieces": entries,
    }
    (batch / "data.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _update_index(base)
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
