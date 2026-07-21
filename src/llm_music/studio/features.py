"""Symbolic-feature measurement for studio pieces.

Reuses the batch pipeline's analyze.extract_features (same metrics, same
FEATURES_VERSION) on a single rendered version, so what the composer eyeballs
in the studio is exactly what the research analysis computes. Results are
cached as features.json inside the version dir and recomputed when the metric
definitions change (version bump).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..analyze import FEATURES_VERSION, extract_features

log = logging.getLogger(__name__)


def piece_features(vdir: Path) -> dict | None:
    """Measure (or load cached) features for one rendered version. Returns the
    feature dict, or None if the piece can't be analyzed."""
    cache = vdir / "features.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("features_version") == FEATURES_VERSION:
                return data
        except json.JSONDecodeError:
            pass  # recompute

    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    piece = {
        "model": meta.get("model", ""),
        "prompt": meta.get("prompt", ""),
        "mode": meta.get("mode"),
        "title": meta.get("title", ""),
    }
    if (vdir / "piece.musicxml").exists():
        piece["score"] = "piece.musicxml"
    elif (vdir / "piece.abc").exists():
        piece["abc"] = (vdir / "piece.abc").read_text(encoding="utf-8")
    else:
        return None

    try:
        feats = extract_features(piece, vdir)
    except Exception as e:  # noqa: BLE001 — a broken piece must not 500 the studio
        log.warning("feature extraction failed for %s: %s", vdir, e)
        return None
    if feats:
        cache.write_text(json.dumps(feats, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return feats
