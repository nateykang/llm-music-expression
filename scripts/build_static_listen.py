#!/usr/bin/env python3
"""Build the static listening page bundle (GitHub Pages) from a suite JSON.

The zero-server fallback of the studio's Listen tab: pieces play from the
Pages site, notes live in the listener's browser (localStorage) until they
download and send the notes file. Media files are copied under opaque names
(w3p2.mp3) so URLs don't leak model identities, and the letter->model mapping
ships only as a base64 blob decoded on reveal — soft blinding, good enough
for a good-faith listener.

Usage:  python scripts/build_static_listen.py [--suite scripts/suites/20260826.json]
Writes: docs/listen/data.js + docs/listen/media/*  (index.html/app.js are static)
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODE_LABELS = {"codegen": "code", "abc": "ABC"}

# Appended to the model's code when rendering the DISPLAY score only. Three
# salvage passes for expressive content the models wrote but music21's
# MusicXML export would drop or mangle (audio/MIDI is unaffected by all three):
#   1. dynamics/text left at the Part level (outside any Measure) vanish on
#      export — move them into the measure at their offset;
#   2. Crescendo/Diminuendo spanners inserted bare (no anchored notes) export
#      as nothing — anchor them to their measure's first/last notes;
#   3. overlapping inserts without Voice objects flatten into sequential
#      8-beat measures — group them into voices.
DISPLAY_POSTLUDE = """

for _part in score.parts:
    try:
        if not _part.getElementsByClass('Measure'):
            _part.makeMeasures(inPlace=True)
    except Exception:
        pass
for _part in score.parts:
    _meas = list(_part.getElementsByClass('Measure'))
    if not _meas:
        continue
    for _dyn in list(_part.getElementsByClass(('Dynamic', 'TextExpression'))):
        try:
            _off = _part.elementOffset(_dyn)
            _target, _local = None, 0.0
            for _m in _meas:
                _mo = _part.elementOffset(_m)
                if _mo <= _off:
                    _target, _local = _m, _off - _mo
            if _target is not None:
                _part.remove(_dyn)
                _target.insert(min(_local, _target.barDuration.quarterLength), _dyn)
        except Exception:
            pass
for _sp in list(score.recurse().getElementsByClass(('Crescendo', 'Diminuendo'))):
    try:
        if not _sp.getSpannedElements():
            _site = _sp.activeSite
            _notes = list(_site.notes) if _site is not None else []
            if _notes:
                _sp.addSpannedElements([_notes[0], _notes[-1]])
    except Exception:
        pass
for _part in score.parts:
    for _m in _part.getElementsByClass('Measure'):
        try:
            _bar = _m.barDuration.quarterLength
        except Exception:
            continue
        _sum = sum(_n.quarterLength for _n in _m.notesAndRests)
        if _sum > _bar + 0.001 and not _m.voices:
            _m.makeVoices(inPlace=True)
"""


def display_score(code: str, out_path: Path) -> bool:
    """Render the engraving copy of a codegen piece from its stored source,
    with the voice-separation postlude. Returns False if the sandbox fails."""
    from llm_music.sandbox import run_music21_code

    with tempfile.TemporaryDirectory(prefix="listen_score_") as td:
        res = run_music21_code(code + DISPLAY_POSTLUDE, Path(td))
        xml = Path(td) / "piece.musicxml"
        if not res.ok or not xml.exists():
            return False
        shutil.copyfile(xml, out_path)
    return True


_manifests: dict = {}


def piece_code(data_root: Path, p: dict) -> str:
    """The stored music21 source for a sampled codegen piece."""
    name = p["batch"]
    if name not in _manifests:
        _manifests[name] = json.loads(
            (data_root / name / "data.json").read_text(encoding="utf-8"))
    for e in _manifests[name]["pieces"]:
        if (e["model"], e.get("prompt"), e.get("mode"), e.get("sample", 0)) == \
           (p["model"], p["prompt"], p["mode"], p["sample"]):
            return e["code"]
    raise SystemExit(f"piece not in manifest: {p['model']} {p['prompt']} s{p['sample']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(ROOT / "scripts/suites/20260826.json"))
    args = ap.parse_args()

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    data_root = ROOT / "docs" / "data"
    out = ROOT / "docs" / "listen"
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)

    groups: dict = {}
    windows = []
    n_files = 0
    for i, w in enumerate(suite["windows"], start=1):
        setup = w["setup"]
        # Each window's setup lists only its own models — merge into the
        # suite-wide legend, insisting the shared letters agree.
        for letter, model in setup["groups"].items():
            if groups.get(letter, model) != model:
                raise SystemExit("suite letters are not consistent across windows")
            groups[letter] = model
        pieces = []
        for p in setup["pieces"]:
            if not p["audio"]:
                raise SystemExit(f"piece without audio: {w['title']} idx {p['idx']}")
            stem = f"w{i}p{p['idx']}"
            shutil.copyfile(data_root / p["batch"] / p["audio"], media / f"{stem}.mp3")
            n_files += 1
            piece = {
                "n": p["idx"] + 1,
                "group": p["group"],
                "mode": MODE_LABELS.get(p["mode"], p["mode"]),
                "prompt": p["prompt_label"],
                "title": p["title"],
                "audio": f"media/{stem}.mp3",
            }
            if p["mode"] == "codegen":
                code = piece_code(data_root, p)
                if not display_score(code, media / f"{stem}.musicxml"):
                    raise SystemExit(f"display render failed: {w['title']} idx {p['idx']}")
                piece["score"] = f"media/{stem}.musicxml"
                n_files += 1
            elif p.get("abc"):
                piece["abc"] = p["abc"]
            pieces.append(piece)
        windows.append({"id": f"w{i}", "title": w["title"], "pieces": pieces})

    key = base64.b64encode(json.dumps(groups).encode()).decode()
    payload = {"suite": suite["seed"], "windows": windows, "key": key}
    (out / "data.js").write_text(
        "window.LISTEN_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    total_mb = sum(f.stat().st_size for f in media.iterdir()) / 1e6
    print(f"{len(windows)} windows, {n_files} media files ({total_mb:.1f} MB) -> {out}")


if __name__ == "__main__":
    main()
