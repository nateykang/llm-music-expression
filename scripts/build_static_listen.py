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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODE_LABELS = {"codegen": "code", "abc": "ABC"}


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
            if p["mode"] == "codegen" and p.get("score"):
                shutil.copyfile(data_root / p["batch"] / p["score"],
                                media / f"{stem}.musicxml")
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
