#!/usr/bin/env python3
"""Build and install the composer's listening-suite: a set of small, focused
blind listening sessions ("windows") over the final corpus, instead of one
mixed queue.

The suite (for models M1..M3 across an ABC batch and a codegen batch):
  - "Code vs ABC — Model X — P"        one window per model, each on a single
                                       prompt (window i gets prompt i): one
                                       codegen piece + one ABC piece (2 pieces —
                                       deliberately light on the listener)
  - "Model comparison (ABC|code) — P"  one window per (method, prompt); one
                                       piece per model (3 pieces), so a composer
                                       who prefers one method can focus on it

Model letters are assigned once per suite and reused in every window, so notes
compose across windows ("Model A breathes more in ABC"). All windows are blind;
revealing any window reveals the shared letters, so reveal after the last one.

Two subcommands, because the data and the studio may live on different hosts:

  build   (on the host holding the full batches) — samples every window without
          piece overlap, renders any missing MP3s (soundfont required), writes
          the suite JSON + the list of files the studio host needs.
  create  (on the studio host, after copying those files into docs/data) —
          writes the sessions into the studio's session store, newest-last so
          the suite reads top-to-bottom in the Listen tab.

Usage:
  python scripts/listening_suite.py build --abc-batch <name> --codegen-batch <name> \
      --models fable-5,gpt-5.6,gemini-3.7-flash [--seed 20260820] [--no-render] \
      [--out /tmp/suite.json] [--files-out /tmp/suite_files.txt]
  python scripts/listening_suite.py create --suite /tmp/suite.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODE_LABELS = {"codegen": "code", "abc": "ABC"}


def _window(root, used, title, batches, models, seed, prompts=None):
    """One window, resampled with a shifted seed until it shares no piece with
    any earlier window. When a cell's pool is too small for that (a model with
    one piece per prompt), accept the overlap with a warning rather than fail."""
    from llm_music.studio.review import build_setup

    setup = None
    for attempt in range(20):
        setup = build_setup(root, batches, models, per_cell=1,
                            seed=seed + 1000 * attempt, blind=True, prompts=prompts)
        keys = {(p["batch"], p["model"], p["prompt"], p["mode"], p["sample"])
                for p in setup["pieces"]}
        if not keys & used:
            break
    else:
        print(f"  note: {title!r} repeats {len(keys & used)} piece(s) heard in "
              f"another window (pool too small to avoid)", file=sys.stderr)
    used |= keys
    return {"title": title, "setup": setup}


def _relabel(setup, letters):
    """Replace the window's own group labels with the suite-wide ones."""
    setup["groups"] = {letters[m]: m for m in {p["model"] for p in setup["pieces"]}}
    for p in setup["pieces"]:
        p["group"] = letters[p["model"]]


def build(args):
    from llm_music.studio.review import load_manifest

    root = Path(args.data_root)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rng = random.Random(args.seed)
    order = models[:]
    rng.shuffle(order)
    letters = {m: f"Model {chr(ord('A') + i)}" for i, m in enumerate(order)}

    abc_manifest = load_manifest(root, args.abc_batch)
    prompts = abc_manifest.get("prompts", [])
    labels = {p.get("prompt"): p.get("prompt_label") or p.get("prompt")
              for p in abc_manifest.get("pieces", [])}

    used: set = set()
    windows = []
    for i, model in enumerate(order):
        prompt = prompts[i % len(prompts)]
        windows.append(_window(root, used,
                               f"Code vs ABC — {letters[model]} — {labels.get(prompt, prompt)}",
                               [args.abc_batch, args.codegen_batch], [model],
                               args.seed + i + 1, prompts=[prompt]))
    for mode, batch in (("abc", args.abc_batch), ("codegen", args.codegen_batch)):
        for j, prompt in enumerate(prompts):
            title = (f"Model comparison ({MODE_LABELS[mode]}) — "
                     f"{labels.get(prompt, prompt)}")
            windows.append(_window(root, used, title, [batch], models,
                                   args.seed + 100 + 10 * j + (0 if mode == "abc" else 5),
                                   prompts=[prompt]))
    for w in windows:
        _relabel(w["setup"], letters)

    if not args.no_render:
        _render_missing(root, windows)

    suite = {"seed": args.seed, "letters": letters, "windows": windows}
    Path(args.out).write_text(json.dumps(suite), encoding="utf-8")
    files = {f"{args.codegen_batch}/data.json"}
    for w in windows:
        for p in w["setup"]["pieces"]:
            for k in ("audio", "score"):
                if p[k]:
                    files.add(p["batch"] + "/" + p[k])
    Path(args.files_out).write_text("\n".join(sorted(files)) + "\n", encoding="utf-8")
    n = sum(len(w["setup"]["pieces"]) for w in windows)
    missing = [(w["title"], p["idx"]) for w in windows
               for p in w["setup"]["pieces"] if not p["audio"]]
    print(f"suite: {len(windows)} windows, {n} pieces -> {args.out}")
    print(f"files for the studio host ({len(files)}): {args.files_out}")
    if missing:
        print(f"WARNING: {len(missing)} pieces still lack audio: {missing}")
    return 1 if missing else 0


def _audio_abc(abc: str) -> str:
    """abc2midi rejects octave-up marks on uppercase notes (D' etc.) and plays
    them an octave low; abcjs engraves them as written. Rewrite D' -> d (same
    pitch) for the AUDIO path only, so score and sound agree."""
    import re
    return re.sub(r"([A-G])'", lambda m: m.group(1).lower(), abc)


def _render_missing(root, windows):
    """MP3s for sampled pieces that lack them (writes audio files only — never
    a batch data.json, which a running batch may be rewriting)."""
    from music21 import converter

    from llm_music.render import abc_to_midi, audio_available, midi_to_audio

    todo = [p for w in windows for p in w["setup"]["pieces"] if not p["audio"]]
    if not todo:
        return
    if not audio_available():
        print("fluidsynth/lame/soundfont missing — cannot render "
              f"{len(todo)} pieces (run scripts/setup_soundfont.sh)", file=sys.stderr)
        return
    for p in todo:
        suffix = "_s" + str(p["sample"]) if p["sample"] else ""
        arel = "audio/" + p["prompt"] + "/" + p["model"] + suffix + ".mp3"
        target = root / p["batch"] / arel
        target.parent.mkdir(parents=True, exist_ok=True)
        ok = target.exists()
        if not ok:
            with tempfile.TemporaryDirectory() as td:
                midi = None
                try:
                    if p.get("abc"):
                        midi = abc_to_midi(_audio_abc(p["abc"]), Path(td))
                    elif p.get("score"):
                        midi = Path(td) / "piece.mid"
                        converter.parse(str(root / p["batch"] / p["score"])).write(
                            "midi", fp=str(midi))
                except Exception as e:
                    print("  midi failed:", p["model"], p["prompt"], e)
                ok = bool(midi) and Path(midi).exists() and midi_to_audio(Path(midi), target)
        if ok:
            p["audio"] = arel
        print(("  rendered " if ok else "  FAILED   ") + f"{p['mode']:7s} {p['model']} {p['prompt']}")


def create(args):
    from llm_music.studio import config as cfg
    from llm_music.studio import review as review_mod
    from llm_music.studio.sessions import SessionStore

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    store = SessionStore(cfg.data_dir())
    problems = 0
    # Newest-last: the Listen tab orders windows by creation time (newest
    # first, immutable), so creating in reverse shows the suite top-to-bottom
    # in its intended order — and it stays put as listeners work.
    for w in reversed(suite["windows"]):
        setup = w["setup"]
        for p in setup["pieces"]:
            for kind in ("audio", "score"):
                if not p[kind]:
                    continue
                try:
                    path = review_mod.resolve_batch_file(
                        cfg.batches_dir(), p["batch"], p[kind])
                    if not path.exists():
                        raise KeyError("missing on disk")
                except KeyError as e:
                    print(f"  FILE PROBLEM in {w['title']!r} idx {p['idx']}: {e}")
                    problems += 1
        meta = store.create(model="", title=w["title"], kind="review")
        store.append_event(meta["id"], {"type": "review_setup", **setup})
        store.touch(meta["id"], batches=setup["batches"], blind=True,
                    revealed=False, n_pieces=len(setup["pieces"]))
        print(f"  created {meta['id']}  {w['title']}  ({len(setup['pieces'])} pieces)")
    print("file problems:", problems)
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sample the suite where the batch data lives")
    b.add_argument("--abc-batch", required=True)
    b.add_argument("--codegen-batch", required=True)
    b.add_argument("--models", required=True, help="comma-separated model ids")
    b.add_argument("--seed", type=int, default=20260820)
    b.add_argument("--data-root", default=str(ROOT / "docs" / "data"))
    b.add_argument("--out", default="/tmp/suite.json")
    b.add_argument("--files-out", default="/tmp/suite_files.txt")
    b.add_argument("--no-render", action="store_true")
    b.set_defaults(fn=build)
    c = sub.add_parser("create", help="write the sessions on the studio host")
    c.add_argument("--suite", default="/tmp/suite.json")
    c.set_defaults(fn=create)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
