#!/usr/bin/env python3
"""Build the stratified sample for the human-corpora genre/mode-bias experiment.

Six arms spanning traditions and textures. Where major/minor is a native
category, sample 75 per mode (seeded); where it's modal/pentatonic music
(Chinese Han, Arab-Andalusian), take 150 with best-effort mode balance —
their "mode" is Krumhansl–Schmuckler-forced and analyzed with that caveat.

  bach        Bach chorales (music21)            75 major / 75 minor
  euro_folk   Essen European folk song           75 / 75
  irish_folk  O'Neill's 1850 (Irish)             75 / 75
  chinese_han Essen Han Chinese folk             150, best-effort split
  classical   Beethoven+Mozart+Haydn (music21)   all available (~50)
  arab_and    Arab-Andalusian MusicXML scores    150 random
              (git clone https://github.com/MTG/arab-andalusian-music
               into data_external/ first)

Writes docs/analysis/human_corpora_sample.json with the piece's blind
note-listing ALREADY RENDERED, so the judging script needs no corpora
installed — the manifest is the experiment's frozen input.

    python scripts/sample_human_corpora.py
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from llm_music.judge import _score_to_text  # noqa: E402

X = 75  # per mode, where mode is a native category
SEED = 0

# Essen filename stems for the European arm (German-language collections);
# han* is the Chinese arm; test*/variant/irl/folkHaydn excluded.
EURO_STEMS = ("altdeu", "ballad", "boehme", "dva", "erk", "fink",
              "kinder", "lot", "lux", "zuccal")


def pieces_from_corpus_paths(paths, arm):
    """Yield (id, title, score) from music21 corpus paths, expanding Opus files."""
    from music21 import corpus
    for p in paths:
        try:
            parsed = corpus.parse(p)
        except Exception:
            continue
        scores = (list(parsed.scores)
                  if hasattr(parsed, "scores") and len(getattr(parsed, "scores", []) or []) > 0
                  else [parsed])
        stem = str(p).replace("\\", "/").split("/corpus/")[-1]
        for i, sc in enumerate(scores):
            title = (sc.metadata.title if sc.metadata and sc.metadata.title else "") or f"{stem}#{i}"
            yield f"{arm}:{stem}#{i}", title, sc


def keyed(items):
    """Attach detected key/mode; drop unanalyzable pieces."""
    out = []
    for pid, title, sc in items:
        try:
            k = sc.analyze("key")
        except Exception:
            continue
        out.append((pid, title, f"{k.tonic.name} {k.mode}", k.mode, sc))
    return out


def render(entries, arm):
    rows = []
    for pid, title, key, mode, sc in entries:
        try:
            rep = _score_to_text(sc, label=pid)
        except Exception:
            continue
        rows.append({"arm": arm, "id": pid, "title": title, "key": key,
                     "mode": mode, "rep": rep})
    return rows


def stratified(entries, rng, per_mode=X):
    maj = [e for e in entries if e[3] == "major"]
    minr = [e for e in entries if e[3] == "minor"]
    rng.shuffle(maj), rng.shuffle(minr)
    if len(maj) >= per_mode and len(minr) >= per_mode:
        return maj[:per_mode] + minr[:per_mode]
    # best effort: all of the scarcer mode, fill with the other up to 2*per_mode
    short, longer = (maj, minr) if len(maj) < len(minr) else (minr, maj)
    return short + longer[:2 * per_mode - len(short)]


def main():
    from music21 import corpus
    from music21.corpus import chorales

    rng = random.Random(SEED)
    all_paths = [str(p).replace("\\", "/") for p in corpus.corpora.CoreCorpus().getPaths()]
    sample: list[dict] = []

    def essen_paths(pred):
        return [p for p in all_paths if "/essenFolksong/" in p
                and pred(p.split("/essenFolksong/")[-1])]

    print("bach…", flush=True)
    bach = keyed((f"bach:chorale#{i}",
                  (sc.metadata.title if sc.metadata and sc.metadata.title else f"chorale-{i}"),
                  sc) for i, sc in enumerate(chorales.Iterator()))
    sample += render(stratified(bach, rng), "bach")

    print("euro_folk…", flush=True)
    euro = keyed(pieces_from_corpus_paths(
        essen_paths(lambda f: f.startswith(EURO_STEMS)), "euro"))
    sample += render(stratified(euro, rng), "euro_folk")

    print("irish_folk…", flush=True)
    irish = keyed(pieces_from_corpus_paths(
        [p for p in all_paths if "/oneills1850/" in p], "irish"))
    sample += render(stratified(irish, rng), "irish_folk")

    print("chinese_han…", flush=True)
    han = keyed(pieces_from_corpus_paths(
        essen_paths(lambda f: f.startswith("han")), "han"))
    sample += render(stratified(han, rng), "chinese_han")

    print("classical…", flush=True)
    cls = keyed(pieces_from_corpus_paths(
        [p for p in all_paths if any(f"/{c}/" in p for c in ("beethoven", "mozart", "haydn"))],
        "classical"))
    rng.shuffle(cls)
    sample += render(cls[:2 * X], "classical")

    print("arab_and…", flush=True)
    from music21 import converter
    xml_dir = ROOT / "data_external/arab-andalusian-music/scores-musicxml"
    if not xml_dir.exists():
        sys.exit("missing data_external/arab-andalusian-music — clone the MTG repo first")
    files = sorted(xml_dir.glob("*.xml"))
    rng.shuffle(files)
    arab = []
    for f in files:
        if len(arab) >= 150:
            break
        try:
            sc = converter.parse(str(f))
            k = sc.analyze("key")
        except Exception:
            continue
        arab.append((f"arab:{f.stem}", f.stem, f"{k.tonic.name} {k.mode}", k.mode, sc))
    sample += render(arab, "arab_and")

    out = ROOT / "docs/analysis/human_corpora_sample.json"
    out.write_text(json.dumps(sample, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"\nwrote {len(sample)} pieces → {out}")
    for arm in ("bach", "euro_folk", "irish_folk", "chinese_han", "classical", "arab_and"):
        rows = [r for r in sample if r["arm"] == arm]
        modes = Counter(r["mode"] for r in rows)
        print(f"  {arm:12s} n={len(rows):3d}  {dict(modes)}")


if __name__ == "__main__":
    main()
