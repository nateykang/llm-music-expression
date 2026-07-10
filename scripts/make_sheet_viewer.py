#!/usr/bin/env python3
"""Build docs/sheets.html — a LOCAL sheet-music viewer for the human-corpora
sample (not published; docs/sheets.html is gitignored).

Picks a few pieces per arm from the frozen sample, re-derives each score from
its deterministic id (music21 corpus path / chorale index / Arab MusicXML
file), and embeds the MusicXML into one page engraved in-browser by Verovio.
Each piece is headed with its panel-mean quality and familiarity so the score
can be eyeballed against how the judges treated it.

    python scripts/make_sheet_viewer.py
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_music.judge import QUALITY_KEYS as QUAL  # noqa: E402

PER_ARM = 3
ARM_LABEL = {
    "bach": "Bach chorales", "classical": "Classical strings",
    "irish_folk": "Irish folk", "euro_folk": "German folk (Essen)",
    "chinese_han": "Chinese Han folk", "arab_and": "Arab-Andalusian",
}


def qual(v):
    s = [v[k]["score"] for k in QUAL if k in v]
    return mean(s) if s else None


def pick_ids(sample, raw):
    """Deterministic picks; euro gets 2 major + 2 minor (one altdeu)."""
    rng = np.random.default_rng(0)
    picks = []
    by_arm = {}
    for r in sample:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm, rs in sorted(by_arm.items()):
        if arm == "euro_folk":
            majors = [r for r in rs if r["mode"] == "major"]
            minors_a = [r for r in rs if r["mode"] == "minor" and "/altdeu" in r["id"]]
            minors_o = [r for r in rs if r["mode"] == "minor" and "/altdeu" not in r["id"]]
            sel = ([majors[i] for i in rng.choice(len(majors), 2, replace=False)]
                   + [minors_a[rng.integers(len(minors_a))]]
                   + [minors_o[rng.integers(len(minors_o))]])
        else:
            sel = [rs[i] for i in rng.choice(len(rs), PER_ARM, replace=False)]
        picks += sel
    return picks


def load_score(pid):
    from music21 import converter, corpus
    from music21.corpus import chorales
    if pid.startswith("arab:"):
        stem = pid.split(":", 1)[1]
        f = ROOT / "data_external/arab-andalusian-music/scores-musicxml" / f"{stem}.xml"
        return converter.parse(str(f))
    if pid.startswith("bach:chorale#"):
        want = int(pid.rsplit("#", 1)[1])
        for i, sc in enumerate(chorales.Iterator()):
            if i == want:
                return sc
        raise KeyError(pid)
    # corpus path pieces: "<arm>:<stem>#<i>"
    stem, idx = pid.split(":", 1)[1].rsplit("#", 1)
    paths = [str(p).replace("\\", "/") for p in corpus.corpora.CoreCorpus().getPaths()]
    full = next(p for p in paths if p.endswith("/" + stem) or p.endswith(stem))
    parsed = corpus.parse(full)
    scores = (list(parsed.scores)
              if hasattr(parsed, "scores") and len(getattr(parsed, "scores", []) or []) > 0
              else [parsed])
    return scores[int(idx)]


def to_xml(sc) -> str:
    from music21.musicxml.m21ToXml import GeneralObjectExporter
    return GeneralObjectExporter().parse(sc).decode("utf-8")


def main():
    analysis = ROOT / "docs/analysis"
    sample = json.loads((analysis / "human_corpora_sample.json").read_text(encoding="utf-8"))
    raw = json.loads((analysis / "judge_human_raw.json").read_text(encoding="utf-8"))
    stats = {}
    for r in raw:
        qs = [q for q in (qual(v) for v in r["panel"].values()) if q is not None]
        fam = [v["familiarity"]["score"] for v in r["panel"].values() if "familiarity" in v]
        stats[r["id"]] = (mean(qs), mean(fam))

    blocks, xmls = [], []
    for r in pick_ids(sample, raw):
        print(f"engraving {r['id']} …", flush=True)
        try:
            xml = to_xml(load_score(r["id"]))
        except Exception as e:
            print(f"  skipped ({e})", flush=True)
            continue
        n = len(xmls)
        xmls.append(xml)
        q, fam = stats.get(r["id"], (float("nan"), float("nan")))
        leak = ' · <b>quoted text in rep</b>' if re.search(r'"[^"]{2,}"', r["rep"]) else ""
        blocks.append(
            f"<section class='piece'><h2>{html.escape(ARM_LABEL.get(r['arm'], r['arm']))} — "
            f"{html.escape(str(r['title']))}</h2>"
            f"<p class='meta'>detected key {html.escape(r['key'])} (algorithmic fit to the "
            f"notes — may differ from the engraved signature) · id {html.escape(r['id'])} · "
            f"panel quality {q:.2f} · familiarity {fam:.2f}{leak}</p>"
            f"<div class='score' id='score-{n}'>rendering…</div></section>")

    xml_scripts = "\n".join(
        f"<script type='application/xml' id='xml-{i}'>{x.replace('</script', '&lt;/script')}"
        f"</script>" for i, x in enumerate(xmls))
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Human corpora — sheet music</title>
<script src="https://www.verovio.org/javascript/latest/verovio-toolkit-wasm.js" defer></script>
<style>
 body {{ font-family: -apple-system, sans-serif; max-width: 1050px; margin: 0 auto;
        padding: 2rem 1rem; background: #faf9f6; color: #222; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.05rem; margin: 2.2rem 0 .2rem; }}
 .meta {{ color: #777; font-size: .85rem; margin: 0 0 .6rem; }}
 .score {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 8px;
          overflow-x: auto; }}
 .score svg {{ max-width: 100%; height: auto; }}
 .note {{ color: #777; font-size: .9rem; }}
</style></head><body>
<h1>Human corpora — sample sheet music (local viewer, first page of each)</h1>
<p class="note">A few pieces per corpus from the judged sample, engraved from the same
sources the note listings were made from. Not published — this file is gitignored.</p>
{"".join(blocks)}
{xml_scripts}
<script>
document.addEventListener("DOMContentLoaded", () => {{
  verovio.module.onRuntimeInitialized = () => {{
    const tk = new verovio.toolkit();
    tk.setOptions({{ scale: 33, pageWidth: 2900, adjustPageHeight: true }});
    for (let i = 0; ; i++) {{
      const el = document.getElementById("xml-" + i);
      const slot = document.getElementById("score-" + i);
      if (!el || !slot) break;
      try {{
        tk.loadData(el.textContent);
        slot.innerHTML = tk.renderToSVG(1);
        const pages = tk.getPageCount();
        if (pages > 1) slot.insertAdjacentHTML("beforeend",
          `<p class='note'>page 1 of ${{pages}}</p>`);
      }} catch (e) {{ slot.textContent = "render failed: " + e; }}
    }}
  }};
}});
</script>
</body></html>"""
    out = ROOT / "docs/sheets.html"
    out.write_text(page, encoding="utf-8")
    print(f"\nWrote {len(xmls)} pieces → {out}")


if __name__ == "__main__":
    main()
