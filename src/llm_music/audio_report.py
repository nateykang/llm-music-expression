"""Generate docs/audio.html — the audio-emotion analysis page (fourth site tab).

Reads docs/analysis/music2emo_full.json (Music2Emo valence/arousal + mood probs +
audio-derived key/chords, plus a librosa acoustic suite), features.csv (computed
proxies) and judge_allmodels_raw.json (judge-perceived valence). Renders the AUDIO
leg of the 3-way convergent-validity triangulation. Everything here is measured on
FluidSynth-synthesized audio, which is out-of-distribution for MERT — read as a
cross-check, not ground truth.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .judge_report import _f, _pearson, _table
from .report import ACCENT, BG, INK, MUTED


def _fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v * 100:.0f}%" if pct else f"{v:.{nd}f}"


def _mode_word(audio_key: str) -> str:
    parts = (audio_key or "").split()
    return parts[-1].lower() if len(parts) == 2 else ""


def render_audio_html(analysis: Path, data_dir: Path, out_path: Path) -> Path:
    m2e_path = analysis / "music2emo_full.json"
    if not m2e_path.exists():
        out_path.write_text("<p>No audio results. Run the Music2Emo batch first.</p>", encoding="utf-8")
        return out_path
    m2e = [e for e in json.loads(m2e_path.read_text(encoding="utf-8")) if "valence" in e]

    # computed metrics + judge valence, indexed by (model, mode, title)
    feats = {}
    for csvf in sorted(data_dir.glob("*/features.csv")):
        for r in csv.DictReader(csvf.open(encoding="utf-8")):
            if r.get("prompt") == "free-form":
                feats[(r["model"], r.get("mode"), r.get("title"))] = r
    judge_val = {}
    jr = analysis / "judge_allmodels_raw.json"
    if jr.exists():
        for p in json.loads(jr.read_text(encoding="utf-8")):
            if p.get("prompt") != "free-form":
                continue
            vs = [v["valence"]["score"] for v in p["panel"].values()
                  if isinstance(v.get("valence"), dict) and "score" in v["valence"]]
            if vs:
                judge_val[(p["model"], p.get("mode"), p.get("title"))] = mean(vs)

    rows = []
    for e in m2e:
        k = (e["model"], e.get("mode"), e.get("title"))
        f = feats.get(k)
        e = dict(e)
        e["_cv"] = _f(f.get("valence")) if f else None
        e["_ca"] = _f(f.get("arousal")) if f else None
        e["_ctempo"] = _f((f or {}).get("tempo_bpm") or (f or {}).get("tempo")) if f else None
        e["_cmode"] = ((f or {}).get("key_mode_best") or "").lower() if f else ""
        e["_jv"] = judge_val.get(k)
        rows.append(e)
    n = len(rows)

    def r_of(ka, kb):
        xs, ys = [], []
        for r in rows:
            if r.get(ka) is not None and r.get(kb) is not None:
                xs.append(r[ka]); ys.append(r[kb])
        return _pearson(xs, ys), len(xs)

    secs = []
    secs.append(
        "<div class='callout'><b>Read as a cross-check, not ground truth.</b> Music2Emo (MERT) was "
        "trained on real human recordings; our audio is FluidSynth-rendered MIDI (mostly piano-ish "
        "timbres), which is <b>out-of-distribution</b>. The value is convergence: do three independent "
        "methods — computed symbolic proxies, the LLM judge, and this audio model — agree?</div>")

    # ---- convergent validity ----
    rv_ca, n_ca = r_of("valence", "_cv")
    rv_ja, n_ja = r_of("valence", "_jv")
    rv_cj, n_cj = r_of("_cv", "_jv")
    ra_ca, _ = r_of("arousal", "_ca")
    rt, n_t = r_of("lib_tempo", "_ctempo")
    km = [(r["_cmode"], _mode_word(r.get("audio_key"))) for r in rows if r["_cmode"] and _mode_word(r.get("audio_key"))]
    key_agree = mean([1 if a == b else 0 for a, b in km]) if km else None

    cv = _table(
        [("pair (valence)", "Pearson correlation of per-piece valence between two independent methods. "
          "Higher = the methods agree on which pieces are brighter/darker."), ("Pearson r", ""), ("n", "")],
        [["computed ↔ audio", _fmt(rv_ca, 3), str(n_ca)],
         ["judge ↔ audio", _fmt(rv_ja, 3), str(n_ja)],
         ["computed ↔ judge", _fmt(rv_cj, 3), str(n_cj)]])
    secs.append("<h2>Convergent validity <span class='sub'>(do the three methods agree?)</span></h2>"
                "<p class='scope'>Valence measured three ways and correlated pairwise. All positive = the "
                "audio model tracks the same bright/dark signal as the symbolic proxy and the judge, despite "
                "the domain shift.</p>" + cv
                + f"<div class='callout'>Also: <b>audio-derived key agrees with the symbolic major/minor mode "
                  f"{_fmt(key_agree, pct=True)}</b> (n={len(km)}) — an independent audio→chord→key pipeline "
                  f"recovers the mode. Audio arousal ↔ computed arousal r={_fmt(ra_ca, 3)}; "
                  f"audio tempo ↔ symbolic tempo r={_fmt(rt, 3)} (n={n_t}).</div>")

    # ---- per model ----
    bymod = defaultdict(list)
    for r in rows:
        bymod[r["model"]].append(r)
    mrows = []
    for m, rs in sorted(bymod.items(), key=lambda kv: -mean(x["valence"] for x in kv[1])):
        moods = Counter(t for x in rs for t in x.get("moods", []))
        top = ", ".join(t for t, _ in moods.most_common(2))
        mrows.append([m, str(len(rs)), _fmt(mean(x["valence"] for x in rs)),
                      _fmt(mean(x["arousal"] for x in rs)), top])
    secs.append("<h2>Audio emotion by model</h2>"
                "<p class='scope'>What MERT hears per model (valence &amp; arousal on a 1–9 scale) and the "
                "two most frequent mood tags. Click a header to sort.</p>"
                + _table([("model", ""), ("n", ""),
                          ("valence", "MERT valence, 1 (dark) – 9 (bright)."),
                          ("arousal", "MERT arousal, 1 (calm) – 9 (energetic)."),
                          ("top moods", "Two most common predicted mood tags for this model.")], mrows))

    # ---- per mode acoustic ----
    bymode = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("valence", "arousal", "lib_tempo", "spec_centroid", "rms_mean", "onset_rate", "chord_distinct"):
            if r.get(k) is not None:
                bymode[r.get("mode")][k].append(r[k])
    order = [("abc", "ABC"), ("codegen", "code-gen"), ("smt-abc", "smt-abc")]
    arows = []
    for mk, lbl in order:
        d = bymode.get(mk)
        if not d or not d.get("valence"):
            continue
        arows.append([lbl, str(len(d["valence"])),
                      _fmt(mean(d["valence"])), _fmt(mean(d["arousal"])),
                      _fmt(mean(d["lib_tempo"]), 1), _fmt(mean(d["spec_centroid"]), 0),
                      _fmt(mean(d["rms_mean"]), 3), _fmt(mean(d["onset_rate"])),
                      _fmt(mean(d["chord_distinct"]), 1)])
    secs.append("<h2>Acoustic profile by generation mode</h2>"
                "<p class='scope'>Audio emotion plus the librosa acoustic suite, per representation.</p>"
                + _table([("mode", ""), ("n", ""), ("valence", "MERT valence 1–9."),
                          ("arousal", "MERT arousal 1–9."),
                          ("tempo", "Audio-estimated tempo (librosa beat-tracking, BPM)."),
                          ("brightness", "Spectral centroid (Hz) — higher = brighter timbre."),
                          ("loudness", "Mean RMS energy of the waveform."),
                          ("onset rate", "Note onsets per second detected in the audio."),
                          ("distinct chords", "Distinct chords from the audio chord-recognition model.")], arows))

    # ---- moods ----
    allm = Counter(t for r in rows for t in r.get("moods", []))
    mtab = [[t, _fmt(c / n, pct=True)] for t, c in allm.most_common(12)]
    secs.append("<h2>Mood tags <span class='sub'>(multi-label, ~10 per piece)</span></h2>"
                "<p class='scope'>Most-assigned mood tags across all pieces. Heavily skewed to "
                "\"sad/melancholic\" — partly real (the music does skew minor/slow) but also inflated by the "
                "dry, synthesized piano timbre, which reads as melancholic to a model trained on real "
                "recordings. Some tags (game, film, commercial) are usage-context, not emotion.</p>"
                + _table([("mood tag", ""), ("% of pieces", "Share of pieces assigned this tag.")], mtab))

    body = "\n".join(secs)
    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio emotion — musical inductive biases</title>
<link rel="stylesheet" href="style.css?v=22">
<style>
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
  .sub {{ color: {MUTED}; font-weight: 400; font-size: .8em; }}
  .scope {{ color: {MUTED}; font-size: .9rem; margin: .25rem 0 1.25rem; }}
  .tscroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .9rem; }}
  th, td {{ text-align: right; padding: .35rem .55rem; border-bottom: 1px solid #e7ddd2; }}
  th {{ color: {MUTED}; font-weight: 600; position: relative; }}
  .tip {{ border-bottom: 1px dotted {MUTED}; cursor: help; outline: none; }}
  #tipbox {{
    position: fixed; z-index: 100; width: 240px; white-space: normal; text-align: left;
    font-weight: 400; font-size: .76rem; line-height: 1.45; color: {BG}; background: {INK};
    padding: .55rem .65rem; border-radius: 7px; box-shadow: 0 6px 20px rgba(0,0,0,.2);
    pointer-events: none; display: none; }}
  td.m, th:first-child {{ text-align: left; font-weight: 600; }}
  h2 {{ margin-top: 2.4rem; }}
  .callout {{ background: #f3ede4; border-left: 3px solid {ACCENT}; padding: .7rem .9rem;
             border-radius: 0 7px 7px 0; font-size: .9rem; margin: .8rem 0 0; }}
  table.sortable th {{ cursor: pointer; user-select: none; white-space: nowrap; }}
  table.sortable th[data-dir=asc]::after {{ content: ' ▲'; font-size: .6em; opacity: .6; }}
  table.sortable th[data-dir=desc]::after {{ content: ' ▼'; font-size: .6em; opacity: .6; }}
</style>
</head><body>
<nav class="tabs">
  <a href="index.html">Browse outputs</a>
  <a href="results.html">Results &amp; analysis</a>
  <a href="judge.html">LLM judge</a>
  <a href="audio.html" class="active">Audio emotion</a>
</nav>
<div class="wrap">
  <h1>What the audio says</h1>
  <p class="scope">A third, independent emotion measurement: the rendered audio run through
     <b>Music2Emo</b> (MERT) for valence/arousal + mood, an audio-derived key/chords, and a librosa
     acoustic suite. Triangulated against the computed proxies and the LLM judge. Scope: {n} free-form
     pieces. Generated by <code>llm-music audio-report</code>.</p>
  {body}
</div>
<script>
  function makeSortable(table){{
    const head = table.tHead, body = table.tBodies[0];
    if(!head || !body) return;
    [...head.rows[0].cells].forEach((th, i) => {{
      th.addEventListener('click', () => {{
        const asc = th.dataset.dir !== 'asc';
        [...head.rows[0].cells].forEach(h => h.removeAttribute('data-dir'));
        th.dataset.dir = asc ? 'asc' : 'desc';
        const num = c => {{ const x = parseFloat(c.textContent.trim().replace(/[%,+\\s]/g, '')); return isNaN(x) ? null : x; }};
        [...body.rows].sort((a, b) => {{
          const ka = num(a.cells[i]), kb = num(b.cells[i]);
          if(ka === null && kb === null) {{ const c = a.cells[i].textContent.trim().localeCompare(b.cells[i].textContent.trim()); return asc ? c : -c; }}
          if(ka === null) return 1;
          if(kb === null) return -1;
          return asc ? ka - kb : kb - ka;
        }}).forEach(r => body.appendChild(r));
      }});
    }});
  }}
  document.querySelectorAll('table.sortable').forEach(makeSortable);
  (function(){{
    var box=document.createElement('div'); box.id='tipbox'; document.body.appendChild(box);
    function show(el){{
      var t=el.getAttribute('data-tip'); if(!t) return;
      box.textContent=t; box.style.display='block';
      var r=el.getBoundingClientRect();
      box.style.left=Math.max(6, Math.min(r.left, window.innerWidth-252))+'px';
      var top=r.top-box.offsetHeight-6; if(top<6) top=r.bottom+6;
      box.style.top=top+'px';
    }}
    function hide(){{ box.style.display='none'; }}
    document.addEventListener('mouseover', function(e){{ var el=e.target.closest&&e.target.closest('.tip'); if(el) show(el); }});
    document.addEventListener('mouseout', function(e){{ if(e.target.closest&&e.target.closest('.tip')) hide(); }});
    document.addEventListener('focusin', function(e){{ var el=e.target.closest&&e.target.closest('.tip'); if(el) show(el); }});
    document.addEventListener('focusout', hide);
  }})();
</script>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
