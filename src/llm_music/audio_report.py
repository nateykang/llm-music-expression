"""Generate docs/audio.html — the audio tab (fourth site tab).

Raw-table format, matching Results & analysis and the LLM-judge tab: per-model
tables you can sort and read directly, no pre-baked correlations. Exposes the full
audio-side measurement suite side by side so comparisons are the reader's to make:

  • Music2Emo (MERT)      — valence/arousal + moods (+ 56-mood probs)
  • audio-derived harmony — key, chord count / distinct / changes
  • librosa acoustic suite — spectral shape, dynamics, timbre (MFCC), pitch-class (chroma)
  • gemini-2.5-pro READ   — judging the blinded NOTATION
  • gemini-2.5-pro HEAR   — the SAME model judging the rendered AUDIO
  • gpt-audio HEAR        — a second listener, audio only
  • computed proxy        — the symbolic valence/arousal from features.csv

Everything audio-side is measured on FluidSynth-synthesized MIDI, out-of-distribution
for models trained on real recordings — read as a cross-check, not ground truth.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .judge_report import SHORT, _table
from .report import ACCENT, BG, INK, MUTED

QUAL = ["coherence", "harmony", "rhythm", "structure", "melody", "emotion", "creativity", "naturalness"]
TOGGLE = [("abc", "ABC"), ("code", "code-gen"), ("all", "both")]
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# librosa acoustic suite: (field, header, tooltip, decimals)
SPECTRAL = [
    ("lib_tempo", "tempo", "librosa beat-tracking tempo (BPM)", 1),
    ("spec_centroid", "centroid", "spectral centroid (Hz) — brightness", 0),
    ("spec_bandwidth", "bandwidth", "spectral bandwidth (Hz)", 0),
    ("spec_rolloff", "rolloff", "spectral rolloff freq (Hz, 85% of energy below)", 0),
    ("spec_flatness", "flatness", "spectral flatness (0 tonal → 1 noise-like)", 4),
    ("spec_contrast", "contrast", "spectral contrast (dB, peak-to-valley)", 1),
    ("zcr", "ZCR", "zero-crossing rate", 3),
    ("rms_mean", "RMS", "mean RMS energy (loudness)", 3),
    ("rms_std", "RMS σ", "RMS std — dynamic variation over the piece", 3),
    ("harmonic_ratio", "harmonic", "harmonic-to-total energy ratio", 2),
    ("onset_rate", "onsets/s", "note onsets per second detected in the audio", 2),
]
MFCC = [(f"mfcc{i}", f"mfcc{i}", f"MFCC coefficient {i} (timbre / spectral envelope)", 1) for i in range(1, 14)]
CHROMA = [(f"chroma{i + 1}", NOTES[i], f"chroma energy for pitch class {NOTES[i]}", 2) for i in range(12)]


def _fmt(v, nd=2, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v * 100:.0f}%" if pct else f"{v:.{nd}f}"


def _m(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and v != v)]
    return mean(vals) if vals else None


def _num(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _mode_filter(items, mode):
    if mode == "all":
        return items
    if mode == "abc":
        return [x for x in items if x.get("mode") in ("abc", "smt-abc")]
    return [x for x in items if x.get("mode") == "codegen"]


def _paned(fn):
    return "".join(
        f"<div class='mode-pane' data-mode='{m}'{'' if m == 'all' else ' hidden'}>{fn(m)}</div>"
        for m, _ in TOGGLE)


def _by_model(recs):
    g = defaultdict(list)
    for r in recs:
        g[r["model"]].append(r)
    return g


def _src(rec, src, dim):
    """Pull a judge dimension from a piece record's sub-verdict (read/hear/gpt)."""
    d = rec.get(src)
    return d.get(dim) if d else None


# ---------- generic per-model feature table ----------
def _feature_table(recs, spec, extra_cols=None, extra_fn=None, sort_col=None):
    """spec = [(field, header, tip, nd)]; one row per model of the mean of each field."""
    g = _by_model(recs)
    rows = []
    for m, rs in g.items():
        cells = [SHORT.get(m, m), str(len(rs))]
        if extra_fn:
            cells += extra_fn(rs)
        for field, _, _, nd in spec:
            cells.append(_fmt(_m([r.get(field) for r in rs]), nd))
        rows.append(cells)
    rows.sort(key=lambda c: c[0])
    cols = [("model", None), ("n", "pieces for this model in the current mode")]
    cols += (extra_cols or [])
    cols += [(h, t) for _, h, t, _ in spec]
    return _table(cols, rows)


# ---------- affect (valence / arousal), all methods ----------
def _affect_table(recs, dim):
    g = _by_model(recs)
    comp = "comp_" + dim[0]
    rows = []
    for m, rs in g.items():
        rows.append([
            SHORT.get(m, m), str(len(rs)),
            _fmt(_m([r.get(comp) for r in rs]), 2),        # computed proxy
            _fmt(_m([r.get(dim) for r in rs]), 2),          # Music2Emo 1-9
            _fmt(_m([_src(r, "read", dim) for r in rs]), 2),
            _fmt(_m([_src(r, "hear", dim) for r in rs]), 2),
            _fmt(_m([_src(r, "gpt", dim) for r in rs]), 2),
        ])
    rows.sort(key=lambda c: (c[5] == "—", -(float(c[5]) if c[5] != "—" else 0)))  # by gemini-hear
    cols = [("model", None), ("n", "pieces for this model in the current mode"),
            ("computed", f"computed symbolic {dim} proxy (features.csv); its own scale"),
            ("Music2Emo", f"MERT audio {dim}, 1–9"),
            ("gemini read", f"gemini judging the blinded NOTATION — {dim}, 1–5"),
            ("gemini hear", f"gemini judging the rendered AUDIO — {dim}, 1–5"),
            ("gpt-audio", f"gpt-audio judging the AUDIO — {dim}, 1–5 (n≤600)")]
    return _table(cols, rows)


def _qual_table(recs, src):
    g = _by_model(recs)
    rows = []
    for m, rs in g.items():
        dims = [_m([_src(r, src, k) for r in rs]) for k in QUAL]
        n = sum(1 for r in rs if r.get(src) is not None)
        rows.append([SHORT.get(m, m), str(n), f"<b>{_fmt(_m(dims))}</b>"] + [_fmt(v) for v in dims])
    rows.sort(key=lambda c: (c[2] == "<b>—</b>", -(float(c[2][3:-4]) if c[2] != "<b>—</b>" else 0)))
    cols = [("model", None), ("n", "pieces judged"), ("overall", "mean of the 8 quality dimensions")]
    cols += [(k, f"{k}, 1–5") for k in QUAL]
    return _table(cols, rows)


def _emotion_label_table(recs):
    g = _by_model(recs)
    rows = []
    for m, rs in sorted(g.items()):
        def top(src):
            c = Counter(r[src]["emotion_label"] for r in rs
                        if r.get(src) and r[src].get("emotion_label"))
            return ", ".join(f"{l} ({n})" for l, n in c.most_common(2)) or "—"
        rows.append([SHORT.get(m, m), top("read"), top("hear"), top("gpt")])
    cols = [("model", None),
            ("gemini read", "most-named emotion when gemini reads the notation"),
            ("gemini hear", "most-named emotion when gemini hears the audio"),
            ("gpt-audio", "most-named emotion when gpt-audio hears the audio")]
    return _table(cols, rows)


def _moods_by_model(recs):
    g = _by_model(recs)
    rows = []
    for m, rs in sorted(g.items()):
        c = Counter(t for r in rs for t in r.get("moods", []))
        rows.append([SHORT.get(m, m), str(len(rs)),
                     ", ".join(f"{t} ({n})" for t, n in c.most_common(4)) or "—"])
    return _table([("model", None), ("n", "pieces"),
                   ("top Music2Emo mood tags", "most-assigned MERT mood tags (count)")], rows)


def _harmony_table(recs):
    def extra(rs):
        keyed = [r.get("audio_key", "") for r in rs if r.get("audio_key")]
        minor = sum(1 for k in keyed if k.endswith("minor"))
        return [_fmt(minor / len(keyed) if keyed else None, pct=True)]
    return _feature_table(
        recs,
        [("chord_n", "chords", "mean chord count (chord_n)", 1),
         ("chord_distinct", "distinct", "mean distinct chords", 1),
         ("chord_changes", "changes", "mean chord changes", 1)],
        extra_cols=[("minor %", "share of pieces the audio→chord→key pipeline reads as minor")],
        extra_fn=extra)


def _acoustic_by_mode(recs):
    keys = [f for f, _, _, _ in SPECTRAL]
    g = defaultdict(lambda: defaultdict(list))
    for r in recs:
        for k in ["valence", "arousal"] + keys:
            if r.get(k) is not None:
                g[r["mode"]][k].append(r[k])
    rows = []
    for mk, lbl in [("abc", "ABC"), ("codegen", "code-gen"), ("smt-abc", "smt-abc")]:
        d = g.get(mk)
        if not d or not d.get("valence"):
            continue
        cells = [lbl, str(len(d["valence"])), _fmt(_m(d["valence"])), _fmt(_m(d["arousal"]))]
        cells += [_fmt(_m(d.get(f, [])), nd) for f, _, _, nd in SPECTRAL]
        rows.append(cells)
    cols = [("mode", None), ("n", "pieces"),
            ("valence", "MERT valence 1–9"), ("arousal", "MERT arousal 1–9")]
    cols += [(h, t) for _, h, t, _ in SPECTRAL]
    return _table(cols, rows)


# ---------- page ----------
def render_audio_html(analysis: Path, data_dir: Path, out_path: Path) -> Path:
    m2e_path = analysis / "music2emo_full.json"
    if not m2e_path.exists():
        out_path.write_text("<p>No audio results. Run the Music2Emo batch first.</p>", encoding="utf-8")
        return out_path
    m2e = [e for e in json.loads(m2e_path.read_text(encoding="utf-8")) if "valence" in e]

    feats = {}
    for csvf in sorted(data_dir.glob("*/features.csv")):
        for r in csv.DictReader(csvf.open(encoding="utf-8")):
            if r.get("prompt") == "free-form":
                feats[(r["model"], r.get("mode"), r.get("title"))] = r

    ja = analysis / "judge_audio_llm.json"
    read, hear, gpt = {}, {}, {}
    if ja.exists():
        for r in json.loads(ja.read_text(encoding="utf-8")):
            k = (r["model"], r.get("mode"), r.get("title"), str(r.get("sample")))
            if r["judge"] == "gemini-2.5-pro":
                (read if r["modality"] == "notation" else hear)[k] = r
            elif r["judge"] == "gpt-audio":
                gpt[k] = r

    # unified per-piece records: all Music2Emo fields under their own names, plus
    # the computed proxy and the three judge sub-verdicts.
    recs = []
    for e in m2e:
        k4 = (e["model"], e.get("mode"), e.get("title"), str(e.get("sample")))
        f = feats.get((e["model"], e.get("mode"), e.get("title")))
        rec = dict(e)
        rec["comp_v"] = _num(f.get("valence")) if f else None
        rec["comp_a"] = _num(f.get("arousal")) if f else None
        rec["read"], rec["hear"], rec["gpt"] = read.get(k4), hear.get(k4), gpt.get(k4)
        recs.append(rec)
    n = len(recs)
    n_read = sum(1 for r in recs if r["read"])
    n_hear = sum(1 for r in recs if r["hear"])
    n_gpt = sum(1 for r in recs if r["gpt"])

    secs = []
    secs.append(
        "<div class='callout'><b>Read as a cross-check, not ground truth.</b> Everything on the audio "
        "side is measured on FluidSynth-rendered MIDI (mostly piano-ish timbres), which is "
        "<b>out-of-distribution</b> for Music2Emo (MERT, trained on real recordings) and flattens the "
        "audio-LLM listeners' dynamics. Tables are raw per-model means — sort any column; comparisons "
        "are yours to make.</div>")

    secs.append("<h2>Valence by model <span class='sub'>(five methods, side by side)</span></h2>"
                "<p class='scope'>Per-model mean valence from every method. Scales differ "
                "(computed = its own proxy scale; Music2Emo 1–9; the LLM judges 1–5) — compare "
                "<i>orderings</i> across columns, not absolute values.</p>"
                + _paned(lambda mo: _affect_table(_mode_filter(recs, mo), "valence")))
    secs.append("<h2>Arousal by model</h2><p class='scope'>Same five methods, arousal.</p>"
                + _paned(lambda mo: _affect_table(_mode_filter(recs, mo), "arousal")))
    secs.append("<h2>Dominant emotion label <span class='sub'>(read vs hear vs gpt-audio)</span></h2>"
                "<p class='scope'>The single emotion each listener named most, per model — how the "
                "named character shifts between reading and hearing.</p>"
                + _paned(lambda mo: _emotion_label_table(_mode_filter(recs, mo))))

    secs.append("<h2>Quality — gemini READ <span class='sub'>(notation)</span></h2>"
                "<p class='scope'>gemini-2.5-pro's 8 quality dimensions judging the blinded "
                "notation.</p>" + _paned(lambda mo: _qual_table(_mode_filter(recs, mo), "read")))
    secs.append("<h2>Quality — gemini HEAR <span class='sub'>(audio)</span></h2>"
                "<p class='scope'>The same model, same rubric, judging the rendered audio. Identical "
                "columns to the READ table above for direct comparison.</p>"
                + _paned(lambda mo: _qual_table(_mode_filter(recs, mo), "hear")))
    secs.append("<h2>Quality — gpt-audio HEAR <span class='sub'>(audio, 2nd listener)</span></h2>"
                "<p class='scope'>A second audio listener, audio only.</p>"
                + _paned(lambda mo: _qual_table(_mode_filter(recs, mo), "gpt")))

    secs.append("<h2>Music2Emo mood tags by model</h2>"
                "<p class='scope'>The most-assigned MERT mood tags per model (multi-label, ~10/piece; "
                "56-mood vocabulary).</p>"
                + _paned(lambda mo: _moods_by_model(_mode_filter(recs, mo))))
    secs.append("<h2>Audio-derived harmony by model</h2>"
                "<p class='scope'>From the audio chord-recognition → key pipeline (independent of the "
                "symbolic notation): key mode and chord statistics.</p>"
                + _paned(lambda mo: _harmony_table(_mode_filter(recs, mo))))
    secs.append("<h2>Spectral shape &amp; dynamics by model <span class='sub'>(librosa)</span></h2>"
                "<p class='scope'>Timbre-shape, loudness, and rhythm-density descriptors of the "
                "rendered audio.</p>"
                + _paned(lambda mo: _feature_table(_mode_filter(recs, mo), SPECTRAL)))
    secs.append("<h2>Timbre — MFCC by model <span class='sub'>(1–13)</span></h2>"
                "<p class='scope'>Mel-frequency cepstral coefficients — the standard timbre "
                "fingerprint. Low-level; here for completeness.</p>"
                + _paned(lambda mo: _feature_table(_mode_filter(recs, mo), MFCC)))
    secs.append("<h2>Pitch-class energy — chroma by model</h2>"
                "<p class='scope'>Mean energy per pitch class (C…B) in the audio — which notes "
                "dominate, independent of octave.</p>"
                + _paned(lambda mo: _feature_table(_mode_filter(recs, mo), CHROMA)))
    secs.append("<h2>Acoustic profile by generation mode</h2>"
                "<p class='scope'>The same audio suite cut by representation (all modes in one view).</p>"
                + _acoustic_by_mode(recs)
                + "<div class='callout' style='font-size:.82rem'>Music2Emo also produces a "
                  "<b>1536-dim MERT embedding</b> per piece (in <code>music2emo_embeddings.npz</code>) — "
                  "not tabular, so not shown here; available for similarity / clustering work.</div>")

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
  .mode-toggle {{ position: sticky; top: 0; z-index: 10; background: {BG}; display: flex;
    gap: 8px; align-items: center; padding: .6rem 0; margin-bottom: .5rem;
    border-bottom: 1px solid #e7ddd2; }}
  .mode-toggle .lbl {{ font-weight: 600; color: {INK}; font-size: .9rem; }}
  .mode-toggle button {{ font: inherit; font-size: .85rem; padding: 4px 13px; border-radius: 7px;
    border: 1px solid #cbb99a; background: #fff; color: {INK}; cursor: pointer; }}
  .mode-toggle button[aria-pressed=true] {{ background: {ACCENT}; color: {BG}; border-color: {ACCENT}; }}
  .mode-pane[hidden] {{ display: none; }}
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
  <p class="scope">Raw per-model tables of the full audio-side measurement suite: <b>Music2Emo</b> (MERT
     valence/arousal + moods), audio-derived harmony, and the librosa acoustic suite (spectral shape,
     dynamics, MFCC timbre, chroma), plus two audio-LLM listeners — <b>gemini-2.5-pro</b> (which also
     judged the notation, so you get read vs hear) and <b>gpt-audio</b> — and the computed symbolic
     proxy. Scope: {n} free-form pieces (gemini read {n_read} · hear {n_hear} · gpt-audio {n_gpt}). Sort
     any column; the "Generation" toggle slices by representation. Generated by
     <code>llm-music audio-report</code>.</p>
  <div class="mode-toggle">
    <span class="lbl">Generation</span>
    <button data-mode="abc">ABC</button>
    <button data-mode="code">code-gen</button>
    <button data-mode="all" aria-pressed="true">both</button>
  </div>
  {body}
</div>
<script>
  function setMode(m){{
    document.querySelectorAll('.mode-pane').forEach(p => {{ p.hidden = p.dataset.mode !== m; }});
    document.querySelectorAll('.mode-toggle button').forEach(b => b.setAttribute('aria-pressed', b.dataset.mode === m));
  }}
  document.querySelectorAll('.mode-toggle button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));

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
