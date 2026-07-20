"""Shared page framework for the generated dashboards (results / judge / audio).

Everything the three report generators have in common lives here: the site
palette, the HTML document skeleton (head, nav tabs, wrap, shared CSS/JS), the
sortable-table + tooltip + mode-toggle machinery, and the small formatting
helpers. A report module supplies only its sections and calls ``page()``.

The generated pages link docs/style.css and use its CSS variables (--bg, --fg,
--muted, --accent, --border) so all four site tabs share one palette. The hex
constants below mirror those variables for the one place CSS can't reach:
matplotlib charts.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


def degenerate_keys(analysis: Path) -> set:
    """(model, mode, title, str(sample)) keys of pieces retroactively flagged as
    degenerate (empty part / silent tail — see scripts/flag_degenerate_codegen.py).
    Report builders drop these from judge/embedding/audio analyses; new batches
    can't produce them (the sandbox validates exports at generation time)."""
    p = analysis / "degenerate_pieces.json"
    if not p.exists():
        return set()
    return {(f["model"], f["mode"], f["title"], str(f["sample"]))
            for f in json.loads(p.read_text(encoding="utf-8"))}


def drop_degenerate(rows, analysis: Path):
    """Filter piece dicts (judge raw / music2emo entries) against the blacklist."""
    bad = degenerate_keys(analysis)
    if not bad:
        return rows
    return [r for r in rows
            if (r.get("model"), r.get("mode"), r.get("title"),
                str(r.get("sample"))) not in bad
            and (r.get("model"), r.get("mode"), r.get("title"),
                 str(r.get("sample") or 0)) not in bad]

# Site palette — keep in sync with the CSS variables in docs/style.css.
BG = "#faf9f7"
INK = "#1d1b18"
MUTED = "#6b665d"
ACCENT = "#7a5c3e"
CARD = "#ffffff"
BORDER = "#e4e0d8"

# Display shortening for model names in dense tables.
SHORT = {"fable-5": "fable", "gpt-5.5": "gpt-5.5", "gemini-2.5-pro": "gemini", "opus-4.8": "opus",
         "opus-4.8-thinking": "opus-think", "sonnet-4.6": "sonnet",
         "sonnet-4.6-thinking": "sonnet-think", "deepseek-v4-pro": "deepseek",
         "gpt-4.1": "gpt-4.1", "grok-4.3": "grok", "qwen3-max": "qwen",
         "llama-4-maverick": "llama"}

# Generation-mode toggle (judge/audio pages): ABC groups abc + smt-abc.
MODE_TOGGLE = [("abc", "ABC"), ("code", "code-gen"), ("all", "both")]
# Representation toggle (results page): ABC (declared K:) vs code-gen (detected).
REP_TOGGLE = [("text", "ABC"), ("code", "code-gen"), ("all", "both")]


def fnum(x) -> float | None:
    """Parse to float, or None (NaN -> None so it drops from averages)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def mode_filter(items, mode):
    """Filter raw pieces / feature rows by generation mode. ABC groups abc + smt-abc
    (both notation-based); code = code-gen; all = everything."""
    if mode == "all":
        return items
    if mode == "abc":
        return [x for x in items if x.get("mode") in ("abc", "smt-abc")]
    return [x for x in items if x.get("mode") == "codegen"]


def paned(fn, options=MODE_TOGGLE, default="all"):
    """Render content once per toggle option into panes; only `default` starts
    visible, the shared toggle JS switches the others in."""
    return "".join(
        f"<div class='mode-pane' data-mode='{key}'{'' if key == default else ' hidden'}>{fn(key)}</div>"
        for key, _ in options)


def toggle(options=MODE_TOGGLE, default="all", label="Generation", note=""):
    """The sticky pane-switcher matching ``paned()``. One per page."""
    btns = "".join(
        f"<button data-mode='{key}'{' aria-pressed=\"true\"' if key == default else ''}>{lbl}</button>"
        for key, lbl in options)
    note_html = f"<span class='note'>{note}</span>" if note else ""
    return (f"<div class='mode-toggle' data-default='{default}'>"
            f"<span class='lbl'>{label}</span>{btns}{note_html}</div>")


def group_by_model(rows) -> dict[str, list]:
    g: dict[str, list] = {}
    for r in rows:
        g.setdefault(r["model"], []).append(r)
    return g


# ---------- glossary & footnotes ----------

# Shared reference list. Prose cites an entry with ``fnote(key)``; page()
# numbers the markers by first appearance and appends a "Footnotes &
# references" section listing only the entries the page actually cites.
REFS = {
    "russell": ('James A. Russell, “A circumplex model of affect”, <i>Journal of Personality '
                'and Social Psychology</i> 39(6), 1980. '
                '<a href="https://doi.org/10.1037/h0077714">doi:10.1037/h0077714</a>. The model '
                'places every emotion on two axes: valence (unpleasant ↔ pleasant) and arousal '
                '(calm ↔ activated).'),
    "krumhansl": ('Carol L. Krumhansl, <i>Cognitive Foundations of Musical Pitch</i>, Oxford '
                  'University Press, 1990. The Krumhansl–Schmuckler key-finding algorithm '
                  'matches a piece’s pitch-class distribution against experimentally measured '
                  'major- and minor-key profiles and picks the best-fitting key.'),
    "muspy": ('Hao-Wen Dong, Ke Chen, Julian McAuley, Taylor Berg-Kirkpatrick, “MusPy: A '
              'toolkit for symbolic music generation”, ISMIR 2020. '
              '<a href="https://arxiv.org/abs/2008.01951">arXiv:2008.01951</a>.'),
    "music21": ('Michael Scott Cuthbert, Christopher Ariza, “music21: A toolkit for '
                'computer-aided musicology and symbolic music data”, ISMIR 2010. '
                '<a href="https://www.music21.org/">music21.org</a>.'),
    "chatmusician": ('Ruibin Yuan et al., “ChatMusician: Understanding and Generating Music '
                     'Intrinsically with LLM”, 2024. '
                     '<a href="https://arxiv.org/abs/2402.16153">arXiv:2402.16153</a>.'),
    "mupt": ('Xingwei Qu et al., “MuPT: A Generative Symbolic Music Pretrained Transformer”, '
             '2024. <a href="https://arxiv.org/abs/2404.06393">arXiv:2404.06393</a>.'),
    "chu": ('Hyeshin Chu et al., “An Empirical Study on How People Perceive AI-generated '
            'Music”, CIKM 2022. '
            '<a href="https://doi.org/10.1145/3511808.3557235">doi:10.1145/3511808.3557235</a>. '
            'Source of the melodiousness / naturalness / creativity / coherence axes in the '
            'judging rubric.'),
    "muspike": ('Qian Liang, Menghaoran Tang, Yi Zeng, “MuSpike: A Benchmark and Evaluation '
                'Framework for Symbolic Music Generation with Spiking Neural Networks”, 2025. '
                '<a href="https://arxiv.org/abs/2508.19251">arXiv:2508.19251</a>. Source of '
                'several harmony/structure metrics and rubric dimensions used here.'),
    "llm-judge": ('Lianmin Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot '
                  'Arena”, NeurIPS 2023. '
                  '<a href="https://arxiv.org/abs/2306.05685">arXiv:2306.05685</a>. '
                  '“LLM-as-judge” = using a language model, given a rubric, as the rater '
                  'instead of human annotators.'),
    "geval": ('Yang Liu et al., “G-Eval: NLG Evaluation using GPT-4 with Better Human '
              'Alignment”, EMNLP 2023. '
              '<a href="https://arxiv.org/abs/2303.16634">arXiv:2303.16634</a>. Source of the '
              'reason-before-score protocol.'),
    "emopia": ('Hsiao-Tzu Hung et al., “EMOPIA: A Multi-Modal Pop Piano Dataset for Emotion '
               'Recognition and Emotion-based Music Generation”, ISMIR 2021. '
               '<a href="https://arxiv.org/abs/2108.01374">arXiv:2108.01374</a>. Uses the same '
               'valence/arousal quadrant scheme for music.'),
    "mert": ('Yizhi Li et al., “MERT: Acoustic Music Understanding Model with Large-Scale '
             'Self-supervised Training”, ICLR 2024. '
             '<a href="https://arxiv.org/abs/2306.00107">arXiv:2306.00107</a>. MERT is a '
             'transformer trained on large amounts of music audio; its internal activations '
             'serve as a general-purpose numeric summary (embedding) of what a recording '
             'sounds like.'),
    "music2emo": ('Jaeyong Kang, Dorien Herremans, “Towards Unified Music Emotion Recognition '
                  'across Dimensional and Categorical Models”, 2025. '
                  '<a href="https://arxiv.org/abs/2502.03979">arXiv:2502.03979</a>. Music2Emo '
                  'is a MERT-based model that predicts valence/arousal and mood tags from '
                  'audio.'),
    "librosa": ('Brian McFee et al., “librosa: Audio and music signal analysis in Python”, '
                'SciPy 2015. <a href="https://librosa.org/">librosa.org</a>.'),
    "pca": ('Principal component analysis (PCA) re-describes a high-dimensional point cloud '
            'along the directions where it varies most; the first few “principal components” '
            'often capture most of the structure. Ian T. Jolliffe, <i>Principal Component '
            'Analysis</i>, 2nd ed., Springer, 2002.'),
    "tsne": ('Laurens van der Maaten, Geoffrey Hinton, “Visualizing Data using t-SNE”, '
             '<i>Journal of Machine Learning Research</i> 9, 2008. t-SNE flattens a '
             'high-dimensional point cloud into a 2-D map that tries to keep similar points '
             'close together; distances between far-apart groups are not meaningful.'),
    "gigamidi": ('Keon Ju Maverick Lee et al., “The GigaMIDI Dataset with Features for '
                 'Expressive Music Performance Detection”, <i>TISMIR</i> 8(1), 2025. '
                 '<a href="https://arxiv.org/abs/2502.17726">arXiv:2502.17726</a>. A '
                 '1.4M-file MIDI corpus, used here as the “real music” prior.'),
    "circle-of-fifths": ('The circle of fifths arranges the 12 keys so that neighbors differ '
                         'by one sharp/flat (C → G → D …); musically related keys sit next to '
                         'each other. <a href="https://en.wikipedia.org/wiki/Circle_of_fifths">'
                         'Wikipedia: Circle of fifths</a>.'),
    "abc": ('ABC notation is a compact plain-text format for writing music (letters for '
            'notes, <code>K:</code> for the key, etc.), popular for folk tunes and easy for '
            'text models to emit. <a href="https://abcnotation.com/">abcnotation.com</a>.'),
    "fluidsynth": ('<a href="https://www.fluidsynth.org/">FluidSynth</a> — an open-source '
                   'software synthesizer that renders MIDI files to audio using sampled '
                   'instrument sounds (SoundFonts).'),
    "sara-fish": ('Project after <a href="https://github.com/sara-fish/llm-musical-self-expression">'
                  'sara-fish/llm-musical-self-expression</a>.'),
}


def dfn(term, gloss):
    """An inline defined term: dotted underline + the shared hover/focus tooltip
    (same machinery as the column-header tips)."""
    return (f"<span class='tip' tabindex='0' data-tip=\"{html.escape(gloss)}\">"
            f"{html.escape(term)}</span>")


def fnote(key):
    """A superscript footnote marker citing REFS[key]. page() assigns numbers by
    first appearance and appends the cited entries to the page."""
    if key not in REFS:
        raise KeyError(f"unknown footnote key: {key}")
    return f"<sup class='fn'><a data-fn='{key}' href='#fn-{key}'>?</a></sup>"


def _apply_footnotes(body):
    """Number every fnote() marker in `body` by first appearance and append the
    footnote list. Bodies without markers pass through untouched."""
    keys = []
    for m in re.finditer(r"data-fn='([a-z0-9-]+)'", body):
        if m.group(1) not in keys:
            keys.append(m.group(1))
    if not keys:
        return body
    for i, k in enumerate(keys, 1):
        body = body.replace(f"data-fn='{k}' href='#fn-{k}'>?<",
                            f"data-fn='{k}' href='#fn-{k}'>{i}<")
    items = "".join(f"<li id='fn-{k}'>{REFS[k]}</li>" for k in keys)
    return (body + "\n  <h2 id='footnotes'>Footnotes &amp; references</h2>"
            f"\n  <ol class='footnotes'>{items}</ol>")


# ---------- table building ----------

def tip(label, tip_text):
    """A <th>, with a hover/focus tooltip when tip_text is given."""
    if not tip_text:
        return f"<th>{html.escape(label)}</th>"
    return (f"<th><span class='tip' tabindex='0' data-tip=\"{html.escape(tip_text)}\">"
            f"{html.escape(label).replace(' ', '&nbsp;')}</span></th>")


def cell(v, kind):
    """Format a value for a table cell: text | int | pct | f0 | f1 | f2.
    A (mean, sd) tuple renders as 'mean ±sd' with the sd de-emphasized; sorting
    still works because the mean leads the cell text."""
    if v is None:
        return "—"
    if isinstance(v, tuple):
        m, sd = v
        if m is None:
            return "—"
        main = cell(m, kind)
        if sd is None:
            return main
        spread = f"{sd * 100:.0f}" if kind == "pct" else cell(sd, kind)
        return f"{main} <span class='sd'>±{spread}</span>"
    if kind == "text":
        return html.escape(str(v))
    if kind == "int":
        return str(int(v))
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "f0":
        return f"{v:.0f}"
    if kind == "f1":
        return f"{v:.1f}"
    return f"{v:.2f}"


def table(cols, rows, left=1):
    """Sortable table. cols = [(label, tip_or_None)]; each row is a list of
    cell-HTML strings, or a (cells, row_class) tuple (rows classed 'ref' stay
    pinned on top when sorting). The first `left` cells are left-aligned."""
    head = "<thead><tr>" + "".join(tip(l, t) for l, t in cols) + "</tr></thead>"
    body = "<tbody>"
    for row in rows:
        cells, cls = row if isinstance(row, tuple) else (row, "")
        body += (f"<tr class='{cls}'>" if cls else "<tr>") + "".join(
            f"<td class='{'m' if i < left else ''}'>{c}</td>" for i, c in enumerate(cells)) + "</tr>"
    body += "</tbody>"
    return f"<div class='tscroll'><table class='sortable'>{head}{body}</table></div>"


def heat(v, scale=0.55):
    """A heatmap <td>: green for positive, red for negative, opacity ∝ |v|."""
    if v is None:
        return "<td>—</td>"
    a = min(0.5, abs(v) / scale * 0.5)
    rgb = "46,160,67" if v >= 0 else "207,90,80"
    return f"<td style='background:rgba({rgb},{a:.2f})'>{v:+.2f}</td>"


def scorebar(v, lo=1.0, hi=5.0, fmt="{:.2f}"):
    """A cell that shows a value with a proportional bar behind it — makes a
    ranking column readable at a glance without a separate chart."""
    if v is None:
        return "—"
    pct = max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100
    return (f"<span class='scorebar'><span class='scorebar-fill' style='width:{pct:.0f}%'></span>"
            f"<b>{fmt.format(v)}</b></span>")


def details_section(summary_html, body_html):
    """Progressive disclosure for completeness-not-headline content (e.g. the
    MFCC table): collapsed by default, one click to expand."""
    return (f"<details class='lowlevel'><summary>{summary_html}</summary>"
            f"{body_html}</details>")


# ---------- page skeleton ----------

TABS = [("index.html", "Browse outputs"), ("results.html", "Results &amp; analysis"),
        ("judge.html", "LLM judge"), ("audio.html", "Audio emotion"),
        ("selfpref.html", "Self-preference"), ("genre.html", "Genre bias"),
        ("studio.html", "Studio")]

SHARED_CSS = """
  .wrap { max-width: 980px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
  .sub { color: var(--muted); font-weight: 400; font-size: .8em; }
  .scope { color: var(--muted); font-size: .9rem; margin: .25rem 0 1.25rem; }
  .tscroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .9rem; }
  th, td { text-align: right; padding: .35rem .55rem; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; position: relative; }
  .tip { border-bottom: 1px dotted var(--muted); cursor: help; outline: none; }
  #tipbox { position: fixed; z-index: 100; width: 240px; white-space: normal; text-align: left;
    font-weight: 400; font-size: .76rem; line-height: 1.45; color: var(--bg); background: var(--fg);
    padding: .55rem .65rem; border-radius: 7px; box-shadow: 0 6px 20px rgba(0,0,0,.2);
    pointer-events: none; display: none; }
  td.m, th:first-child { text-align: left; font-weight: 600; }
  td.sub { color: var(--muted); font-size: .8rem; }
  tr.ref td { font-style: italic; color: var(--accent); background: #f3ede4; border-bottom: 2px solid #d8c9b5; }
  table.heat td { text-align: center; }
  h2 { margin-top: 2.4rem; }
  figure { margin: 1.5rem 0; }
  figcaption { color: var(--muted); font-size: .85rem; margin-top: .4rem; }
  .callout { background: #f3ede4; border-left: 3px solid var(--accent); padding: .7rem .9rem;
    border-radius: 0 7px 7px 0; font-size: .9rem; margin: .8rem 0 0; }
  .mode-toggle { position: sticky; top: 0; z-index: 10; background: var(--bg); display: flex;
    gap: 8px; align-items: center; padding: .6rem 0; margin-bottom: .5rem;
    border-bottom: 1px solid var(--border); }
  .mode-toggle .lbl { font-weight: 600; color: var(--fg); font-size: .9rem; }
  .mode-toggle .note { font-size: .8rem; color: var(--muted); }
  .mode-toggle button { font: inherit; font-size: .85rem; padding: 4px 13px; border-radius: 7px;
    border: 1px solid #cbb99a; background: #fff; color: var(--fg); cursor: pointer; }
  .mode-toggle button[aria-pressed=true] { background: var(--accent); color: var(--bg); border-color: var(--accent); }
  .mode-pane[hidden] { display: none; }
  table.sortable th { cursor: pointer; user-select: none; white-space: nowrap; }
  table.sortable th[data-dir=asc]::after { content: ' ▲'; font-size: .6em; opacity: .6; }
  table.sortable th[data-dir=desc]::after { content: ' ▼'; font-size: .6em; opacity: .6; }
  .scorebar { position: relative; display: inline-block; min-width: 84px; padding: 1px 6px; }
  .scorebar-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px;
    background: color-mix(in srgb, var(--accent) 22%, transparent); }
  .scorebar b { position: relative; }
  details.lowlevel { margin: 1.2rem 0; border: 1px solid var(--border); border-radius: 8px;
    padding: .5rem .9rem; background: var(--card); }
  details.lowlevel > summary { cursor: pointer; color: var(--muted); font-size: .92rem;
    font-weight: 600; }
  details.lowlevel[open] > summary { margin-bottom: .5rem; }
  details.lowlevel h2 { margin-top: .6rem; font-size: 1.05rem; }
  .toc { font-size: .8rem; color: var(--muted); margin: .2rem 0 1rem; line-height: 1.9; }
  .toc a { color: var(--accent); text-decoration: none; white-space: nowrap; margin-right: .9rem; }
  .sd { color: var(--muted); font-size: .78em; font-weight: 400; }
  sup.fn { line-height: 0; }
  sup.fn a { text-decoration: none; color: var(--accent); font-weight: 600; font-size: .78em;
    padding: 0 1px; }
  ol.footnotes { color: var(--muted); font-size: .82rem; line-height: 1.6; padding-left: 1.3rem;
    margin-top: .6rem; }
  ol.footnotes li { margin-bottom: .4rem; }
  ol.footnotes li:target { color: var(--fg); background: #f3ede4; border-radius: 4px; }
"""

# Pane toggle (calls the optional window.__onModeChange hook, e.g. the key widget),
# column-sort with 'ref' rows pinned on top and empty cells sinking last, and the
# floating #tipbox for header tooltips.
SHARED_JS = r"""
(function(){
  var tog = document.querySelector('.mode-toggle');
  function setMode(m){
    document.querySelectorAll('.mode-pane').forEach(function(p){ p.hidden = p.dataset.mode !== m; });
    if (tog) tog.querySelectorAll('button').forEach(function(b){ b.setAttribute('aria-pressed', String(b.dataset.mode === m)); });
    if (window.__onModeChange) window.__onModeChange(m);
  }
  if (tog){
    tog.querySelectorAll('button').forEach(function(b){ b.addEventListener('click', function(){ setMode(b.dataset.mode); }); });
    setMode(tog.dataset.default || 'all');
  }

  function makeSortable(table){
    var head = table.tHead, body = table.tBodies[0];
    if (!head || !body) return;
    var cells = head.rows[0].cells;
    Array.prototype.forEach.call(cells, function(th, idx){
      th.addEventListener('click', function(){
        var asc = th.dataset.dir !== 'asc';
        Array.prototype.forEach.call(cells, function(h){ h.removeAttribute('data-dir'); });
        th.dataset.dir = asc ? 'asc' : 'desc';
        var num = function(c){ var n = parseFloat(c.textContent.trim().replace(/[%,+\s]/g, '')); return isNaN(n) ? null : n; };
        var rows = Array.prototype.slice.call(body.rows);
        var refs = rows.filter(function(r){ return r.classList.contains('ref'); });
        var data = rows.filter(function(r){ return !r.classList.contains('ref'); });
        data.sort(function(a, b){
          var ka = num(a.cells[idx]), kb = num(b.cells[idx]);
          if (ka === null && kb === null){ var c = a.cells[idx].textContent.trim().localeCompare(b.cells[idx].textContent.trim()); return asc ? c : -c; }
          if (ka === null) return 1;
          if (kb === null) return -1;
          return asc ? ka - kb : kb - ka;
        });
        refs.concat(data).forEach(function(r){ body.appendChild(r); });
      });
    });
  }
  document.querySelectorAll('table.sortable').forEach(makeSortable);

  // Jump nav: long pages get a compact section index built from their h2s
  // (skipping ones inside collapsed low-level <details>).
  var h2s = Array.prototype.filter.call(document.querySelectorAll('.wrap > h2, .wrap h2'), function(h){
    return !h.closest('details.lowlevel');
  });
  if (h2s.length >= 4){
    var toc = document.createElement('nav'); toc.className = 'toc';
    h2s.forEach(function(h, i){
      if (!h.id) h.id = 'sec-' + (i + 1);
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = (h.childNodes[0] && h.childNodes[0].textContent || h.textContent).trim();
      toc.appendChild(a);
    });
    var anchor = document.querySelector('.mode-toggle') || document.querySelector('.wrap h1');
    if (anchor) anchor.insertAdjacentElement('afterend', toc);
  }

  var box = document.createElement('div'); box.id = 'tipbox'; document.body.appendChild(box);
  function show(el){
    var t = el.getAttribute('data-tip'); if (!t) return;
    box.textContent = t; box.style.display = 'block';
    var r = el.getBoundingClientRect();
    box.style.left = Math.max(6, Math.min(r.left, window.innerWidth - 252)) + 'px';
    var top = r.top - box.offsetHeight - 6; if (top < 6) top = r.bottom + 6;
    box.style.top = top + 'px';
  }
  function hide(){ box.style.display = 'none'; }
  document.addEventListener('mouseover', function(e){ var el = e.target.closest && e.target.closest('.tip'); if (el) show(el); });
  document.addEventListener('mouseout', function(e){ if (e.target.closest && e.target.closest('.tip')) hide(); });
  document.addEventListener('focusin', function(e){ var el = e.target.closest && e.target.closest('.tip'); if (el) show(el); });
  document.addEventListener('focusout', hide);
})();
"""


def _nav(active):
    links = "\n".join(
        f'  <a href="{f}"{" class=\"active\"" if f == active else ""}>{lbl}</a>'
        for f, lbl in TABS)
    return f'<nav class="tabs">\n{links}\n</nav>'


def page(title, active, body, extra_css="", extra_head=""):
    """The full HTML document: head (style.css link + shared CSS + extra_css),
    nav tabs with `active` (a TABS filename) highlighted, the .wrap'd body, and
    one copy of the shared JS. Page-specific <script>/<style> for the body can
    also be embedded in `body` directly — it runs before the shared JS.
    fnote() markers in the body are numbered here and their references appended
    as a footnote section."""
    body = _apply_footnotes(body)
    style = SHARED_CSS + (extra_css or "")
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="style.css?v=22">
<style>{style}</style>
{extra_head}
</head><body>
{_nav(active)}
<div class="wrap">
{body}
</div>
<script>{SHARED_JS}</script>
</body></html>"""
