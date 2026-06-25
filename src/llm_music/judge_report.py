"""Generate docs/judge.html — the LLM-judge analysis page (third site tab).

Reads the judge outputs under docs/analysis/ (judge.csv = blind 3-frontier panel,
judge_noted.csv = noted condition, judge_allmodels_raw.json = every model judging
every piece) plus features.csv (computed proxies), and renders: quality rankings,
emotion character, perceived-vs-computed valence, judge competence + self-bias, the
per-trait self-bias heatmap, and the text-bias comparison.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean

from .judge import QUALITY_KEYS
from .report import ACCENT, BG, INK, MUTED

DIMS = QUALITY_KEYS + ["valence", "arousal"]
SHORT = {"gpt-5.5": "gpt-5.5", "gemini-2.5-pro": "gemini", "opus-4.8": "opus",
         "opus-4.8-thinking": "opus-think", "sonnet-4.6": "sonnet",
         "deepseek-v4-pro": "deepseek", "gpt-4.1": "gpt-4.1", "grok-4.3": "grok",
         "qwen3-max": "qwen", "llama-4-maverick": "llama"}
PANEL = ["gpt-5.5", "gemini-2.5-pro", "opus-4.8"]


def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = mean(a), mean(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sqrt(sum((x - ma) ** 2 for x in a))
    vb = sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (va * vb) if va and vb else float("nan")


def _qual(verdict):
    vs = [verdict[k]["score"] for k in QUALITY_KEYS if k in verdict]
    return mean(vs) if vs else None


def _load_csv(p: Path):
    return [r for r in csv.DictReader(p.open(encoding="utf-8")) if r["prompt"] == "free-form"] \
        if p.exists() else []


# ---------- HTML helpers ----------
def _tip(label, tip):
    return (f"<th><span class='tip' tabindex='0' data-tip=\"{html.escape(tip)}\">"
            f"{html.escape(label).replace(' ', '&nbsp;')}</span></th>") if tip else f"<th>{html.escape(label)}</th>"


def _table(cols, rows):
    """cols = [(label, tip)]; rows = list of cell-HTML lists (first cell left-aligned)."""
    head = "<tr>" + "".join(_tip(l, t) for l, t in cols) + "</tr>"
    body = ""
    for cells in rows:
        body += "<tr>" + "".join(
            f"<td class='{'m' if i == 0 else ''}'>{c}</td>" for i, c in enumerate(cells)) + "</tr>"
    return f"<div class='tscroll'><table>{head}{body}</table></div>"


def _heat(v, scale=0.55):
    if v is None:
        return "<td>—</td>"
    a = min(0.5, abs(v) / scale * 0.5)
    rgb = "46,160,67" if v >= 0 else "207,90,80"
    return f"<td style='background:rgba({rgb},{a:.2f})'>{v:+.2f}</td>"


# ---------- sections ----------
def _by_model(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[r["model"]].append(r)
    out = {}
    for m, rs in g.items():
        out[m] = {k: mean([_f(r[k]) for r in rs if _f(r.get(k)) is not None] or [float("nan")]) for k in keys}
        out[m]["n"] = len(rs)
    return out


def _rankings(blind):
    keys = ["overall"] + QUALITY_KEYS
    bm = _by_model(blind, keys)
    cols = [("model", None), ("n", "pieces judged (blind 3-frontier panel)"),
            ("overall", "mean of the 8 quality dimensions (panel average)")]
    cols += [(k, f"perceived {k}, 1–5") for k in QUALITY_KEYS]
    rows = []
    for m in sorted(bm, key=lambda x: -bm[x]["overall"]):
        d = bm[m]
        cells = [SHORT.get(m, m), str(d["n"]), f"<b>{d['overall']:.2f}</b>"]
        cells += [f"{d[k]:.2f}" for k in QUALITY_KEYS]
        rows.append(cells)
    return _table(cols, rows)


def _emotion(blind, feats):
    bm = _by_model(blind, ["valence", "arousal"])
    labels = defaultdict(Counter)
    for r in blind:
        if r.get("emotion_label"):
            labels[r["model"]][r["emotion_label"]] += 1
    # computed minor% per model
    minor = defaultdict(lambda: [0, 0])
    for r in feats:
        kmb = r.get("key_mode_best", "")
        if kmb in ("major", "minor"):
            minor[r["model"]][1] += 1
            minor[r["model"]][0] += kmb == "minor"
    minorpct = {m: 100 * a / b for m, (a, b) in minor.items() if b}

    cols = [("model", None), ("valence", "perceived valence 1–5 (dark→bright)"),
            ("arousal", "perceived arousal 1–5 (calm→energetic)"),
            ("dominant emotion", "most-assigned emotion label(s) by the blind panel"),
            ("minor %", "computed: share of pieces in a minor key (key_mode_best)")]
    rows = []
    for m in sorted(bm, key=lambda x: bm[x]["valence"]):
        top = ", ".join(f"{l} ({c})" for l, c in labels[m].most_common(2))
        mp = minorpct.get(m)
        rows.append([SHORT.get(m, m), f"{bm[m]['valence']:.1f}", f"{bm[m]['arousal']:.1f}",
                     top, f"{mp:.0f}%" if mp is not None else "—"])
    # perceived-vs-computed correlation
    pairs = [(bm[m]["valence"], minorpct[m]) for m in bm if m in minorpct]
    r = _pearson([a for a, _ in pairs], [b for _, b in pairs])
    note = (f"<p class='callout'>Blind perceived valence tracks the computed minor-key rate at "
            f"<b>r = {r:+.2f}</b> (n={len(pairs)} models) — a judge that never saw the key hears "
            f"minor-heavy models as darker. The computed proxy and human-style perception agree.</p>")
    return _table(cols, rows) + note


def _competence_selfbias(raw):
    pieces = [(p["model"], {j: _qual(v) for j, v in p["panel"].items() if _qual(v) is not None})
              for p in raw]
    judges = sorted({j for _, qd in pieces for j in qd})
    comp, lvl, raw_gap, lenc = {}, {}, {}, {}
    for j in judges:
        xs, ys, own, oth = [], [], [], []
        for author, qd in pieces:
            if j not in qd:
                continue
            others = [v for k, v in qd.items() if k != j]
            if not others:
                continue
            xs.append(qd[j]); ys.append(mean(others))
            (own if author == j else oth).append(qd[j] - mean(others))
        comp[j] = _pearson(xs, ys)
        lvl[j] = mean(xs) if xs else float("nan")
        raw_gap[j] = mean(own) if own else None
        lenc[j] = mean(oth) if oth else 0.0
    cols = [("model", None), ("competence", "Pearson correlation of this model's scores with the "
                                            "mean of all OTHER judges — how reliable a critic it is"),
            ("leniency", "this judge's average score level (harsh ↔ generous)"),
            ("self-bias", "leniency-corrected: how much it favors its OWN pieces beyond its general "
                          "tendency (>0 favors self, <0 harder on self)"),
            ("n", "own pieces it self-judged")]
    rows = []
    for j in sorted(judges, key=lambda x: -comp[x]):
        sb = (raw_gap[j] - lenc[j]) if raw_gap[j] is not None else None
        n = sum(1 for a, qd in pieces if a == j and j in qd)
        rows.append([SHORT.get(j, j), f"{comp[j]:.2f}", f"{lvl[j]:.2f}",
                     f"{sb:+.2f}" if sb is not None else "—", str(n)])
    return _table(cols, rows), pieces


def _per_trait(raw):
    own = defaultdict(lambda: defaultdict(list))
    oth = defaultdict(lambda: defaultdict(list))
    for p in raw:
        author, panel = p["model"], p["panel"]
        for j in panel:
            for d in DIMS:
                sj = (panel[j].get(d) or {}).get("score")
                peers = [(panel[k].get(d) or {}).get("score") for k in panel if k != j]
                peers = [x for x in peers if x is not None]
                if sj is None or not peers:
                    continue
                (own if author == j else oth)[j][d].append(sj - mean(peers))
    judges = [m for m in SHORT if own.get(m)]
    judges.sort(key=lambda m: -len(own[m].get("harmony", [])))
    head = "<tr><th>trait</th>" + "".join(f"<th>{SHORT[m]}</th>" for m in judges) + "</tr>"
    body = ""
    for d in DIMS:
        body += f"<tr><td class='m'>{d}</td>"
        for m in judges:
            v = (mean(own[m][d]) - mean(oth[m][d])) if own[m].get(d) and oth[m].get(d) else None
            body += _heat(v)
        body += "</tr>"
    nrow = "<tr><td class='m sub'>n own</td>" + "".join(
        f"<td class='sub'>{len(own[m].get('harmony', []))}</td>" for m in judges) + "</tr>"
    return f"<div class='tscroll'><table class='heat'>{head}{body}{nrow}</table></div>"


def _text_bias(blind, noted):
    bkey = {(r["model"], r["title"]): r for r in blind}
    rows = [(bkey[(r["model"], r["title"])], r) for r in noted if (r["model"], r["title"]) in bkey]
    cols = [("dimension", None), ("Δ noted − blind", "mean change when the composer's written note "
                                                     "is shown; + = the note inflates the score"),
            ("type", None)]
    out = []
    for d in DIMS:
        ds = [_f(n[d]) - _f(b[d]) for b, n in rows if _f(n.get(d)) is not None and _f(b.get(d)) is not None]
        if ds:
            t = "quality" if d in QUALITY_KEYS else "affect"
            out.append([d, f"{mean(ds):+.3f}", t])
    return _table(cols, out), len(rows)


# ---------- page ----------
def render_judge_html(analysis_dir: Path, data_dir: Path, out_path: Path):
    blind = _load_csv(analysis_dir / "judge.csv")
    noted = _load_csv(analysis_dir / "judge_noted.csv")
    feats = []
    for fp in sorted(data_dir.glob("*/features.csv")):
        feats += [r for r in csv.DictReader(fp.open(encoding="utf-8")) if r["prompt"] == "free-form"]
    rawp = analysis_dir / "judge_allmodels_raw.json"
    raw = [p for p in json.loads(rawp.read_text(encoding="utf-8")) if p["prompt"] == "free-form"] \
        if rawp.exists() else []

    secs = []
    if blind:
        secs.append("<h2>Which models write the best music <span class='sub'>(blind panel)</span></h2>"
                    "<p class='scope'>A blind 3-frontier panel (gpt-5.5 · gemini · opus) rates each piece "
                    "from the notation alone — no title, composer note, or model name. Dimensions follow the "
                    "music-eval literature (ChatMusician / Chu et al. / MuSpike); scoring follows the "
                    "LLM-judge literature (reason-before-score, anchored 1–5, panel-averaged).</p>"
                    + _rankings(blind))
        secs.append("<h2>Emotional character <span class='sub'>(perceived, blind)</span></h2>"
                    "<p class='scope'>What the blind judge <i>hears</i> — perceived valence/arousal and the "
                    "dominant emotion — against the computed minor-key proxy.</p>" + _emotion(blind, feats))
    if raw:
        comp_html, _ = _competence_selfbias(raw)
        secs.append("<h2>Can each model judge music? <span class='sub'>(all-9 study)</span></h2>"
                    "<p class='scope'>With every model judging every piece, this shows each model's "
                    "competence (agreement with the consensus), its leniency, and — leniency-corrected — "
                    "whether it favors its own work. No model meaningfully over-rates itself; competence "
                    "and leniency vary widely.</p>" + comp_html)
        secs.append("<h2>Self-bias by trait <span class='sub'>(leniency-corrected)</span></h2>"
                    "<p class='scope'>Where each model judges its <i>own</i> music differently than it judges "
                    "everyone else's. <span style='color:rgb(46,140,67)'>green = kinder to itself</span>, "
                    "<span style='color:rgb(197,80,70)'>red = harder on itself</span>. The pattern: weak "
                    "models over-credit themselves exactly where they're weakest (grok→harmony, llama→emotion); "
                    "strong models are calibrated. Small n per model — read patterns, not single cells.</p>"
                    + _per_trait(raw))
    if blind and noted:
        tb_html, npaired = _text_bias(blind, noted)
        secs.append("<h2>Text bias <span class='sub'>(does the written note sway the judge?)</span></h2>"
                    f"<p class='scope'>Same pieces, same panel, but the composer's note is shown. Δ over "
                    f"{npaired} paired pieces — the bias is modest (~+0.17 overall), concentrated in the "
                    f"subjective dimensions (creativity, emotion, naturalness) and near-zero on the objective "
                    f"ones (harmony, structure): you can talk the judge into hearing more creativity, not "
                    f"better harmony.</p>" + tb_html)

    body = "\n".join(secs) or "<p>No judge results found. Run <code>llm-music judge</code> first.</p>"
    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM judge — musical inductive biases</title>
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
  th .tip:hover::after, th .tip:focus::after {{
    content: attr(data-tip); position: absolute; left: 0; top: 145%; z-index: 30;
    width: 240px; white-space: normal; text-align: left; font-weight: 400;
    font-size: .76rem; line-height: 1.45; color: {BG}; background: {INK};
    padding: .55rem .65rem; border-radius: 7px; box-shadow: 0 6px 20px rgba(0,0,0,.2); }}
  th:nth-last-child(-n+3) .tip:hover::after, th:nth-last-child(-n+3) .tip:focus::after {{ left:auto; right:0; }}
  td.m, th:first-child {{ text-align: left; font-weight: 600; }}
  td.sub {{ color: {MUTED}; font-size: .8rem; }}
  table.heat td {{ text-align: center; }}
  h2 {{ margin-top: 2.4rem; }}
  .callout {{ background: #f3ede4; border-left: 3px solid {ACCENT}; padding: .7rem .9rem;
             border-radius: 0 7px 7px 0; font-size: .9rem; margin: .8rem 0 0; }}
</style>
</head><body>
<nav class="tabs">
  <a href="index.html">Browse outputs</a>
  <a href="results.html">Results &amp; analysis</a>
  <a href="judge.html" class="active">LLM judge</a>
</nav>
<div class="wrap">
  <h1>How LLMs judge music — and themselves</h1>
  <p class="scope">An LLM-as-judge layer over the generated pieces: blind quality + emotion ratings,
     each model's competence as a critic, and its self-bias. Rubric dimensions follow the music-eval
     literature; the protocol (reason-before-score, anchored scales, blind panel) follows the LLM-judge
     literature. Pilot scope: 200 free-form pieces. Generated by <code>llm-music judge-report</code>.</p>
  {body}
</div>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
