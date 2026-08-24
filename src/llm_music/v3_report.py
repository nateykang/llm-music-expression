"""v3 corpus results page, organized around the four EDA questions:

  1. Do LLMs write simpler music than humans? (Bach-chorale baseline,
     literature metrics only)
  2. Do more capable LLMs write more complex music? (average complexity rank)
  3. What does each model prefer? (key/mode, tempo, dynamics — style, not skill)
  4. Do model families share traits? (the same lens, by family)

Builds docs/v3/results.html from the three v3 batches' features.csv +
data.json and the cached Bach reference; extends the corpus pill on the
v1/v2 results pages. Research-notebook conventions: prose figcaptions,
sortable tables, no summary widgets.
"""

from __future__ import annotations

import csv
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

from .config import DOCS_DIR
from .report import KEY_WIDGET_CSS, KEY_WIDGET_TMPL, _key_reference
from .report_common import (cell, details_section, fnote, heat, page, paned,
                            table, tip, toggle)

V3_BATCHES = ["20260819_201530__models_40_prompts_3",
              "20260819_203512__models_2_prompts_3",
              "20260820_061907__models_42_prompts_3"]

# base model -> (family, release month). Preview-only models carry their
# preview date; the paper's roster table documents access dates separately.
RELEASE = {
    "gpt-5": ("openai", "2025-08"), "gpt-5.1": ("openai", "2025-11"),
    "gpt-5.2": ("openai", "2025-12"), "gpt-5.4": ("openai", "2026-03"),
    "gpt-5.5": ("openai", "2026-04"), "gpt-5.6": ("openai", "2026-07"),
    "opus-4.1": ("anthropic", "2025-08"), "sonnet-4.5": ("anthropic", "2025-09"),
    "opus-4.5": ("anthropic", "2025-11"), "sonnet-4.6": ("anthropic", "2026-02"),
    "opus-4.8": ("anthropic", "2026-05"), "fable-5": ("anthropic", "2026-08"),
    "gemini-3-flash": ("google", "2025-12"), "gemini-3.1-pro": ("google", "2026-02"),
    "gemini-3.5-flash": ("google", "2026-05"), "gemini-3.6-flash": ("google", "2026-07"),
    "gemini-3.7-flash": ("google", "2026-08"),
    "grok-4.3": ("xai", "2026-04"), "grok-4.6": ("xai", "2026-08"),
    "kimi-k2": ("kimi", "2025-07"), "kimi-k3": ("kimi", "2026-07"),
}
BASES = list(RELEASE)  # insertion order groups families, oldest first

# The complexity panel: canonical symbolic-music metrics from the literature
# (no homegrown measures). direction +1 = higher reads as "more complex".
COMPLEX_PANEL = [
    ("pitch_class_entropy", "pitch-class entropy", +1, "MusPy",
     "Entropy of the 12 pitch classes (Dong et al., 2020). Higher = more "
     "chromatic / varied pitch material."),
    ("pitch_entropy", "pitch entropy", +1, "MusPy",
     "Entropy over exact pitches (Dong et al., 2020). Higher = more varied."),
    ("n_pitches_used", "pitches used", +1, "MusPy",
     "Count of distinct pitches (Dong et al., 2020)."),
    ("pitch_range", "pitch range", +1, "MusPy",
     "Semitones between lowest and highest note (Dong et al., 2020)."),
    ("polyphony", "polyphony", +1, "MusPy",
     "Mean simultaneous notes (Dong et al., 2020). Higher = thicker texture."),
    ("scale_consistency", "scale consistency", -1, "MusPy",
     "Share of notes in the best-fitting major/minor scale (Dong et al., 2020). "
     "LOWER = more chromatic, so lower reads as more complex."),
    ("groove_consistency", "groove consistency", -1, "MusPy",
     "Bar-to-bar rhythmic self-similarity (Dong et al., 2020). LOWER = more "
     "rhythmic variety."),
    ("structureness", "structureness", -1, "Wu & Yang",
     "Repetition of material across the piece (Wu & Yang, 2020). LOWER = more "
     "through-composed; high values are simple/repetitive in the information "
     "sense. Bach chorales sit near 0.9."),
]


def _table_raw(cols, rows):
    """Like report_common.table() but rows are lists of fully-formed <td> cells
    (needed when cells carry their own style/class, e.g. heat())."""
    head = "<thead><tr>" + "".join(tip(l, t) for l, t in cols) + "</tr></thead>"
    body = "<tbody>" + "".join("<tr>" + "".join(r) + "</tr>" for r in rows) + "</tbody>"
    return f"<div class='tscroll'><table class='sortable'>{head}{body}</table></div>"


def _num(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _vals(rows, col):
    return [x for r in rows if (x := _num(r.get(col))) is not None]


def _med(rows, col):
    vals = _vals(rows, col)
    return st.median(vals) if vals else None


def _quart(rows, col):
    vals = _vals(rows, col)
    if len(vals) < 4:
        return (None, st.median(vals) if vals else None, None)
    q = st.quantiles(vals, n=4)
    return (q[0], st.median(vals), q[2])


def _base(arm):
    return arm[:-9] if arm.endswith("-thinking") else arm


def _load():
    feats, pieces = [], []
    for b in V3_BATCHES:
        with (DOCS_DIR / "data" / b / "features.csv").open(encoding="utf-8", newline="") as f:
            feats.extend(csv.DictReader(f))
        pieces.extend(json.loads((DOCS_DIR / "data" / b / "data.json")
                                 .read_text(encoding="utf-8"))["pieces"])
    return feats, pieces


def _bach_rows():
    d = json.loads((DOCS_DIR / "analysis" / "bach_reference.json").read_text(encoding="utf-8"))
    return d["rows"]


def _abc(feats):
    return [r for r in feats if r.get("mode") in ("abc", "smt-abc")]


def _declared_mode(r):
    return r.get("key_declared_mode") or r.get("key_mode") or ""


def _minor_pct(rows):
    modes = [m for r in rows if (m := _declared_mode(r)) in ("major", "minor")]
    return 100.0 * sum(m == "minor" for m in modes) / len(modes) if modes else None


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 4:
        return None
    xs, ys = zip(*pairs)
    return st.correlation(rank(list(xs)), rank(list(ys)))


# ---------- corpus health ----------

def _completion_section(pieces):
    ok, tot = Counter(), Counter()
    for p in pieces:
        key = (p["model"], "abc" if p["mode"] in ("abc", "smt-abc") else "codegen")
        tot[key] += 1
        ok[key] += bool(p.get("ok"))
    rows = []
    for b in BASES:
        fam, rel = RELEASE[b]
        cells = [f"<td class='m'>{b}</td>", f"<td class='m'>{fam}</td>",
                 f"<td class='m'>{rel}</td>"]
        for gm in ("abc", "codegen"):
            for arm in (b, b + "-thinking"):
                n, k = tot.get((arm, gm), 0), ok.get((arm, gm), 0)
                cells.append("<td>—</td>" if n == 0 else
                             f"<td>{k}/{n}</td>" if k == n else
                             f"<td style='background:rgba(207,90,80,{min(.5, (n-k)/n)})'>{k}/{n}</td>")
        rows.append(cells)
    cols = [("model", None), ("family", None), ("released", None),
            ("ABC", "Pieces that produced a valid score, of pieces attempted."),
            ("ABC think", None), ("code-gen", None), ("code-gen think", None)]
    return ("<h2>Corpus health: who completed what</h2>"
            "<figure>" + _table_raw(cols, rows) + "</figure>")


# ---------- Q1: simpler than humans? ----------

def _q1_bach_section(feats, bach):
    llm_abc = _abc(feats)
    llm_code = [r for r in feats if r.get("mode") == "codegen"]
    rows = []
    for col, label, direction, src, tp in COMPLEX_PANEL:
        b1, bmed, b3 = _quart(bach, col)
        a1, amed, a3 = _quart(llm_abc, col)
        cmed = _med(llm_code, col)
        if amed is None or bmed is None:
            verdict = "<td>—</td>"
        else:
            outside = (amed < b1) if b1 is not None else False
            above = (amed > b3) if b3 is not None else False
            toward_simple = (amed < bmed) if direction > 0 else (amed > bmed)
            strong = outside or above
            word = "simpler" if toward_simple else "more complex"
            shade = ("rgba(207,90,80,.25)" if toward_simple else "rgba(46,160,67,.22)") \
                if strong else "none"
            verdict = (f"<td style='background:{shade}'>{word}{'' if strong else ' (≈)'}"
                       "</td>")
        fmt = "f2" if col not in ("n_pitches_used", "pitch_range") else "f0"
        iqr = lambda a, m, c: ("—" if m is None else
                               cell(m, fmt) + (f" <span class='sd'>[{cell(a, fmt)}–{cell(c, fmt)}]</span>"
                                               if a is not None else ""))
        rows.append([
            f"<td class='m'>" + tip(label, tp).replace("<th>", "").replace("</th>", "")
            .replace("<th ", "<span ") + "</td>",
            f"<td class='m'>{src}</td>",
            f"<td class='m'>{'higher = complex' if direction > 0 else 'lower = complex'}</td>",
            f"<td>{iqr(b1, bmed, b3)}</td>",
            f"<td>{iqr(a1, amed, a3)}</td>",
            f"<td>{cell(cmed, fmt)}</td>",
            verdict,
        ])
    cols = [("metric", None), ("source", "Where the metric comes from — MusPy is Dong "
                                         "et al. (2020); structureness is Wu & Yang (2020)."),
            ("direction", None),
            ("Bach chorales", "Median [IQR] over 40 chorales from music21's built-in "
                              "corpus, measured with the identical pipeline."),
            ("LLM — ABC", "Median [IQR] over all 2,515 ABC pieces."),
            ("LLM — code-gen", "Median over the 2,292 completed code-gen pieces."),
            ("verdict", "Direction of the LLM–Bach difference, shaded when the LLM "
                        "median falls outside Bach's interquartile range.")]
    return (
        "<h2>1 · Do LLMs write simpler music than humans? <span class='sub'>(Bach-chorale baseline)</span></h2>"
        "<figure><figcaption>Canonical symbolic-music metrics (nothing homegrown) computed "
        "identically on Bach chorales and on the corpus. Caveats worth carrying: the "
        "chorales are one genre (4-voice SATB hymns), n=40, and 'complex' is a reading of "
        "each metric's direction, not a single ground truth.</figcaption>"
        + _table_raw(cols, rows) + "</figure>")


# ---------- Q2: capability vs complexity ----------

def _complexity_ranks(feats):
    """Per base model: rank on each panel metric (1 = most complex, ABC pieces),
    plus the mean rank. Returns {base: {"ranks": {col: r}, "avg": x}}."""
    sub = _abc(feats)
    med = {b: {} for b in BASES}
    for b in BASES:
        mine = [r for r in sub if _base(r["model"]) == b]
        for col, *_ in COMPLEX_PANEL:
            med[b][col] = _med(mine, col)
    out = {b: {"ranks": {}} for b in BASES}
    for col, _l, direction, *_ in COMPLEX_PANEL:
        vals = [(b, med[b][col]) for b in BASES if med[b][col] is not None]
        vals.sort(key=lambda t: t[1], reverse=(direction > 0))
        for i, (b, _v) in enumerate(vals, 1):
            out[b]["ranks"][col] = i
    for b in BASES:
        rs = list(out[b]["ranks"].values())
        out[b]["avg"] = sum(rs) / len(rs) if rs else None
    return out


def _q2_ranking_section(feats):
    ranks = _complexity_ranks(feats)
    order = sorted((b for b in BASES if ranks[b]["avg"] is not None),
                   key=lambda b: ranks[b]["avg"])
    rows = []
    for i, b in enumerate(order, 1):
        fam, rel = RELEASE[b]
        r = ranks[b]
        rows.append([str(i), b, fam, rel, cell(r["avg"], "f1")]
                    + [cell(r["ranks"].get(col), "int") for col, *_ in COMPLEX_PANEL])
    cols = ([("#", None), ("model", None), ("family", None), ("released", None),
             ("avg rank", "Mean of the eight per-metric ranks; 1 = most complex "
                          "of the 21 models on that metric. ABC pieces, ±thinking "
                          "arms pooled.")]
            + [(lab, tp) for _c, lab, _d, _s, tp in COMPLEX_PANEL])
    body = table(cols, rows, left=4)
    # supporting evidence: release-date trend on the same panel
    sub = _abc(feats)
    order_rel = {b: i for i, b in enumerate(sorted(BASES, key=lambda b: RELEASE[b][1]))}
    trows = []
    for col, lab, direction, _s, _t in COMPLEX_PANEL:
        xs, ys = [], []
        for b in BASES:
            mine = [r for r in sub if _base(r["model"]) == b]
            if mine:
                xs.append(order_rel[b])
                ys.append(_med(mine, col))
        rho = _spearman(xs, ys)
        rho_c = rho * direction if rho is not None else None  # + = newer more complex
        trows.append([f"<td class='m'>{lab}</td>",
                      heat(rho_c, 1.0) if rho_c is not None else "<td>—</td>"])
    trend = _table_raw([("metric", None),
                        ("ρ (newer → more complex)",
                         "Spearman correlation between release rank and the metric, "
                         "oriented so positive = newer models more complex. n=21 "
                         "models; |ρ| under ~0.45 is within noise.")], trows)
    return (
        "<h2>2 · Do more capable LLMs write more complex music?</h2>"
        "<figure><figcaption>Models ranked by their average rank across the same eight "
        "literature metrics used against Bach (1 = most complex). Release date is the "
        "capability proxy in the trend table — and on ABC the trend runs mostly the "
        "wrong way for the hypothesis: newer models write more consonant, more "
        "repetitive, thinner-textured music.</figcaption>"
        + body + "<br>" + trend + "</figure>")


def _thinking_section(feats):
    def pane(mode_key):
        gm = {"abc": ("abc", "smt-abc"), "code": ("codegen",),
              "all": ("abc", "smt-abc", "codegen")}[mode_key]
        sub = [r for r in feats if r.get("mode") in gm]
        rows = []
        for b in BASES:
            t = [r for r in sub if r["model"] == b + "-thinking"]
            n = [r for r in sub if r["model"] == b]
            if not (t and n):
                continue
            deltas = [
                ((_med(t, "length_seconds") or 0) - (_med(n, "length_seconds") or 0), 30),
                ((_med(t, "note_density") or 0) - (_med(n, "note_density") or 0), 0.8),
                ((_med(t, "pitch_class_entropy") or 0) - (_med(n, "pitch_class_entropy") or 0), 0.25),
                ((_med(t, "groove_consistency") or 0) - (_med(n, "groove_consistency") or 0), 0.15),
                ((_med(t, "structureness") or 0) - (_med(n, "structureness") or 0), 0.1),
                ((_med(t, "polyphony") or 0) - (_med(n, "polyphony") or 0), 0.6),
            ]
            rows.append([f"<td class='m'>{b}</td>"] + [heat(d, sc) for d, sc in deltas])
        return _table_raw([("model", None), ("len s", None), ("notes/beat", None),
                           ("pc entropy", None), ("groove", None), ("structure", None),
                           ("polyphony", None)], rows)
    return (
        "<h3>Thinking − non-thinking, same model <span class='sub'>(median deltas)</span></h3>"
        + toggle(default="abc")
        + "<figure><figcaption>The other capability axis: identical model, more reasoning. "
        "Green = the thinking arm is higher. For always-thinking models (fable-5, kimi-k3, "
        "gemini pro/3.5+ flash, grok-4.6) the pair is effort low vs high. Code-gen deltas "
        "carry a survivorship caveat: failed pieces (concentrated in early non-thinking "
        "arms) are missing non-randomly.</figcaption>"
        + paned(pane, default="abc") + "</figure>")


# ---------- Q3: preferences ----------

def _key_widget(feats):
    """The v1 interactive circle-of-fifths histogram, fed with v3 data: all
    three prompt variants, ±thinking arms pooled per base model (21 buttons)."""
    buckets = {"text": {}, "code": {}, "all": {}}
    for r in feats:
        tonic = (r.get("key_declared_tonic") or r.get("key_tonic") or "").replace("-", "b")
        mode = r.get("key_mode_best") or r.get("key_mode")
        if not tonic or mode in ("", "?", None):
            continue
        lab = tonic if mode == "major" else tonic + "m"
        rep = "code" if r.get("mode") == "codegen" else "text"
        b = _base(r["model"])
        for k in (rep, "all"):
            buckets[k].setdefault(b, Counter())[lab] += 1
    dists = {k: {m: dict(c) for m, c in v.items() if sum(c.values()) >= 10}
             for k, v in buckets.items()}
    models = [b for b in BASES if b in (set(dists["all"]) | set(dists["text"]))]
    return (KEY_WIDGET_TMPL
            .replace("in free-form, as a share of its pieces",
                     "across the three self-expression variants, as a share of its "
                     "pieces (±thinking arms pooled)")
            .replace("__FN_COF__", fnote("circle-of-fifths"))
            .replace("__FN_GIGA__", fnote("gigamidi"))
            .replace("__TEXT__", json.dumps(dists["text"]))
            .replace("__CODE__", json.dumps(dists["code"]))
            .replace("__ALL__", json.dumps(dists["all"]))
            .replace("__MODELS__", json.dumps(models))
            .replace("__REF__", json.dumps(_key_reference())))


def _pref_row(rows_abc, rows_code):
    minor = _minor_pct(rows_abc)
    keys = Counter()
    for r in rows_abc:
        tonic = r.get("key_declared_tonic") or r.get("key_tonic")
        mode = _declared_mode(r)
        if tonic and mode:
            keys[f"{tonic} {mode}"] += 1
    top = ", ".join(f"{k} {100*v/len(rows_abc):.0f}%" for k, v in keys.most_common(2)) \
        if rows_abc else "—"
    real_tempo = [r for r in rows_abc if _num(r.get("tempo_defaulted")) == 0]
    return [
        top,
        cell((minor or 0) / 100, "pct"),
        cell(_med(real_tempo, "tempo_bpm"), "f0"),
        cell(100 * len(real_tempo) / len(rows_abc) / 100 if rows_abc else None, "pct"),
    ]


PREF_COLS = [
    ("top keys", "Most frequent declared keys (ABC K:), share of the model's pieces."),
    ("minor", "Share of pieces in a minor key (declared where available)."),
    ("tempo", "Median written tempo (bpm), among pieces that set one."),
    ("sets tempo", "Share of ABC pieces that declare a tempo (Q:) at all."),
]



_DYN_USER = {"pppp": -2, "ppp": -1, "pp": 0, "p": 1, "mp": 2, "mf": 3,
             "f": 4, "ff": 5, "fff": 6, "ffff": 7}
_DYN_NAME = {v: k for k, v in _DYN_USER.items()}
_ABC_DYN_RE = re.compile(r"!(pppp|ppp|pp|mp|mf|p|ffff|fff|ff|f)!")


def _q3_dynamics_section(feats, pieces):
    """Dynamics preferences. Span uses the mark ladder where ff−pp = 5 and
    mf−mp = 1; changes are normalized per beat; the average dynamic is the mean
    mark on the same ladder (p and mf average to mp)."""
    abc = _abc(feats)
    code = [r for r in feats if r.get("mode") == "codegen"]
    abc_text = {}
    for pc in pieces:
        if pc.get("abc") and pc.get("ok"):
            abc_text.setdefault(_base(pc["model"]), []).append(pc["abc"])

    def stats(rows):
        withd = [r for r in rows if (_num(r.get("n_dynamic_marks")) or 0) > 0]
        pct = 100.0 * len(withd) / len(rows) if rows else None
        span = _med(withd, "dynamic_span")
        per_beat = []
        for r in withd:
            ch, nn, nd = (_num(r.get("dynamic_changes")) or 0,
                          _num(r.get("n_notes")), _num(r.get("note_density")))
            if nn and nd:
                per_beat.append(ch / (nn / nd))
        dpb = st.median(per_beat) if per_beat else None
        return pct, span, dpb

    rows = []
    for b in BASES:
        pa, sa, da = stats([r for r in abc if _base(r["model"]) == b])
        pcx, sc, dc = stats([r for r in code if _base(r["model"]) == b])
        levels = [_DYN_USER[m] for t in abc_text.get(b, []) for m in _ABC_DYN_RE.findall(t)]
        if levels:
            mean = sum(levels) / len(levels)
            name = _DYN_NAME.get(round(max(-2, min(7, mean))), "?")
            avg_dyn = f"{name} <span class='sd'>({mean:.1f})</span>"
        else:
            avg_dyn = "—"
        rows.append([b,
                     cell((pa or 0) / 100 if pa is not None else None, "pct"),
                     cell((pcx or 0) / 100 if pcx is not None else None, "pct"),
                     cell(sa, "f1"), cell(sc, "f1"),
                     cell(da, "f2"), cell(dc, "f2"),
                     avg_dyn])
    cols = [("model", None),
            ("uses dyn (ABC)", "Share of ABC pieces with any written dynamic mark."),
            ("uses dyn (code)", "Share of code-gen pieces with any written dynamic mark."),
            ("span (ABC)", "Median softest-to-loudest distance on the mark ladder "
                           "(ff−pp = 5, mf−mp = 1), among pieces with dynamics."),
            ("span (code)", None),
            ("Δ/beat (ABC)", "Median dynamic changes per beat among pieces with "
                             "dynamics — length-invariant."),
            ("Δ/beat (code)", None),
            ("avg dynamic", "Mean of all written marks on the same ladder (ABC): "
                            "a p piece and an mf piece average to mp.")]
    return (
        "<h3>Dynamics <span class='sub'>(written marks, not synthesis)</span></h3>"
        "<figure><figcaption>Models rarely write dynamics in ABC at all (7% of pieces "
        "corpus-wide — the same rate as earlier ABC corpora, so it is model behavior, "
        "not a pipeline artifact: dynamics decorations are rare in ABC training data) "
        "but use them freely as music21 Dynamic objects. The medium shapes the "
        "expressive vocabulary.</figcaption>"
        + table(cols, rows, left=1) + "</figure>")


def _q3_pref_section(feats, pieces):
    abc, code = _abc(feats), [r for r in feats if r.get("mode") == "codegen"]
    rows = []
    for b in BASES:
        rows.append([b] + _pref_row([r for r in abc if _base(r["model"]) == b],
                                    [r for r in code if _base(r["model"]) == b]))
    return (
        "<h2>3 · What does each model prefer? <span class='sub'>(style, not skill)</span></h2>"
        "<figure><figcaption>Tonality, tempo, and dynamics are choices, not capabilities — "
        "two models of equal skill can sit at opposite ends. ±thinking arms pooled.</figcaption>"
        + table([("model", None)] + PREF_COLS, rows, left=2) + "</figure>"
        + _q3_dynamics_section(feats, pieces)
        + _key_widget(feats))


# ---------- Q4: families ----------

def _q4_family_section(feats):
    ranks = _complexity_ranks(feats)
    abc, code = _abc(feats), [r for r in feats if r.get("mode") == "codegen"]
    fams = []
    for b in BASES:
        f = RELEASE[b][0]
        if f not in fams:
            fams.append(f)
    rows = []
    for fam in fams:
        bs = [b for b in BASES if RELEASE[b][0] == fam]
        fa = [r for r in abc if RELEASE.get(_base(r["model"]), ("?",))[0] == fam]
        fc = [r for r in code if RELEASE.get(_base(r["model"]), ("?",))[0] == fam]
        avg = [ranks[b]["avg"] for b in bs if ranks[b]["avg"] is not None]
        rows.append([fam, str(len(bs))] + _pref_row(fa, fc)
                    + [cell(sum(avg) / len(avg) if avg else None, "f1")])
    return (
        "<h2>4 · Do model families share traits?</h2>"
        "<figure><figcaption>The same preference lens as §3, pooled per lab, plus each "
        "family's mean complexity rank from §2 (lower = more complex). Family-level "
        "claims here are descriptive; per-model spread is visible in §2–3.</figcaption>"
        + table([("family", None), ("models", None)] + PREF_COLS
                + [("complexity rank", "Mean of the family's models' average complexity "
                                       "ranks from §2 (1 = most complex).")],
                rows, left=2) + "</figure>")


# ---------- appendix ----------

FEAT_COLS = [
    ("n", "Pieces aggregated in this row."),
    ("minor", "Share of pieces in a minor key."),
    ("len s", "Median rendered length in seconds."),
    ("notes/beat", "Median notes per quarter-note beat (tempo-invariant)."),
    ("pc entropy", "Pitch-class entropy (MusPy)."),
    ("rhy entropy", "Entropy of note-duration classes."),
    ("polyphony", "Mean simultaneous notes (MusPy)."),
    ("structure", "Structureness (Wu & Yang, 2020)."),
    ("consonance", "Share of consonant vertical intervals."),
    ("in-scale", "Share of notes inside the detected key's scale (MusPy)."),
    ("instr", "Median distinct instruments."),
    ("dyn Δ", "Median written dynamic changes."),
]


def _arm_row(arm, rows):
    return [
        arm,
        cell(len(rows), "int"),
        cell((_minor_pct(rows) or 0) / 100, "pct"),
        cell(_med(rows, "length_seconds"), "f0"),
        cell(_med(rows, "note_density"), "f2"),
        cell(_med(rows, "pitch_class_entropy"), "f2"),
        cell(_med(rows, "rhythm_entropy"), "f2"),
        cell(_med(rows, "polyphony"), "f2"),
        cell(_med(rows, "structureness"), "f2"),
        cell(_med(rows, "consonance_rate"), "pct"),
        cell(_med(rows, "pitch_in_scale_rate"), "pct"),
        cell(_med(rows, "n_instruments"), "f0"),
        cell(_med(rows, "dynamic_changes"), "f0"),
    ]


def _appendix(feats):
    def pane(mode_key):
        gm = {"abc": ("abc", "smt-abc"), "code": ("codegen",),
              "all": ("abc", "smt-abc", "codegen")}[mode_key]
        sub = [r for r in feats if r.get("mode") in gm]
        rows = []
        for b in BASES:
            for arm in (b, b + "-thinking"):
                mine = [r for r in sub if r["model"] == arm]
                if mine:
                    rows.append(_arm_row(arm, mine))
        return table([("arm", None)] + FEAT_COLS, rows)
    per_arm = (toggle(default="abc")
               + "<figure><figcaption>Every arm across all prompt variants; sort any "
               "column, tooltips define the features.</figcaption>"
               + paned(pane, default="abc") + "</figure>")
    sub = _abc(feats)
    prow = []
    for p in ("express-yourself", "uniquely-you", "emotional-state"):
        mine = [r for r in sub if r["prompt"] == p]
        prow.append([p, cell(len(mine), "int"),
                     cell((_minor_pct(mine) or 0) / 100, "pct"),
                     cell(_med(mine, "length_seconds"), "f0"),
                     cell(_med(mine, "note_density"), "f2"),
                     cell(_med(mine, "pitch_class_entropy"), "f2"),
                     cell(_med(mine, "valence"), "f2"),
                     cell(_med(mine, "arousal"), "f2")])
    prompts = ("<figure><figcaption>The three self-expression phrasings are the "
               "system-prompt manipulation. Differences are what one sentence of framing "
               "does to the music.</figcaption>"
               + table([("variant", None), ("n", None), ("minor", None), ("len s", None),
                        ("notes/beat", None), ("pc entropy", None),
                        ("valence", "Mode-based affect proxy, −1…1."),
                        ("arousal", "Tempo + density affect proxy, 0…1.")], prow))
    return ("<h2>Appendix</h2>"
            + details_section("Full per-arm feature table (42 arms × 12 features)", per_arm)
            + details_section("Prompt-variant effects", prompts))


# ---------- corpus pill + page assembly ----------

V3_PILL = ("<a href='{href}' style=\"font-size:.85rem;padding:4px 13px;border-radius:7px;"
           "border:1px solid #cbb99a;background:#fff;color:var(--fg);text-decoration:none;"
           "font-weight:400\">v3 — 42 arms × 60</a>")


def _corpus_toggle_v3():
    a = ("<a href='{h}' style=\"font-size:.85rem;padding:4px 13px;border-radius:7px;"
         "border:1px solid #cbb99a;background:#fff;color:var(--fg);text-decoration:none;"
         "font-weight:400\">{t}</a>")
    active = ("<span style=\"font-size:.85rem;padding:4px 13px;border-radius:7px;"
              "border:1px solid var(--accent);background:var(--accent);color:var(--bg);"
              "font-weight:400\">v3 — 42 arms × 60</span>")
    return ("<div id='corpus-toggle' style='max-width:980px;margin:.9rem auto -1.1rem;"
            "padding:0 1.25rem;display:flex;gap:8px;align-items:center'>"
            "<span style='font-weight:600;font-size:.9rem;color:var(--fg)'>Corpus</span>"
            + a.format(h="../results.html", t="v1 (pilot)")
            + a.format(h="../v2/results.html", t="v2 — 15 models × 100")
            + active + "</div>")


def _inject_pill(path: Path, href: str):
    html_text = path.read_text(encoding="utf-8")
    if "v3/results.html" in html_text:
        return False
    pill = V3_PILL.format(href=href)
    new = re.sub(r'(<div id="corpus-toggle".*?)(</div>)', r"\1" + pill + r"\2",
                 html_text, count=1, flags=re.S)
    if new == html_text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def build() -> Path:
    feats, pieces = _load()
    bach = _bach_rows()
    body = (
        "<h1>v3 corpus — objective features <span class='sub'>42 arms × 3 prompts × 20 samples</span></h1>"
        "<p class='scope'>21 models (±thinking pairs) from five labs, released 2025-07 – "
        "2026-08. Each arm wrote 60 pieces per medium (3 self-expression variants × 20 "
        "samples): ABC notation and toolkit-free music21 code. The variant sentence is the "
        "system prompt; the user prompt is a fixed frame. Generated 2026-08-19/20; "
        "features v3. Four questions organize this page: are LLMs simpler than humans, "
        "does capability buy complexity, what does each model prefer, and do families "
        "share traits.</p>"
        + _completion_section(pieces)
        + _q1_bach_section(feats, bach)
        + _q2_ranking_section(feats)
        + _thinking_section(feats)
        + _q3_pref_section(feats, pieces)
        + _q4_family_section(feats)
        + _appendix(feats)
    )
    # House style for this page: no explanatory blurbs under section titles —
    # definitions live in the column tooltips instead.
    body = re.sub(r"<figcaption>.*?</figcaption>", "", body, flags=re.S)
    html_text = page("v3 corpus — objective features", "results.html", body,
                     extra_css=KEY_WIDGET_CSS)
    # subdir page: nav + asset links go up one level; then add the corpus pill.
    html_text = re.sub(r'(<a href=")(?!https?|\.\./|#)', r"\1../", html_text)
    html_text = re.sub(r'(<link[^>]*href=")(?!https?|\.\./)', r"\1../", html_text)
    html_text = html_text.replace("</nav>", "</nav>" + _corpus_toggle_v3(), 1)
    out = DOCS_DIR / "v3" / "results.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    for p, href in ((DOCS_DIR / "results.html", "v3/results.html"),
                    (DOCS_DIR / "v2" / "results.html", "../v3/results.html")):
        _inject_pill(p, href)
    return out


if __name__ == "__main__":
    print(build())
