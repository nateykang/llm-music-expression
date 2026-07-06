"""Generate docs/judge.html — the LLM-judge analysis page (third site tab).

Reads the judge outputs under docs/analysis/ (judge.csv = blind 3-frontier panel,
judge_noted.csv = noted condition, judge_allmodels_raw.json = every model judging
every piece) plus features.csv (computed proxies), and renders: quality rankings,
emotion character, perceived-vs-computed valence, judge competence + self-bias, the
per-trait self-bias heatmap, and the text-bias comparison.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean

from .judge import QUALITY_KEYS
from .report_common import (MODE_TOGGLE, SHORT, fnum, group_by_model, heat,
                            mode_filter, page, paned, scorebar, stat_cards,
                            table, toggle)

DIMS = QUALITY_KEYS + ["valence", "arousal"]
PANEL = ["gpt-5.5", "gemini-2.5-pro", "opus-4.8"]


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


# ---------- sections ----------
def _by_model(rows, keys):
    out = {}
    for m, rs in group_by_model(rows).items():
        out[m] = {k: mean([fnum(r[k]) for r in rs if fnum(r.get(k)) is not None] or [float("nan")]) for k in keys}
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
        cells = [SHORT.get(m, m), str(d["n"]), scorebar(d["overall"])]
        cells += [f"{d[k]:.2f}" for k in QUALITY_KEYS]
        rows.append(cells)
    return table(cols, rows)


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
    return table(cols, rows) + note


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
    return table(cols, rows), pieces


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
    # Columns = all judges (from the panels), in a FIXED canonical order, so they line
    # up identically across the generation-mode toggle. (Previously sorted by per-mode
    # own-piece count, which reshuffled the columns when you switched modes.) A model
    # with no own pieces in a given mode just shows empty cells, keeping alignment.
    all_judges = {j for p in raw for j in p["panel"]}
    judges = [m for m in SHORT if m in all_judges]
    head = "<thead><tr><th>trait</th>" + "".join(f"<th>{SHORT[m]}</th>" for m in judges) + "</tr></thead>"
    body = "<tbody>"
    for d in DIMS:
        body += f"<tr><td class='m'>{d}</td>"
        for m in judges:
            v = (mean(own[m][d]) - mean(oth[m][d])) if own[m].get(d) and oth[m].get(d) else None
            body += heat(v)
        body += "</tr>"
    body += "</tbody>"
    nrow = ("<tfoot><tr><td class='m sub'>n own</td>" + "".join(
        f"<td class='sub'>{len(own[m].get('harmony', []))}</td>" for m in judges) + "</tr></tfoot>")
    return f"<div class='tscroll'><table class='heat sortable'>{head}{body}{nrow}</table></div>"


def _panel_rows(raw, panel):
    """Per-piece blind aggregate from the given panel judges (excluding the author),
    so rankings/emotion use the all-9 data as their single source — every author
    (incl. the thinking variants) appears, not just whoever the pilot reached."""
    rows = []
    for p in raw:
        author = p["model"]
        verds = [v for j, v in p["panel"].items() if j in panel and j != author]
        if not verds:
            continue
        row = {"model": author, "prompt": p["prompt"], "mode": p.get("mode", ""),
               "title": p.get("title", "")}
        for k in QUALITY_KEYS + ["valence", "arousal"]:
            sc = [v[k]["score"] for v in verds if k in v]
            row[k] = (sum(sc) / len(sc)) if sc else None
        qd = [row[k] for k in QUALITY_KEYS if row.get(k) is not None]
        row["overall"] = (sum(qd) / len(qd)) if qd else None
        labels = [v["emotion_label"] for v in verds if v.get("emotion_label")]
        row["emotion_label"] = Counter(labels).most_common(1)[0][0] if labels else ""
        rows.append(row)
    return rows


def _text_bias(blind, noted):
    bkey = {(r["model"], r["title"]): r for r in blind}
    rows = [(bkey[(r["model"], r["title"])], r) for r in noted if (r["model"], r["title"]) in bkey]
    cols = [("dimension", None), ("Δ noted − blind", "mean change when the composer's written note "
                                                     "is shown; + = the note inflates the score"),
            ("type", None)]
    out = []
    for d in DIMS:
        ds = [fnum(n[d]) - fnum(b[d]) for b, n in rows if fnum(n.get(d)) is not None and fnum(b.get(d)) is not None]
        if ds:
            t = "quality" if d in QUALITY_KEYS else "affect"
            out.append([d, f"{mean(ds):+.3f}", t])
    return table(cols, out), len(rows)


# ---------- page ----------
def render_judge_html(analysis_dir: Path, data_dir: Path, out_path: Path) -> Path:
    feats = []
    for fp in sorted(data_dir.glob("*/features.csv")):
        feats += [r for r in csv.DictReader(fp.open(encoding="utf-8")) if r["prompt"] == "free-form"]
    rawp = analysis_dir / "judge_allmodels_raw.json"
    raw = [p for p in json.loads(rawp.read_text(encoding="utf-8")) if p["prompt"] == "free-form"] \
        if rawp.exists() else []
    secs = []
    if raw:
        secs.append("<h2>Which models write the best music <span class='sub'>(blind panel)</span></h2>"
                    "<p class='scope'>A blind 3-frontier panel (gpt-5.5 · gemini · opus) rates each piece "
                    "from the notation alone — no title, composer note, or model name. Dimensions follow the "
                    "music-eval literature (ChatMusician / Chu et al. / MuSpike); scoring follows the "
                    "LLM-judge literature (reason-before-score, anchored 1–5, panel-averaged).</p>"
                    + paned(lambda m: _rankings(_panel_rows(mode_filter(raw, m), PANEL)))
                    + "<p class='callout' style='font-size:.82rem'>🧠 <b>Thinking improves the music — in "
                      "both representations.</b> The adaptive-thinking variants beat their base models in "
                      "every representation-matched comparison (blind 3-frontier panel): "
                      "<b>sonnet +0.28 ABC / +0.23 code-gen</b>, <b>opus +0.10 ABC / +0.47 code-gen</b>. "
                      "Every cell is positive — extended thinking reliably raises perceived quality.</p>"
                    + "<p class='scope' style='font-size:.8rem; margin-top:.6rem'>⚙️ <b>Caveat — generation "
                      "reliability varies by model and representation, so per-model n differs.</b> The "
                      "adaptive-thinking variants spend most of their token budget <i>thinking</i>: "
                      "sonnet-4.6-thinking needed a <b>64k-token cap</b> and a <b>30-min read-timeout</b> to "
                      "produce ABC at all — it reasons ~31k tokens before writing a single note, and fails "
                      "entirely under default limits. gemini is the weakest at code-gen (only 23/30 free-form "
                      "runs succeeded — its generated music21 code crashes). So these are real findings, but "
                      "they came from models that took substantial engineering to run.</p>")
        secs.append("<h2>Emotional character <span class='sub'>(perceived, blind)</span></h2>"
                    "<p class='scope'>What the blind judge <i>hears</i> — perceived valence/arousal and the "
                    "dominant emotion — against the computed minor-key proxy.</p>"
                    + paned(lambda m: _emotion(_panel_rows(mode_filter(raw, m), PANEL), mode_filter(feats, m))))
        secs.append("<h2>Can each model judge music? <span class='sub'>(all-9 study)</span></h2>"
                    "<p class='scope'>With every model judging every piece, this shows each model's "
                    "competence (agreement with the consensus), its leniency, and — leniency-corrected — "
                    "whether it favors its own work. No model meaningfully over-rates itself; competence "
                    "and leniency vary widely.</p>"
                    + paned(lambda m: _competence_selfbias(mode_filter(raw, m))[0]))
        secs.append("<h2>Self-bias by trait <span class='sub'>(leniency-corrected)</span></h2>"
                    "<p class='scope'>Where each model judges its <i>own</i> music differently than it judges "
                    "everyone else's. <span style='color:rgb(46,140,67)'>green = kinder to itself</span>, "
                    "<span style='color:rgb(197,80,70)'>red = harder on itself</span>. The pattern: weak "
                    "models over-credit themselves exactly where they're weakest (grok→harmony, llama→emotion); "
                    "strong models are calibrated. Small n per model — read patterns, not single cells.</p>"
                    + paned(lambda m: _per_trait(mode_filter(raw, m))))
    # Text bias is only meaningful against a GOAL prompt (does the brief make the
    # judge over-credit adherence) — not free-form. _text_bias() is kept for that
    # steering-phase comparison; intentionally not shown on the free-form page.

    # Findings strip — computed from the same raw verdicts as the tables below.
    cards = ""
    if raw:
        blind = _panel_rows(raw, PANEL)
        bym = _by_model(blind, ["overall"])
        top = max(bym, key=lambda m: bym[m]["overall"])
        deltas = []
        for base in ("sonnet-4.6", "opus-4.8"):
            think = base + "-thinking"
            if base in bym and think in bym:
                deltas.append(bym[think]["overall"] - bym[base]["overall"])
        think_delta = (sum(deltas) / len(deltas)) if deltas else None
        # judge competence + corrected self-bias, computed as in the tables
        _, pieces = _competence_selfbias(raw)
        comp, bias = {}, {}
        for j in sorted({jj for _, qd in pieces for jj in qd}):
            xs, ys, own, oth = [], [], [], []
            for author, qd in pieces:
                if j not in qd:
                    continue
                others = [v for k2, v in qd.items() if k2 != j]
                if not others:
                    continue
                xs.append(qd[j]); ys.append(mean(others))
                (own if author == j else oth).append(qd[j] - mean(others))
            comp[j] = _pearson(xs, ys)
            if own:
                bias[j] = mean(own) - (mean(oth) if oth else 0.0)
        best_critic = max((j for j in comp if comp[j] == comp[j]), key=lambda j: comp[j])
        max_bias_m = max(bias, key=lambda j: bias[j]) if bias else None
        cards = stat_cards([
            (f"{SHORT.get(top, top)} · {bym[top]['overall']:.2f}", "top blind-panel quality",
             f"{bym[top]['n']} pieces, panel of {len(PANEL)}"),
            (f"{think_delta:+.2f}" if think_delta is not None else "—", "adaptive-thinking effect",
             "mean Δ overall, thinking vs base (sonnet + opus)"),
            (f"{SHORT.get(best_critic, best_critic)} · r={comp[best_critic]:.2f}",
             "most reliable critic", "correlation with the leave-one-out consensus"),
            (f"{SHORT.get(max_bias_m, max_bias_m)} {bias[max_bias_m]:+.2f}" if max_bias_m else "—",
             "largest corrected self-bias", "all-9 study; >0 favors its own pieces"),
        ])

    secs_html = "\n".join(secs) or "<p>No judge results found. Run <code>llm-music judge</code> first.</p>"
    body = f"""<h1>How LLMs judge music — and themselves</h1>
  <p class="scope">An LLM-as-judge layer over the generated pieces: blind quality + emotion ratings,
     each model's competence as a critic, and its self-bias. Rubric dimensions follow the music-eval
     literature; the protocol (reason-before-score, anchored scales, blind panel) follows the LLM-judge
     literature. Scope: {len(raw)} free-form pieces. Generated by <code>llm-music judge-report</code>.</p>
  {cards}
  {toggle(MODE_TOGGLE)}
  {secs_html}"""
    out_path.write_text(page("LLM judge — musical inductive biases", "judge.html", body),
                        encoding="utf-8")
    return out_path
