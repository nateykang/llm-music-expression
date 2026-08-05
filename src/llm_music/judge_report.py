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
from .report_common import (MODE_TOGGLE, SHORT, drop_degenerate, fnote, fnum,
                            group_by_model,
                            heat, mode_filter, page, paned, scorebar, table,
                            toggle)

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
            f"<b>r = {r:+.2f}</b> (Pearson correlation: −1…+1, where −1 = move in perfect "
            f"opposition and 0 = unrelated; n={len(pairs)} models) — a judge that never saw the "
            f"key hears minor-heavy models as darker. The computed proxy and human-style "
            f"perception agree.</p>")
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


def _topline(verdict):
    t = verdict.get("topline")
    return t.get("score") if isinstance(t, dict) else None



def _key_affinity(raw, feats, out_dir: Path | None = None):
    """Do judges favor pieces in the keys/modes they themselves produce?
    dev(J,i) = topline_J(i) - mean(other judges), J's own pieces excluded so
    the measure is orthogonal to self-bias."""
    keymap = {}
    for r in feats:
        tonic = r.get("key_declared_tonic") or r.get("key_tonic") or ""
        kmode = (r.get("key_declared_mode") or r.get("key_mode_best") or "").lower()
        keymap[(r["model"], str(r.get("sample") or 0))] = (tonic, kmode)
    devs = defaultdict(list)
    own_keys = defaultdict(list)
    for pc in raw:
        pk = (pc["model"], str(pc.get("sample") or 0))
        if pk not in keymap:
            continue
        tonic, kmode = keymap[pk]
        own_keys[pc["model"]].append((tonic, kmode))
        tl = {j: _topline(v) for j, v in pc["panel"].items()}
        tl = {j: s for j, s in tl.items() if s is not None}
        for j, s in tl.items():
            if j == pc["model"]:
                continue
            others = [x for k, x in tl.items() if k != j and k != pc["model"]]
            if len(others) >= 5:
                devs[j].append((s - mean(others), tonic, kmode))
    rows, xs, ys, labs = [], [], [], []
    k_xs, k_ys, k_labs = [], [], []
    for j in [m for m in SHORT if m in devs]:
        own = own_keys.get(j, [])
        pm = sum(1 for _, m in own if m == "minor") / len(own) if own else None
        dm = [d for d, _, m in devs[j] if m == "minor"]
        dj = [d for d, _, m in devs[j] if m == "major"]
        aff = (mean(dm) - mean(dj)) if dm and dj else None
        if own:
            modal = max(set(own), key=own.count)
            same = [d for d, t, m in devs[j] if (t, m) == modal]
            other = [d for d, t, m in devs[j] if (t, m) != modal]
            ka = (mean(same) - mean(other)) if len(same) >= 20 and other else None
            modal_lab = f"{modal[0]} {modal[1][:3]}" if modal[0] else "—"
        else:
            ka, modal_lab = None, "—"
        rows.append([SHORT.get(j, j),
                     f"{pm:.0%}" if pm is not None else "—",
                     f"{aff:+.2f}" if aff is not None else "—",
                     modal_lab,
                     f"{ka:+.2f}" if ka is not None else "—"])
        if aff is not None and pm is not None:
            xs.append(pm); ys.append(aff); labs.append(SHORT.get(j, j))
        if ka is not None and own:
            k_xs.append(own.count(modal) / len(own))
            k_ys.append(ka); k_labs.append(SHORT.get(j, j))
    r = _pearson(xs, ys) if len(xs) >= 3 else float("nan")
    cols = [("judge", None),
            ("own %minor", "share of this model's OWN pieces in a minor mode"),
            ("minor affinity", "how much this judge over-scores minor pieces relative to "
                               "the rest of the panel (dev on minor minus dev on major)"),
            ("own modal key", "the single key this model writes in most"),
            ("same-key affinity", "over-scoring of pieces in the judge's own modal key "
                                  "vs all other keys")]
    note = (f"<p class='callout'>Judges' minor-affinity tracks their own minor-production rate at "
            f"<b>r = {r:+.2f}</b> (n={len(xs)} judges): models tend to favor music written in the "
            f"modes they compose in themselves. Caveat: mode and author are not independent in this "
            f"corpus (major pieces come overwhelmingly from the majority-major authors), so taste "
            f"for an author's <i>style</i> can masquerade as mode affinity — a transposed/mode-"
            f"flipped control set is the clean follow-up.</p>")
    fig_html = ""
    if out_dir is not None and len(xs) >= 3:
        fig_html = _affinity_figure(
            [1 - m for m in xs], [-a for a in ys], labs, out_dir, "mode_bias_v2.png",
            xlabel="share of the judge's own free-form pieces in major keys",
            ylabel="bias towards pieces in major keys (topline pts vs panel)",
            title="LLMs' Bias in Generation Persisting in Evaluation — mode",
            caption="Each point is one judge: how much of its own music is in major (x) "
                    "against how much it over-scores major pieces relative to the rest of "
                    "the panel, own pieces excluded (y). The v1 counterpart "
                    "(self-preference tab) showed the same relationship on Bach chorales "
                    "and v1 pieces.")
    if out_dir is not None and len(k_xs) >= 3:
        fig_html += _affinity_figure(
            k_xs, k_ys, k_labs, out_dir, "key_bias_v2.png",
            xlabel="share of the judge's own pieces in its single most-written key",
            ylabel="bias towards pieces in that exact key (topline pts vs panel)",
            title="LLMs' Bias in Generation Persisting in Evaluation — specific key",
            caption="Same idea one level finer: each judge's concentration on its own "
                    "modal key (x) against how much it over-scores pieces written in "
                    "exactly that key, relative to the rest of the panel (y).")
    return table(cols, rows) + note + fig_html


def _affinity_figure(xs, ys, labels, out_dir: Path, name: str, *,
                     xlabel: str, ylabel: str, title: str, caption: str) -> str:
    """v2 counterparts of v1's mode_bias_combined.png (selfpref tab): a judge's
    generation-side habit (x) against its evaluation-side bias (y), one point
    per judge, with a permutation test on the correlation."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#26231f", "#6b6660"
    r0 = float(np.corrcoef(xs, ys)[0, 1])
    rng = np.random.default_rng(0)
    p = sum(abs(float(np.corrcoef(rng.permutation(np.asarray(xs, float)), ys)[0, 1])) >= abs(r0)
            for _ in range(20000)) / 20000

    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.scatter(xs, ys, s=42, color="#4a5a7a", label="LLM pieces (v2)")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=7, color=MUTED, xytext=(4, 3),
                    textcoords="offset points")
    z = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs), max(xs), 10)
    ax.plot(xr, z[0] * xr + z[1], color="#4a5a7a", ls=":", lw=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(0, color=MUTED, lw=.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_title(f"{title}  (r = {r0:+.2f}, perm p = {p:.3f})", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=160)
    plt.close(fig)
    return (f"<figure class='chart'><img src='analysis/{name}' alt='{xlabel} vs {ylabel}'>"
            f"<figcaption>{caption}</figcaption></figure>")

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
def render_judge_html(analysis_dir: Path, data_dir: Path, out_path: Path, *,
                      panel: list | None = None, variant: str = "v1") -> Path:
    """variant="v1" reproduces the historical page byte-identically (default
    3-frontier PANEL, v1 narrative). variant="v2" renders the full-matrix
    corpus: `panel` should then list every judge; v1-specific callouts are
    replaced with matrix-appropriate text."""
    panel = panel or PANEL
    feats = []
    for fp in sorted(data_dir.glob("*/features.csv")):
        feats += [r for r in csv.DictReader(fp.open(encoding="utf-8")) if r["prompt"] == "free-form"]
    rawp = analysis_dir / "judge_allmodels_raw.json"
    raw = drop_degenerate(
        [p for p in json.loads(rawp.read_text(encoding="utf-8"))
         if p["prompt"] == "free-form"],
        analysis_dir) if rawp.exists() else []
    secs = []
    if raw and variant == "v2":
        npanel = len(panel)
        secs.append(f"<h2>Which models write the best music <span class='sub'>(blind {npanel}-model panel)</span></h2>"
                    f"<p class='scope'>Every roster model judges every piece — a full {npanel}×{npanel} "
                    "author-judge matrix, blind: the judge sees stripped notation only (no title, composer "
                    "note, or model name), and each author's ranking below averages the OTHER models' "
                    "verdicts (self-verdicts are held out for the self-bias analyses). Dimensions follow the "
                    f"music-eval literature (ChatMusician{fnote('chatmusician')} / Chu et "
                    f"al.{fnote('chu')} / MuSpike{fnote('muspike')}); scoring follows the LLM-judge "
                    f"literature{fnote('llm-judge')} — reason-before-score{fnote('geval')}, anchored "
                    "1–5 scales, plus a top-line holistic score asked last.</p>"
                    + paned(lambda m: _rankings(_panel_rows(mode_filter(raw, m), panel))))
    elif raw:
        secs.append("<h2>Which models write the best music <span class='sub'>(blind panel)</span></h2>"
                    "<p class='scope'>A blind 3-frontier panel (gpt-5.5 · gemini · opus) rates each piece "
                    "from the notation alone — no title, composer note, or model name. Dimensions follow the "
                    f"music-eval literature (ChatMusician{fnote('chatmusician')} / Chu et "
                    f"al.{fnote('chu')} / MuSpike{fnote('muspike')}); scoring follows the LLM-judge "
                    f"literature{fnote('llm-judge')} — each judge writes a short justification before "
                    f"scoring{fnote('geval')}, uses 1–5 scales with described anchor points, and the "
                    "panel's scores are averaged.</p>"
                    + paned(lambda m: _rankings(_panel_rows(mode_filter(raw, m), PANEL)))
                    + "<p class='callout' style='font-size:.82rem'><b>Thinking improves the music — in "
                      "both representations.</b> The adaptive-thinking variants beat their base models in "
                      "every representation-matched comparison (blind 3-frontier panel): "
                      "<b>sonnet +0.28 ABC / +0.23 code-gen</b>, <b>opus +0.10 ABC / +0.47 code-gen</b>. "
                      "Every cell is positive — extended thinking reliably raises perceived quality.</p>"
                    + "<p class='scope' style='font-size:.8rem; margin-top:.6rem'><b>Caveat — generation "
                      "reliability varies by model and representation, so per-model n differs.</b> The "
                      "adaptive-thinking variants spend most of their token budget <i>thinking</i>: "
                      "sonnet-4.6-thinking needed a <b>64k-token cap</b> and a <b>30-min read-timeout</b> to "
                      "produce ABC at all — it reasons ~31k tokens before writing a single note, and fails "
                      "entirely under default limits. gemini is the weakest at code-gen (only 23/30 free-form "
                      "runs succeeded — its generated music21 code crashes). So these are real findings, but "
                      "they came from models that took substantial engineering to run.</p>")
    if raw:
        secs.append("<h2>Emotional character <span class='sub'>(perceived, blind)</span></h2>"
                    "<p class='scope'>What the blind judge <i>hears</i> — perceived valence (how positive "
                    "the music sounds, dark ↔ bright) and arousal (how energetic, calm ↔ excited), the two "
                    f"axes of the Russell circumplex model of emotion{fnote('russell')} — plus the "
                    "dominant emotion label, against the computed minor-key proxy.</p>"
                    + paned(lambda m: _emotion(_panel_rows(mode_filter(raw, m), panel), mode_filter(feats, m))))
        allx = f"all-{len(panel)}" if variant == "v2" else "all-9"
        conclusion = ("Self-judging is blind (pieces are stripped), so any own-piece favoritism "
                      "is implicit self-recognition, not label bias."
                      if variant == "v2" else
                      "No model meaningfully over-rates itself; competence and leniency vary widely.")
        secs.append(f"<h2>Can each model judge music? <span class='sub'>({allx} study)</span></h2>"
                    "<p class='scope'>With every model judging every piece, this shows each model's "
                    "competence (agreement with the consensus), its leniency, and — leniency-corrected — "
                    f"whether it favors its own work. {conclusion}</p>"
                    + paned(lambda m: _competence_selfbias(mode_filter(raw, m))[0]))
        trait_scope = (
            "<p class='scope'>Where each model judges its <i>own</i> music differently than it judges "
            "everyone else's, per trait. "
            "<span style='color:rgb(46,140,67)'>green = kinder to itself</span>, "
            "<span style='color:rgb(197,80,70)'>red = harder on itself</span>. "
            "Blind self-judging with n=100 own pieces per judge.</p>"
            if variant == "v2" else
            "<p class='scope'>Where each model judges its <i>own</i> music differently than it judges "
            "everyone else's. <span style='color:rgb(46,140,67)'>green = kinder to itself</span>, "
            "<span style='color:rgb(197,80,70)'>red = harder on itself</span>. The pattern: weak "
            "models over-credit themselves exactly where they're weakest (grok→harmony, llama→emotion); "
            "strong models are calibrated. Small n per model — read patterns, not single cells.</p>")
        trait_title = ("Self-Preference Bias By Trait" if variant == "v2"
                       else "Self-bias by trait")
        secs.append(f"<h2>{trait_title} <span class='sub'>(leniency-corrected)</span></h2>"
                    + trait_scope
                    + paned(lambda m: _per_trait(mode_filter(raw, m))))
        if variant == "v2":
            secs.append("<h2>Do judges favor their own keys? <span class='sub'>(key/mode affinity)</span></h2>"
                        "<p class='scope'>Each judge's over- or under-scoring of pieces by key and mode "
                        "(measured as deviation from the rest of the panel on the same pieces, own pieces "
                        "excluded) against the keys that judge itself composes in.</p>"
                        + _key_affinity(raw, feats, analysis_dir))
    # Text bias is only meaningful against a GOAL prompt (does the brief make the
    # judge over-credit adherence) — not free-form. _text_bias() is kept for that
    # steering-phase comparison; intentionally not shown on the free-form page.

    secs_html = "\n".join(secs) or "<p>No judge results found. Run <code>llm-music judge</code> first.</p>"
    body = f"""<h1>How LLMs judge music — and themselves</h1>
  <p class="scope">An LLM-as-judge layer — language models, given a fixed rubric, standing in for
     human raters{fnote("llm-judge")} — over the generated pieces: blind quality + emotion ratings,
     each model's competence as a critic, and its self-bias. Rubric dimensions follow the music-eval
     literature; the protocol (reason-before-score, anchored scales, blind panel) follows the LLM-judge
     literature. Scope: {len(raw)} free-form pieces. Generated by <code>llm-music judge-report</code>.</p>
  {toggle(MODE_TOGGLE)}
  {secs_html}"""
    out_path.write_text(page("LLM judge — musical inductive biases", "judge.html", body),
                        encoding="utf-8")
    return out_path
