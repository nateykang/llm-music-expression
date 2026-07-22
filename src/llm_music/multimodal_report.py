"""Generate docs/multimodal.html — the description↔music (text vs score) tab.

Currently covers the en-route vs post-hoc description experiment: for every
piece, the composing call's own description is compared against a fresh
music-only description by the same model (scrubbed score, blind LLM ratings,
per-piece content contrast). Everything is computed at build time from
committed data: description_arms_summary.json (stats), valence_comparison.json
(per-piece ratings + text measures), description_contrast.json (unique-claim
extraction), and the batch data.json files (example texts).

Build: ``llm-music multimodal-report``
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from .config import REPO_ROOT
from .report_common import SHORT, details_section, fnote, page, table

DOCS = REPO_ROOT / "docs"
ANALYSIS = DOCS / "analysis"
RATERS = ["fable-5", "gpt-5.6-thinking"]
CAT_LABEL = {
    "intent_or_goal": "intent / goal", "process_narrative": "process narrative",
    "emotional_self_attribution": "emotional self-attribution",
    "evaluative_praise": "evaluative praise", "technical_structure": "technical structure",
    "programmatic_imagery": "programmatic imagery",
    "admission_of_limitation": "admission of limitation", "other": "other",
}


def _rater_mean(row, arm, text, scale):
    vals = [row[arm][r][f"{text}.{scale}"] for r in RATERS
            if r in row[arm] and f"{text}.{scale}" in row[arm][r]]
    return sum(vals) / len(vals) if vals else None


def _arm_means(rows, text, scale):
    er = [v for r in rows if (v := _rater_mean(r, "enroute", text, scale)) is not None]
    ph = [v for r in rows if (v := _rater_mean(r, "posthoc", text, scale)) is not None]
    return sum(er) / len(er), sum(ph) / len(ph)


def _texts_by_key():
    """(model, mode, sample, batch) -> (enroute short, posthoc short)."""
    out = {}
    rows = json.loads((ANALYSIS / "valence_comparison.json").read_text())
    for batch in sorted({r["batch"] for r in rows}):
        data = json.loads((DOCS / "data" / batch / "data.json").read_text())
        for p in data["pieces"]:
            ind = p.get("independent_description")
            if p.get("ok") and ind and ind.get("short_description"):
                out[(p["model"], p["mode"], p.get("sample", 0), batch)] = (
                    p.get("short_description", ""), ind["short_description"])
    return out


def _p(v):
    return f"{v:.0e}" if v is not None else "—"


def _heatcell(v, scale=0.55):
    """Heat-shaded cell CONTENT (a span, not a <td> — table() adds the td)."""
    if v is None:
        return "—"
    a = min(0.5, abs(v) / scale * 0.5)
    rgb = "46,160,67" if v >= 0 else "207,90,80"
    return (f"<span style='background:rgba({rgb},{a:.2f});display:inline-block;"
            f"min-width:3.2em;padding:1px 6px;border-radius:4px'>{v:+.2f}</span>")


def render_multimodal_html() -> str:
    summary = json.loads((ANALYSIS / "description_arms_summary.json").read_text())
    rows = json.loads((ANALYSIS / "valence_comparison.json").read_text())
    contrast = json.loads((ANALYSIS / "description_contrast.json").read_text())
    n = summary["n_pieces"]

    # ---- headline table ----
    scale_meta = [("evaluative_positivity", "evaluative positivity", "1–7"),
                  ("weakness_admission", "weakness admission", "1–5"),
                  ("affect_valence", "affect valence", "1–9")]
    head_rows = []
    for key, label, rng in scale_meta:
        for text in ("short", "long"):
            st = summary["scales"][key][text]
            er, ph = _arm_means(rows, text, key)
            adj = st.get("length_adjusted_mean_delta")
            head_rows.append([
                f"{label} <span class='sd'>{rng} · {text}</span>",
                f"{er:.2f}", f"{ph:.2f}",
                f"{st['overall']['mean_delta']:+.2f}",
                f"{adj:+.2f}" if adj is not None else "—",
                _p(st["overall"].get("wilcoxon_p"))])
    headline = table(
        [("scale", None), ("en route", "mean over pieces, both raters averaged"),
         ("post hoc", None), ("Δ (er−ph)", "within-piece paired difference"),
         ("Δ len-adj", "regression intercept at equal text lengths (long only)"),
         ("p", "two-sided Wilcoxon signed-rank on the paired deltas")],
        head_rows)

    # ---- per-composer table ----
    def mean_delta(rs, text, scale):
        ds = [a - b for r in rs
              if (a := _rater_mean(r, "enroute", text, scale)) is not None
              and (b := _rater_mean(r, "posthoc", text, scale)) is not None]
        return sum(ds) / len(ds) if ds else None

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    model_rows = []
    for m, rs in sorted(by_model.items()):
        dw = [r["enroute"]["measures"]["long.words"] - r["posthoc"]["measures"]["long.words"]
              for r in rs]
        model_rows.append([
            SHORT.get(m, m), str(len(rs)),
            _heatcell(mean_delta(rs, "long", "evaluative_positivity")),
            _heatcell(mean_delta(rs, "short", "evaluative_positivity")),
            _heatcell(mean_delta(rs, "long", "affect_valence"), scale=1.2),
            _heatcell(mean_delta(rs, "short", "affect_valence"), scale=1.2),
            f"{sum(dw) / len(dw):+.0f}"])
    per_model = table(
        [("composer", None), ("n", "pieces with both description arms"),
         ("Δ eval. long", "evaluative positivity, en route − post hoc, long text; "
          "negative = the post-hoc text is rated more positively about the piece"),
         ("Δ eval. short", "same, on the one-sentence short text (length-matched "
          "by construction)"),
         ("Δ affect long", "affect valence, en route − post hoc, long text; positive "
          "= en route describes the music's emotional content as more positive"),
         ("Δ affect short", "same, short text"),
         ("Δ words", "long text, en route − post hoc; negative = post hoc is longer")],
        model_rows)

    # ---- mode strata ----
    evl = summary["scales"]["evaluative_positivity"]["long"]["by_mode"]
    evs = summary["scales"]["evaluative_positivity"]["short"]["by_mode"]
    avl = summary["scales"]["affect_valence"]["long"]["by_mode"]
    avs = summary["scales"]["affect_valence"]["short"]["by_mode"]
    strata = table(
        [("mode", None), ("n", None),
         ("Δ eval. long", None), ("p", "Wilcoxon, long-text delta"),
         ("Δ eval. short", None),
         ("Δ affect long", None), ("p ", "Wilcoxon, long-text delta"),
         ("Δ affect short", None)],
        [[m, str(evl[m]["n"]),
          _heatcell(evl[m]["mean_delta"]), _p(evl[m].get("wilcoxon_p")),
          _heatcell(evs[m]["mean_delta"]),
          _heatcell(avl[m]["mean_delta"], scale=1.2), _p(avl[m].get("wilcoxon_p")),
          _heatcell(avs[m]["mean_delta"], scale=1.2)]
         for m in evl])

    # ---- contrast categories ----
    cat_rows = []
    er_c = summary["contrast_per_piece"]["only_enroute"]
    ph_c = summary["contrast_per_piece"]["only_posthoc"]
    for cat in sorted(CAT_LABEL, key=lambda c: -(er_c.get(c, 0) + ph_c.get(c, 0))):
        cat_rows.append([CAT_LABEL[cat],
                         f"{er_c.get(cat, 0):.2f}", f"{ph_c.get(cat, 0):.2f}"])
    cat_table = table(
        [("category", None),
         ("en route only", "mean statements per piece that appear in the en-route "
          "text but not the post-hoc text"),
         ("post hoc only", "mean statements per piece unique to the post-hoc text")],
        cat_rows)

    ex_items = {"only_enroute": [], "only_posthoc": []}
    for p in sorted(contrast["pieces"], key=lambda x: x["id"]):
        for side in ex_items:
            for it in p[side]:
                if it["category"] in ("intent_or_goal", "emotional_self_attribution") \
                        and side == "only_enroute" and len(ex_items[side]) < 3:
                    ex_items[side].append((p["model"], it))
                if it["category"] == "technical_structure" and side == "only_posthoc" \
                        and len(ex_items[side]) < 3:
                    ex_items[side].append((p["model"], it))
    ex_html = ""
    for side, label in (("only_enroute", "Only en route says"),
                        ("only_posthoc", "Only post hoc says")):
        lis = "".join(
            f"<li><b>{SHORT.get(m, m)}</b> <span class='sd'>"
            f"[{CAT_LABEL[it['category']]}]</span> {html.escape(it['statement'])}</li>"
            for m, it in ex_items[side])
        ex_html += f"<p><b>{label}:</b></p><ul>{lis}</ul>"

    # ---- example pairs (chosen by rule, not by hand) ----
    texts = _texts_by_key()
    keyed = [(r, texts.get((r["model"], r["mode"], r["sample"], r["batch"]))) for r in rows]
    keyed = [(r, t) for r, t in keyed if t]

    def delta(r, scale):
        a = _rater_mean(r, "enroute", "short", scale)
        b = _rater_mean(r, "posthoc", "short", scale)
        return a - b if a is not None and b is not None else 0.0

    rosiest = max(keyed, key=lambda rt: delta(rt[0], "affect_valence"))
    praised = min(keyed, key=lambda rt: delta(rt[0], "evaluative_positivity"))
    examples = ""
    for (r, (er_t, ph_t)), why in (
            (rosiest, "largest en-route affect premium in the corpus"),
            (praised, "largest post-hoc praise premium in the corpus")):
        examples += (
            f"<p><b>{SHORT.get(r['model'], r['model'])} × {r['mode']}</b> "
            f"<span class='sd'>({why})</span><br>"
            f"<i>en route:</i> {html.escape(er_t)}<br>"
            f"<i>post hoc:</i> {html.escape(ph_t)}</p>")

    # ---- style / length table ----
    det = summary["deterministic"]
    style_rows = []
    for key, label in (("words", "length (words)"),
                       ("first_person_per_100w", "first-person per 100 words"),
                       ("vader", f"VADER compound{fnote('vader')}")):
        for text in ("short", "long"):
            d = det[f"{text}.{key}"]
            style_rows.append([f"{label} <span class='sd'>{text}</span>",
                               f"{d['enroute_mean']:.2f}", f"{d['posthoc_mean']:.2f}",
                               _p(d.get("wilcoxon_p"))])
    style = table([("measure", None), ("en route", None), ("post hoc", None),
                   ("p", None)], style_rows)

    agree = summary["rater_agreement_r"]
    ev_s = summary["scales"]["evaluative_positivity"]["short"]["overall"]
    av_s = summary["scales"]["affect_valence"]["short"]["overall"]
    wa = summary["scales"]["weakness_admission"]["short"]
    floor = wa.get("floor_rate", {})

    body = f"""
<h1>Multimodal <span class="sub">what models say about their music vs what the score says</span></h1>
<p class="scope">En-route vs post-hoc descriptions, {n} pieces × 13 composers. Each piece has
two descriptions by the <i>same model</i>: the one written in the same response as the music
(<b>en route</b>) and a fresh, stateless call given only the finished score, <i>anonymously</i>
(<b>post hoc</b>) — the describing call is never told the piece is its own, and titles,
comments, and credits are scrubbed so nothing of the composing call's framing leaks through.
Both texts were rated blind by {" and ".join(RATERS)} on three anchored scales, and a per-piece
contrast pass lists what each text says that the other doesn't.</p>

<h2>Summary: how the two descriptions differ</h2>
{headline}
<p class="scope">The scales, in plain terms. <b>Evaluative positivity</b> (1–7): how positively
the text talks about the piece's <i>quality or craft</i> — 1 = openly critical, 4 = describes
without evaluating, 7 = effusive praise. <b>Weakness admission</b> (1–5): does the text
acknowledge flaws, limitations, or unrealized ambitions — 1 = none at all. <b>Affect
valence</b>{fnote('russell')} (1–9): how positive the <i>emotions the text says the music
expresses</i> — grief and dread low, joy and serenity high, 5 = neutral or ambivalent —
independent of any claims about quality. Length adjustment (Δ len-adj): the per-piece delta is
regressed on the log ratio of the two texts' word counts; the reported value is the intercept,
i.e. the expected delta for a pair of equal-length texts. The one-sentence short descriptions
are length-matched by construction, so they carry no adjusted column.</p>
<p class="callout">Two opposite effects, cleanly separated by construct. <b>Post hoc is rated
<i>more</i> evaluatively positive</b> (Δ = {ev_s['mean_delta']:+.2f} on the length-matched short
text, p ≈ {_p(ev_s['wilcoxon_p'])}): describing a finished artifact invites appraisal language,
while en route spends its words on intent. But <b>en route paints the <i>emotional content</i>
rosier</b> (Δ = {av_s['mean_delta']:+.2f}, p ≈ {_p(av_s['wilcoxon_p'])}): the same notes get a
sunnier story from the model that just wrote them. Weakness admission is floored in
<i>both</i> arms ({floor.get('enroute', 0):.0%} vs {floor.get('posthoc', 0):.0%} of short texts
at 1/5) — models don't note flaws even in music they don't know they wrote.</p>

<h2>Per composer</h2>
{per_model}
<p class="scope">The praise inversion is driven by gpt-5.5, gpt-4.1, and the opus pair; grok,
fable, and llama tilt slightly the other way. The en-route affect premium is near-universal.
n varies because it counts successfully generated pieces: every free-form composer wrote 30/30
in abc, but codegen and sparse-toolkit generation failed at different rates per model
(gemini 23/30 codegen and no surviving sparse pieces; gpt-4.1 only 5/15 sparse), fable and
kimi compose only in the 3×5 sparse batches, and one gemini abc piece failed post-hoc
generation.</p>

<h2>Mode strata</h2>
{strata}
<p class="scope">abc pieces were described from their (scrubbed) ABC, codegen pieces from the
rendered MusicXML{fnote('music21')}. codegen-sparse is the toolkit-ablation corpus — generated
under a manipulated toolkit doc, so it is reported as its own stratum and never pooled. The
effects agree in direction and magnitude across all three.</p>

<h2>What each arm uniquely says</h2>
{cat_table}
{ex_html}
<p class="scope">The asymmetry in one line: <b>en route narrates a self; post hoc reads a
score.</b> En-route-only content is intent, imagery, and emotional self-attribution; post-hoc
descriptions contribute ~11 unique technical statements per piece. The ~4 technical claims per
piece unique to en route are candidate <i>phantom claims</i> — asserted by the composer but not
recovered by an independent reading of the score.</p>

<h2>Style and length</h2>
{style}
<p class="scope">Post hoc runs longer — the opposite of what "it just describes what's there"
would predict — and drops first person almost entirely (the styles are so distinct that true
rater blinding to arm is impossible; the deterministic measures above don't care). VADER agrees
with the raters on direction.</p>

{details_section("Methods fine print", f'''
<p>Ratings: one call per (piece, arm, rater) covering short and long text together;
reason-before-score{fnote("geval")}; scales anchored (evaluative positivity 1–7 with 4 = purely
descriptive; weakness admission 1–5; affect valence 1–9{fnote("russell")} with 5 = neutral).
Raters saw the text only — no model name, title, or arm label. Rater agreement (Pearson):
evaluative positivity {agree["evaluative_positivity"]:.2f}, weakness admission
{agree["weakness_admission"]:.2f}, affect valence {agree["affect_valence"]:.2f}.
fable-5 and gpt-5.6-thinking judge only in the free-form corpus; fable-5 composes 15
sparse-corpus pieces and rates its own text there (gpt-5.6-thinking, which composes nowhere,
covers the bias check). Contrast pass: both arms shown as anonymous texts A/B (assignment
randomized per piece), rater fable-5, fixed category taxonomy{fnote("llm-judge")}.</p>
<p>Length control: the short description is a single sentence in both arms by construction and
is the primary endpoint; long-text deltas are also reported at equal lengths via the intercept
of Δ regressed on log word-ratio. One of 809 pieces failed post-hoc generation (JSON parse) and
is excluded. Raw data: <a href="analysis/valence_comparison.json">valence_comparison.json</a>,
<a href="analysis/description_contrast.json">description_contrast.json</a>,
<a href="analysis/description_arms_summary.json">description_arms_summary.json</a>.</p>''')}
"""
    return page("Multimodal — LLM music self-expression", "multimodal.html", body)


def write_multimodal_report() -> Path:
    out = DOCS / "multimodal.html"
    out.write_text(render_multimodal_html(), encoding="utf-8")
    return out
