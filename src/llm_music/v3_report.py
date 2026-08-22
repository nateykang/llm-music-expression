"""v3 corpus results page: objective-feature EDA over the 42-arm corpus.

Builds docs/v3/results.html from the three v3 batches' features.csv +
data.json, and extends the corpus pill on the v1/v2 results pages with a v3
link. Follows the site's research-notebook conventions: prose figcaptions,
sortable tables, no summary widgets.
"""

from __future__ import annotations

import csv
import re
import statistics as st
from collections import Counter
from pathlib import Path

from .config import DOCS_DIR
from .report_common import cell, heat, page, paned, table, toggle

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

COMPLEXITY = ["pitch_class_entropy", "rhythm_entropy", "polyphony", "structureness",
              "n_pitches_used", "pitch_range"]
HARMONY = ["consonance_rate", "chord_tone_rate", "scale_consistency"]


def _num(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _med(rows, col):
    vals = [x for r in rows if (x := _num(r.get(col))) is not None]
    return st.median(vals) if vals else None


def _base(arm):
    return arm[:-9] if arm.endswith("-thinking") else arm


def _load():
    feats, pieces = [], []
    for b in V3_BATCHES:
        with (DOCS_DIR / "data" / b / "features.csv").open(encoding="utf-8", newline="") as f:
            feats.extend(csv.DictReader(f))
        import json
        pieces.extend(json.loads((DOCS_DIR / "data" / b / "data.json")
                                 .read_text(encoding="utf-8"))["pieces"])
    return feats, pieces


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


FEAT_COLS = [
    ("n", "Pieces aggregated in this row."),
    ("minor", "Share of pieces in a minor key — the model's DECLARED key (ABC K:) "
              "where available, else music21's Krumhansl–Schmuckler detection."),
    ("len s", "Median rendered length in seconds."),
    ("notes/beat", "Median rhythmic density: notes per quarter-note beat (tempo-invariant)."),
    ("pc entropy", "Pitch-class entropy (0–3.58 bits): how evenly the twelve pitch classes "
                   "are used. Higher = more chromatic/varied pitch material."),
    ("rhy entropy", "Entropy of note-duration classes. Higher = more varied rhythms."),
    ("polyphony", "Mean number of simultaneous notes."),
    ("structure", "Structureness: repetition of material across the piece (self-similarity)."),
    ("consonance", "Share of simultaneous intervals that are consonant."),
    ("in-scale", "Share of notes inside the detected key's scale."),
    ("instr", "Median number of distinct instruments."),
    ("dyn Δ", "Median count of written dynamic changes (marks + hairpins)."),
]


def _arm_row(arm, rows):
    return [
        f"<td class='m'>{arm}</td>",
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


def _completion_section(pieces):
    ok = Counter()
    tot = Counter()
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
    return (
        "<h2>Completion: who can write music in each medium?</h2>"
        "<figure><figcaption>Every model completed essentially all ABC pieces, while "
        "toolkit-free code-gen completion climbs steadily with release date and with "
        "reasoning — every code-gen failure is a hallucinated music21 API call, so this "
        "column measures closed-book library recall, not musicality. Cells are ok/attempted "
        "after up to 5 error-fed retries; red intensity marks the shortfall.</figcaption>"
        + table(cols, rows, left=3) + "</figure>")


def _per_arm_section(feats):
    def pane(mode_key):
        gm = {"abc": ("abc", "smt-abc"), "code": ("codegen",), "all": ("abc", "smt-abc", "codegen")}[mode_key]
        sub = [r for r in feats if r.get("mode") in gm]
        rows = []
        for b in BASES:
            for arm in (b, b + "-thinking"):
                mine = [r for r in sub if r["model"] == arm]
                if mine:
                    rows.append(_arm_row(arm, mine))
        return table([("arm", None)] + FEAT_COLS, rows)
    return (
        "<h2>Objective features by arm <span class='sub'>(medians)</span></h2>"
        + toggle(default="abc")
        + "<figure><figcaption>Each row is one model arm across all three prompt variants "
        "(up to 60 pieces). Sort any column; the tooltips define each feature.</figcaption>"
        + paned(pane, default="abc") + "</figure>")


def _thinking_section(feats):
    sub = [r for r in feats if r.get("mode") in ("abc", "smt-abc")]
    cols = [("model", None), ("len s", None), ("notes/beat", None),
            ("pc entropy", None), ("rhy entropy", None), ("structure", None), ("instr", None)]
    rows = []
    for b in BASES:
        t = [r for r in sub if r["model"] == b + "-thinking"]
        n = [r for r in sub if r["model"] == b]
        if not (t and n):
            continue
        deltas = [
            (_med(t, "length_seconds") or 0) - (_med(n, "length_seconds") or 0),
            (_med(t, "note_density") or 0) - (_med(n, "note_density") or 0),
            (_med(t, "pitch_class_entropy") or 0) - (_med(n, "pitch_class_entropy") or 0),
            (_med(t, "rhythm_entropy") or 0) - (_med(n, "rhythm_entropy") or 0),
            (_med(t, "structureness") or 0) - (_med(n, "structureness") or 0),
            (_med(t, "n_instruments") or 0) - (_med(n, "n_instruments") or 0),
        ]
        scale = [30, 0.8, 0.25, 0.25, 0.1, 1.5]
        rows.append([f"<td class='m'>{b}</td>"] + [
            heat(d, s).replace(">+0.00<", ">0<") for d, s in zip(deltas, scale)])
    return (
        "<h2>Does thinking change the music? <span class='sub'>(thinking − non-thinking, ABC)</span></h2>"
        "<figure><figcaption>Median differences between each model's thinking and "
        "non-thinking arms on the same prompts. Green = the thinking arm is higher. "
        "For always-thinking models (fable-5, kimi-k3, gemini pro/3.5+ flash, grok-4.6) "
        "the pair is effort low vs high rather than off vs on.</figcaption>"
        + table([c for c in [("model", None), ("len s", None), ("notes/beat", None),
                             ("pc entropy", None), ("rhy entropy", None),
                             ("structure", None), ("instr", None)]],
                rows) + "</figure>")


def _family_section(feats):
    sub = [r for r in feats if r.get("mode") in ("abc", "smt-abc")]
    fams = sorted({RELEASE[b][0] for b in BASES})
    rows = []
    for fam in fams:
        mine = [r for r in sub if RELEASE.get(_base(r["model"]), ("?",))[0] == fam]
        rows.append(_arm_row(fam, mine))
    return (
        "<h2>Family signatures <span class='sub'>(ABC, medians)</span></h2>"
        "<figure><figcaption>All arms of a lab pooled. Differences here are "
        "family-level style traits; per-arm variation is in the table above.</figcaption>"
        + table([("family", None)] + FEAT_COLS, rows) + "</figure>")


def _trend_section(feats):
    sub = [r for r in feats if r.get("mode") in ("abc", "smt-abc")]
    order = {b: i for i, b in enumerate(sorted(BASES, key=lambda b: RELEASE[b][1]))}
    rows = []
    for feat in COMPLEXITY + HARMONY:
        xs, ys = [], []
        for b in BASES:
            mine = [r for r in sub if _base(r["model"]) == b]
            if mine:
                xs.append(order[b])
                ys.append(_med(mine, feat))
        rho = _spearman(xs, ys)
        rows.append([f"<td class='m'>{feat}</td>",
                     heat(rho, 1.0) if rho is not None else "<td>—</td>"])
    return (
        "<h2>Do newer models write more complex music? <span class='sub'>(release rank vs feature, Spearman)</span></h2>"
        "<figure><figcaption>Correlation between a model's release date (rank across the "
        "21 base models) and its median feature value, ABC pieces. Exploratory: n=21 "
        "models, so |ρ| below ~0.45 is within noise.</figcaption>"
        + table([("feature", None), ("ρ", None)], rows) + "</figure>")


def _keys_section(feats):
    sub = [r for r in feats if r.get("mode") in ("abc", "smt-abc")]
    rows = []
    for b in BASES:
        mine = [r for r in sub if _base(r["model"]) == b]
        if not mine:
            continue
        keys = Counter()
        for r in mine:
            tonic = r.get("key_declared_tonic") or r.get("key_tonic")
            mode = _declared_mode(r)
            if tonic and mode:
                keys[f"{tonic} {mode}"] += 1
        top = ", ".join(f"{k} {100*v/len(mine):.0f}%" for k, v in keys.most_common(3))
        rows.append([f"<td class='m'>{b}</td>",
                     cell((_minor_pct(mine) or 0) / 100, "pct"),
                     f"<td class='m'>{top}</td>"])
    return (
        "<h2>Key &amp; mode signatures <span class='sub'>(ABC, declared keys)</span></h2>"
        "<figure><figcaption>What each model reaches for tonally when asked only to "
        "express itself. Both arms pooled per model.</figcaption>"
        + table([("model", None), ("minor", None), ("top keys", None)], rows, left=1)
        + "</figure>")


def _prompt_section(feats):
    sub = [r for r in feats if r.get("mode") in ("abc", "smt-abc")]
    rows = []
    for p in ("express-yourself", "uniquely-you", "emotional-state"):
        mine = [r for r in sub if r["prompt"] == p]
        rows.append([f"<td class='m'>{p}</td>", cell(len(mine), "int"),
                     cell((_minor_pct(mine) or 0) / 100, "pct"),
                     cell(_med(mine, "length_seconds"), "f0"),
                     cell(_med(mine, "note_density"), "f2"),
                     cell(_med(mine, "pitch_class_entropy"), "f2"),
                     cell(_med(mine, "valence"), "f2"),
                     cell(_med(mine, "arousal"), "f2")])
    return (
        "<h2>Prompt-variant effects <span class='sub'>(ABC, all arms pooled)</span></h2>"
        "<figure><figcaption>The three self-expression phrasings are the system-prompt "
        "manipulation ('You are a composer. &lt;variant&gt;'). Differences here are what "
        "one sentence of framing does to the music.</figcaption>"
        + table([("variant", None), ("n", None), ("minor", None), ("len s", None),
                 ("notes/beat", None), ("pc entropy", None),
                 ("valence", "Mode-based affect proxy, −1…1."),
                 ("arousal", "Tempo + density affect proxy, 0…1.")], rows)
        + "</figure>")


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
    body = (
        "<h1>v3 corpus — objective features <span class='sub'>42 arms × 3 prompts × 20 samples</span></h1>"
        "<p class='scope'>21 models (±thinking pairs) from five labs, released 2025-07 – "
        "2026-08. Each arm wrote 60 pieces per medium (3 self-expression variants × 20 "
        "samples): ABC notation and toolkit-free music21 code. The variant sentence is the "
        "system prompt; the user prompt is a fixed frame. Generated 2026-08-19/20; features v3.</p>"
        + _completion_section(pieces)
        + _per_arm_section(feats)
        + _thinking_section(feats)
        + _family_section(feats)
        + _trend_section(feats)
        + _keys_section(feats)
        + _prompt_section(feats)
    )
    html_text = page("v3 corpus — objective features", "results.html", body)
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
