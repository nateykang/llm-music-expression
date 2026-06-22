"""Generate the analysis dashboard — summary stats + charts, notebook-style.

Reads every ``features.csv`` (produced by ``llm-music analyze``) across all
batches, aggregates by model, and writes ``docs/results.html`` plus chart PNGs
into ``docs/analysis/``. The dashboard answers the inductive-bias question at a
glance: what do models default to (key/mode, affect, tempo, texture)?

Run: ``llm-music report``  (after ``llm-music analyze`` on the batches).
"""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

# Site palette (matches docs/style.css).
BG = "#faf8f5"
INK = "#2b2420"
MUTED = "#6b5d52"
ACCENT = "#7a5a3a"
# Warm qualitative palette for per-model series.
PALETTE = ["#7a5a3a", "#b5651d", "#3a6b5a", "#8a3a4a", "#4a5a7a",
           "#9a7a3a", "#5a7a4a", "#7a4a6a", "#3a7a7a", "#aa5a3a"]


def _f(v):
    """Parse a CSV cell to float, or None (NaN -> None so it drops from averages)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x  # NaN check


def load_features(data_dir: Path) -> list[dict]:
    """Load every features.csv, tagging each row with its batch + mode."""
    rows: list[dict] = []
    for csv_path in sorted(data_dir.glob("*/features.csv")):
        batch = csv_path.parent.name
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_batch"] = batch
                for k in ("valence", "arousal", "tempo_bpm", "scale_consistency",
                          "pitch_class_entropy", "note_density", "length_seconds",
                          "pitch_range", "polyphony", "n_voices"):
                    r[k] = _f(r.get(k))
                rows.append(r)
    return rows


def _model_order(rows: list[dict]) -> list[str]:
    counts = defaultdict(int)
    for r in rows:
        counts[r["model"]] += 1
    # Most-sampled first (stable, readable).
    return [m for m, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _agg(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (mean(vals) if vals else None), vals


def summarize(rows: list[dict]) -> list[dict]:
    """Per-model summary row: defaults across all of that model's pieces."""
    out = []
    for model in _model_order(rows):
        rs = [r for r in rows if r["model"] == model]
        minor = sum(r.get("key_mode") == "minor" for r in rs) / len(rs)
        out.append({
            "model": model, "n": len(rs), "minor_frac": minor,
            "valence": _agg(rs, "valence")[0], "arousal": _agg(rs, "arousal")[0],
            "tempo": _agg(rs, "tempo_bpm")[0],
            "scale_consistency": _agg(rs, "scale_consistency")[0],
            "note_density": _agg(rs, "note_density")[0],
            "length": _agg(rs, "length_seconds")[0],
            "pitch_range": _agg(rs, "pitch_range")[0],
        })
    return out


# --- charts -------------------------------------------------------------------

def _style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _jitter(i, n, spread=0.16):
    # deterministic spread of n points around integer i (no RNG: scripts forbid it)
    if n == 1:
        return [i]
    return [i - spread + 2 * spread * k / (n - 1) for k in range(n)]


def make_charts(rows: list[dict], out_dir: Path) -> list[tuple[str, str]]:
    """Render charts to PNGs. Returns [(filename, caption)]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    models = _model_order(rows)
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    charts: list[tuple[str, str]] = []

    def fig():
        f, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
        f.patch.set_facecolor(BG)
        _style_ax(ax)
        return f, ax

    def save(f, name):
        f.tight_layout()
        f.savefig(out_dir / name, facecolor=BG)
        plt.close(f)

    # 1. Mode preference per model (minor vs major fraction).
    f, ax = fig()
    ys = range(len(models))
    minor = [sum(r.get("key_mode") == "minor" for r in rows if r["model"] == m)
             / max(1, sum(r["model"] == m for r in rows)) for m in models]
    major = [sum(r.get("key_mode") == "major" for r in rows if r["model"] == m)
             / max(1, sum(r["model"] == m for r in rows)) for m in models]
    ax.barh(list(ys), minor, color="#5a6b8a", label="minor")
    ax.barh(list(ys), major, left=minor, color="#c08a3a", label="major")
    ax.set_yticks(list(ys)); ax.set_yticklabels(models)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of pieces")
    ax.set_title("Mode preference by model (minor vs major)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()
    save(f, "mode_by_model.png")
    charts.append(("mode_by_model.png",
                   "Each model's split between minor- and major-key pieces — the headline tonal bias."))

    # 2. Affect map: valence (jittered) × arousal, Russell circumplex.
    f, ax = fig()
    for m in models:
        rs = [r for r in rows if r["model"] == m and r.get("valence") is not None]
        xs = [r["valence"] + (0.06 * (k - len(rs) / 2) / max(1, len(rs))) for k, r in enumerate(rs)]
        ys2 = [r["arousal"] for r in rs]
        ax.scatter(xs, ys2, s=42, color=color[m], label=m, alpha=0.85, edgecolors="none")
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=":"); ax.axvline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(0, 1)
    ax.set_xlabel("valence  (minor ← → major)"); ax.set_ylabel("arousal  (calm → energetic)")
    ax.set_title("Affect map (valence × arousal)")
    for x, y, t in [(-0.7, 0.93, "tense/dark"), (0.7, 0.93, "bright/excited"),
                    (-0.7, 0.04, "sad/somber"), (0.7, 0.04, "serene/content")]:
        ax.text(x, y, t, fontsize=7.5, color=MUTED, ha="center")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(f, "affect_map.png")
    charts.append(("affect_map.png",
                   "Where each piece lands on the valence–arousal plane. Clustering low-left = a pull toward somber music."))

    # 3. Tempo distribution by model (strip plot).
    f, ax = fig()
    for i, m in enumerate(models):
        ts = [r["tempo_bpm"] for r in rows if r["model"] == m and r.get("tempo_bpm")]
        ax.scatter(_jitter(i, len(ts)), ts, s=34, color=color[m], alpha=0.8, edgecolors="none")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("tempo (BPM)"); ax.set_title("Tempo by model")
    ax.axhline(120, color=MUTED, lw=0.7, ls=":")
    ax.text(len(models) - 0.5, 122, "120 (default)", fontsize=7, color=MUTED, ha="right")
    save(f, "tempo_by_model.png")
    charts.append(("tempo_by_model.png",
                   "Tempo choices. A pile-up at 120 BPM is the un-thought-about default; spread below it is deliberate."))

    # 4. Texture: scale consistency × pitch-class entropy.
    f, ax = fig()
    for m in models:
        rs = [r for r in rows if r["model"] == m
              and r.get("scale_consistency") is not None and r.get("pitch_class_entropy") is not None]
        ax.scatter([r["scale_consistency"] for r in rs], [r["pitch_class_entropy"] for r in rs],
                   s=42, color=color[m], label=m, alpha=0.85, edgecolors="none")
    ax.set_xlabel("scale consistency (diatonic →)"); ax.set_ylabel("pitch-class entropy (chromatic →)")
    ax.set_title("Tonal texture: diatonic vs chromatic")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(f, "texture.png")
    charts.append(("texture.png",
                   "High scale-consistency + low entropy = tonal/simple; lower-right would be adventurous harmony."))

    return charts


# --- HTML ---------------------------------------------------------------------

def _fmt(v, pct=False, dp=2):
    if v is None:
        return "—"
    if pct:
        return f"{v*100:.0f}%"
    return f"{v:.{dp}f}"


def render_html(rows: list[dict], charts: list[tuple[str, str]], out_path: Path) -> None:
    summary = summarize(rows)
    n_pieces = len(rows)
    n_models = len({r["model"] for r in rows})
    n_batches = len({r["_batch"] for r in rows})

    # Free-form-only defaults (the purest bias probe).
    ff = [r for r in rows if r["prompt"] == "free-form"]
    ff_summary = summarize(ff) if ff else []

    def table(summ, caption):
        head = ("<tr><th>model</th><th>n</th><th>minor</th><th>valence</th>"
                "<th>arousal</th><th>tempo</th><th>scale&nbsp;consist.</th>"
                "<th>note&nbsp;density</th><th>length&nbsp;(s)</th></tr>")
        body = ""
        for s in summ:
            body += (f"<tr><td class='m'>{html.escape(s['model'])}</td><td>{s['n']}</td>"
                     f"<td>{_fmt(s['minor_frac'], pct=True)}</td>"
                     f"<td>{_fmt(s['valence'])}</td><td>{_fmt(s['arousal'])}</td>"
                     f"<td>{_fmt(s['tempo'], dp=0)}</td>"
                     f"<td>{_fmt(s['scale_consistency'])}</td>"
                     f"<td>{_fmt(s['note_density'])}</td>"
                     f"<td>{_fmt(s['length'], dp=0)}</td></tr>")
        return f"<figure><figcaption>{caption}</figcaption><table>{head}{body}</table></figure>"

    chart_html = "".join(
        f"<figure class='chart'><img src='analysis/{fn}' alt='{html.escape(cap)}'>"
        f"<figcaption>{html.escape(cap)}</figcaption></figure>"
        for fn, cap in charts
    )

    ff_section = (f"<h2>Free-form defaults <span class='sub'>(the purest bias probe — "
                  f"{len(ff)} pieces)</span></h2>{table(ff_summary, 'What each model reaches for when asked only to express itself.')}"
                  if ff_summary else "")

    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results — LLM musical inductive biases</title>
<link rel="stylesheet" href="style.css?v=22">
<style>
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
  .sub {{ color: {MUTED}; font-weight: 400; font-size: .8em; }}
  .scope {{ color: {MUTED}; font-size: .9rem; margin: .25rem 0 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .9rem; }}
  th, td {{ text-align: right; padding: .35rem .55rem; border-bottom: 1px solid #e7ddd2; }}
  th {{ color: {MUTED}; font-weight: 600; }}
  td.m, th:first-child {{ text-align: left; font-weight: 600; }}
  figure {{ margin: 1.5rem 0; }}
  figcaption {{ color: {MUTED}; font-size: .85rem; margin-top: .4rem; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem 2rem; }}
  .chart img {{ width: 100%; border: 1px solid #e7ddd2; border-radius: 8px; }}
  @media (max-width: 760px) {{ .charts {{ grid-template-columns: 1fr; }} }}
  nav.top a {{ color: {ACCENT}; text-decoration: none; font-weight: 600; }}
</style>
</head><body>
<nav class="tabs">
  <a href="index.html">Browse outputs</a>
  <a href="results.html" class="active">Results &amp; analysis</a>
</nav>
<div class="wrap">
  <h1>What do LLMs default to, musically?</h1>
  <p class="scope">Summary metrics across <b>{n_pieces} generated pieces</b>
     ({n_models} models, {n_batches} experiments, code-gen + ABC + SMT-ABC).
     Metrics are standard symbolic-music descriptors (MusPy + music21); affect is a
     valence/arousal proxy (Russell circumplex). Generated by <code>llm-music report</code>.</p>

  {ff_section}

  <h2>All pieces, by model</h2>
  {table(summary, 'Aggregated over every prompt and generation mode. "n" is how many pieces that model contributed.')}

  <h2>Charts</h2>
  <div class="charts">{chart_html}</div>

  <p class="scope" style="margin-top:2rem">Note: with small n per model this is a
     <b>snapshot</b>, not a settled result — the sampling run (many free-form pieces per
     model) is what turns these into statistically-backed claims.</p>
</div>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")

