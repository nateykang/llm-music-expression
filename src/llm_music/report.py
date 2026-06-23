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
import json
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

# Circle of fifths: major keys centered on C, relative minors centered on A,
# aligned column-wise (each position is a relative pair sharing a key signature).
MAJOR_FIFTHS = ["Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#"]
MINOR_FIFTHS = ["Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m"]
KEY_MAJOR = "#BA7517"
KEY_MINOR = "#378ADD"
# Preferred model order for the key widget (others appended as found).
KEY_MODEL_ORDER = ["gpt-5.5", "opus-4.8", "opus-4.8-thinking", "sonnet-4.6", "gpt-4.1",
                   "gemini-2.5-pro", "grok-4.3", "deepseek-v4-pro", "qwen3-max", "llama-4-maverick"]

# Page-wide representation toggle: ABC (declared K:) vs code-gen (music21-detected).
REPS = [("text", "ABC"), ("code", "code-gen"), ("all", "both")]


def _rep_filter(rows, rep):
    if rep == "all":
        return rows
    want_code = rep == "code"
    return [r for r in rows if (r.get("mode") == "codegen") == want_code]

# Summary-table columns: (summary-key, header label, hover definition, format).
COLUMNS = [
    ("model", "model", "The language model that generated the pieces.", "text"),
    ("n", "n", "Number of pieces this row aggregates.", "int"),
    ("minor_frac", "minor", "Share of pieces in a minor key. Mode comes from the model's "
        "DECLARED key (the ABC K: field — its stated intent) where available, falling back to "
        "music21's Krumhansl–Schmuckler detection for code-gen pieces (which have no K:).", "pct"),
    ("mode_match", "mode match", "Intent-vs-execution: how often the actual notes (music21 "
        "detection) match the major/minor mode the model DECLARED in K:. 100% = it plays what it "
        "writes; lower = it declares one mode but the notes lean the other way. Blank for code-gen "
        "(no K: field to compare against).", "pct"),
    ("valence", "valence", "How positive/bright vs negative/dark the mood sounds (−1 to +1). A "
        "deliberately simple proxy: major key → +1, minor key → −1 (the strongest single cue for "
        "musical 'happiness'). Concept from the Russell circumplex; the mapping is ours, not a "
        "trained emotion model — read it as 'bright vs dark', not literal joy.", "f2"),
    ("arousal", "arousal", "An 'energy' level from 0 (calm) to 1 (energetic) = 0.6 × tempo + 0.4 × "
        "rhythmic-density, with tempo rescaled 50 BPM→0 / 160 BPM→1 and density (notes/beat) "
        "rescaled 0→0 / 4→1, each clipped to [0,1]. Our heuristic, not a standard metric.", "f2"),
    ("tempo", "tempo", "Tempo in beats per minute, from the score's metronome mark (120 if unset).", "f0"),
    ("scale_consistency", "scale consist.", "MusPy scale consistency: the largest fraction of notes "
        "that fit a single major or minor scale. 1.0 = perfectly diatonic; lower = more chromatic / "
        "out-of-key notes.", "f2"),
    ("consonance", "consonance", "MuSpike-style Pitch Consonance: the fraction of vertical sonorities "
        "(the score chordified, ≥2 notes) that are consonant. Higher = harmonically cleaner vertical "
        "writing; lower = more clashing/dissonant simultaneities.", "f2"),
    ("chord_tone", "chord-tones", "MuSpike-style Chord-Tone ratio: per bar the prevailing harmony is the "
        "3 most-present pitch classes; this is the share of notes belonging to it. Higher = notes stay "
        "within the underlying chord; lower = more non-chord / passing tones.", "pct"),
    ("note_density", "note density", "Average note onsets per beat (a beat = one quarter note), so it is "
        "tempo-invariant — rhythmic busyness, not real-time speed. ~1 = about one note per beat; higher = "
        "runs or chords. (Our descriptor, not from a specific paper.)", "f2"),
    ("length", "length (s)", "Duration of the piece in seconds.", "f0"),
]

# Reliability table columns (computed from batch manifests, not features.csv).
REL_COLUMNS = [
    ("model", "model", "The model.", "text"),
    ("gen", "method", "Generation method.", "text"),
    ("n", "n", "Pieces generated.", "int"),
    ("first_ok", "1st-try valid", "Share that passed the validity gate on the FIRST attempt "
        "(code-gen: executed without error; ABC: passed the syntax gate). A ChatMusician-style "
        "format-success rate. Caveat: the gates differ in strictness — code must run, while our ABC "
        "gate is lenient, so ABC's number overstates true musical validity.", "pct"),
    ("fail", "failed", "Share that never produced a valid result within the attempt budget.", "pct"),
    ("avg_attempts", "avg tries", "Mean attempts until a valid generation (1 = first try). Higher = the "
        "model slipped and the harness had to retry.", "f2"),
]


def _cell(v, kind):
    if v is None:
        return "—"
    if kind == "text":
        return html.escape(str(v))
    if kind == "int":
        return str(int(v))
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "f0":
        return f"{v:.0f}"
    return f"{v:.2f}"


def _table_html(rows, columns):
    head = "<tr>" + "".join(
        f"<th><span class='tip' tabindex='0' data-tip=\"{html.escape(tip)}\">"
        f"{html.escape(lbl).replace(' ', '&nbsp;')}</span></th>"
        for _, lbl, tip, _ in columns) + "</tr>"
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td class='{'m' if key in ('model', 'gen') else ''}'>{_cell(r.get(key), kind)}</td>"
            for key, _, _, kind in columns) + "</tr>"
    return f"<div class='tscroll'><table>{head}{body}</table></div>"


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
                          "pitch_range", "polyphony", "n_voices",
                          "consonance_rate", "chord_tone_rate", "mode_match"):
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
        # minor% from the declared-preferred mode (K: field where available).
        modes = [r.get("key_mode_best") for r in rs if r.get("key_mode_best")]
        minor = (sum(m == "minor" for m in modes) / len(modes)) if modes else None
        matches = [r["mode_match"] for r in rs if r.get("mode_match") is not None]
        out.append({
            "model": model, "n": len(rs), "minor_frac": minor,
            "mode_match": (mean(matches) if matches else None),
            "valence": _agg(rs, "valence")[0], "arousal": _agg(rs, "arousal")[0],
            "tempo": _agg(rs, "tempo_bpm")[0],
            "scale_consistency": _agg(rs, "scale_consistency")[0],
            "consonance": _agg(rs, "consonance_rate")[0],
            "chord_tone": _agg(rs, "chord_tone_rate")[0],
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


# --- key choices (circle of fifths) -------------------------------------------

def key_distributions(rows, min_n=10):
    """Free-form per-model key counts, split by representation:
    {'text': {model:{key:n}}, 'code': ..., 'all': ...}. 'text' = ABC/SMT-ABC
    (declared K:), 'code' = code-gen (music21-detected), 'all' = both."""
    from collections import Counter

    buckets = {"text": {}, "code": {}, "all": {}}
    for r in rows:
        if r["prompt"] != "free-form":
            continue
        tonic = (r.get("key_declared_tonic") or r.get("key_tonic") or "").replace("-", "b")
        mode = r.get("key_mode_best") or r.get("key_mode")
        if not tonic or mode in ("", "?"):
            continue
        lab = tonic if mode == "major" else tonic + "m"
        rep = "code" if r.get("mode") == "codegen" else "text"
        for b in (rep, "all"):
            buckets[b].setdefault(r["model"], Counter())[lab] += 1
    keep = lambda d: {m: dict(c) for m, c in d.items() if sum(c.values()) >= min_n}
    return {k: keep(v) for k, v in buckets.items()}


def make_key_chart(ff, out_dir):
    """Static circle-of-fifths key chart (free-form, all models) for PDF/print."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    agg = {}
    for c in ff.values():
        for k, v in c.items():
            agg[k] = agg.get(k, 0) + v
    maj = [agg.get(k, 0) for k in MAJOR_FIFTHS]
    minr = [agg.get(k, 0) for k in MINOR_FIFTHS]
    xs = list(range(len(MAJOR_FIFTHS)))
    f, ax = plt.subplots(figsize=(8.4, 4.2), dpi=130)
    f.patch.set_facecolor(BG)
    _style_ax(ax)
    b1 = ax.bar([x - 0.2 for x in xs], maj, 0.4, color=KEY_MAJOR, label="major")
    b2 = ax.bar([x + 0.2 for x in xs], minr, 0.4, color=KEY_MINOR, label="minor")
    total = sum(maj) + sum(minr) or 1
    pct = lambda v: ("" if not v else "<1%" if 100 * v / total < 0.5 else f"{round(100 * v / total)}%")
    ax.bar_label(b1, labels=[pct(v) for v in maj], padding=2, fontsize=7, color=MUTED)
    ax.bar_label(b2, labels=[pct(v) for v in minr], padding=2, fontsize=7, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}\n{mk}" for k, mk in zip(MAJOR_FIFTHS, MINOR_FIFTHS)], fontsize=8)
    ax.set_ylabel("pieces")
    ax.set_title("Key choice on the circle of fifths (free-form, all models)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    f.tight_layout()
    f.savefig(out_dir / "key_circle.png", facecolor=BG)
    plt.close(f)
    return ("key_circle.png",
            "Major keys (amber) along the circle of fifths centered on C; relative minors (blue) "
            "centered on A and column-aligned. Free-form, all models — usage concentrates on C major / D minor.")


KEY_WIDGET_CSS = """
  .keyviz { margin: 1rem 0 1.5rem; }
  .kv-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .kv-btn { font-size: 13px; padding: 5px 10px; border: 1px solid #e0d5c8; background: #fff;
            color: #2b2420; border-radius: 6px; cursor: pointer; }
  .kv-btn:hover { border-color: #c9b69f; }
  .kv-btn[aria-pressed=true] { background: #7a5a3a; color: #fff; border-color: #7a5a3a; }
  .kv-stat { font-size: .9rem; color: #6b5d52; margin: .6rem 0; min-height: 20px; }
  .kv-legend { display: flex; gap: 16px; font-size: .8rem; color: #6b5d52; margin-bottom: 8px; }
  .kv-sw { width: 11px; height: 11px; border-radius: 2px; display: inline-block; vertical-align: -1px; margin-right: 5px; }
  .kv-chartwrap { position: relative; width: 100%; height: 320px; }
"""

KEY_WIDGET_TMPL = """
<h2>Key choices <span class='sub'>(circle of fifths — major on C, relative minors on A)</span></h2>
<p class="scope">Which keys each model chooses in free-form. Toggle representation — ABC uses the model's declared K:, code-gen uses music21's detected key. Click a model to filter. (Note the "default" models swing toward minor in code-gen, where there's no lazy K:C default.)</p>
<div class="keyviz">
  <div id="kv-models" class="kv-row" style="margin-bottom:.5rem;"></div>
  <div id="kv-stat" class="kv-stat"></div>
  <div class="kv-legend"><span><span class="kv-sw" style="background:#BA7517;"></span>major (top label)</span><span><span class="kv-sw" style="background:#378ADD;"></span>minor (bottom label)</span><span style="margin-left:auto;">← flats · sharps →</span></div>
  <div class="kv-chartwrap"><canvas id="kv-chart" role="img" aria-label="Key choices along the circle of fifths, major in amber and minor in blue"></canvas></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
(function(){
  const TEXT=__TEXT__, CODE=__CODE__, ALL=__ALL__, MODELS=__MODELS__;
  const majorKeys=["Db","Ab","Eb","Bb","F","C","G","D","A","E","B","F#"];
  const minorKeys=["Bbm","Fm","Cm","Gm","Dm","Am","Em","Bm","F#m","C#m","G#m","D#m"];
  const pp=t=>t.replace('b','\\u266d').replace('#','\\u266f');
  const majorPretty=majorKeys.map(pp);
  const minorPretty=minorKeys.map(t=>pp(t.slice(0,-1))+'m');
  const xlabels=majorPretty.map((mj,i)=>[mj,minorPretty[i]]);
  let scope='text', current='All';
  const base=()=>scope==='code'?CODE:scope==='all'?ALL:TEXT;
  function srcFor(sel){const D=base();if(sel!=='All')return D[sel]||{};const t={};for(const m in D)for(const k in D[m])t[k]=(t[k]||0)+D[m][k];return t;}
  const major=sel=>{const s=srcFor(sel);return majorKeys.map(t=>s[t]||0);};
  const minor=sel=>{const s=srcFor(sel);return minorKeys.map(t=>s[t]||0);};
  function updateStat(sel){const M=major(sel),N=minor(sel);const tot=M.reduce((a,b)=>a+b,0)+N.reduce((a,b)=>a+b,0)||1;const ms=N.reduce((a,b)=>a+b,0);let fav='',fmax=-1;M.forEach((v,i)=>{if(v>fmax){fmax=v;fav=majorPretty[i];}});N.forEach((v,i)=>{if(v>fmax){fmax=v;fav=minorPretty[i];}});document.getElementById('kv-stat').textContent=(sel==='All'?'all models':sel)+' \\u00b7 '+tot+' pieces \\u00b7 '+Math.round(100*ms/tot)+'% minor \\u00b7 favourite: '+fav+' ('+fmax+')';}
  function styleBtns(box,val,attr){Array.prototype.forEach.call(box.querySelectorAll('button'),function(b){b.setAttribute('aria-pressed',b.dataset[attr]===val);});}
  function render(){chart.data.datasets[0].data=major(current);chart.data.datasets[1].data=minor(current);chart.update();updateStat(current);styleBtns(document.getElementById('kv-models'),current,'m');}
  window.__kvSetScope=function(rep){scope=rep;render();};
  const bw=document.getElementById('kv-models');
  ['All'].concat(MODELS).forEach(function(m){const b=document.createElement('button');b.className='kv-btn';b.textContent=m;b.dataset.m=m;b.onclick=function(){current=m;render();};bw.appendChild(b);});
  const countLabels={id:'countLabels',afterDatasetsDraw:function(ch){var x=ch.ctx;var total=0;ch.data.datasets.forEach(function(ds){ds.data.forEach(function(v){total+=v;});});if(!total)return;x.save();x.font='11px -apple-system,system-ui,sans-serif';x.fillStyle='#6b5d52';x.textAlign='center';ch.data.datasets.forEach(function(ds,di){var meta=ch.getDatasetMeta(di);meta.data.forEach(function(bar,i){var v=ds.data[i];if(v>0){var p=100*v/total;x.fillText(p<0.5?'<1%':Math.round(p)+'%',bar.x,bar.y-4);}});});x.restore();}};Chart.register(countLabels);
  const chart=new Chart(document.getElementById('kv-chart'),{type:'bar',data:{labels:xlabels,datasets:[{label:'major',data:major('All'),backgroundColor:'#BA7517',borderWidth:0,borderRadius:2},{label:'minor',data:minor('All'),backgroundColor:'#378ADD',borderWidth:0,borderRadius:2}]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:300},plugins:{legend:{display:false},tooltip:{callbacks:{title:function(items){const i=items[0].dataIndex;return items[0].dataset.label==='minor'?minorPretty[i]+' minor':majorPretty[i]+' major';},label:function(ctx){const M=major(current),N=minor(current);const tot=M.reduce((a,b)=>a+b,0)+N.reduce((a,b)=>a+b,0)||1;const v=ctx.parsed.y;return v+' piece'+(v===1?'':'s')+' \\u00b7 '+Math.round(100*v/tot)+'%';}}}},scales:{x:{grid:{display:false},ticks:{color:'#6b5d52',font:{size:12},autoSkip:false}},y:{beginAtZero:true,ticks:{precision:0,color:'#6b5d52'},grid:{color:'rgba(0,0,0,0.08)'}}}}});
  render();
})();
</script>
"""


def _key_widget_html(dists):
    text, code, allb = dists["text"], dists["code"], dists["all"]
    allmodels = set(text) | set(code) | set(allb)
    present = [m for m in KEY_MODEL_ORDER if m in allmodels]
    present += [m for m in sorted(allmodels) if m not in present]
    return (KEY_WIDGET_TMPL
            .replace("__TEXT__", json.dumps(text))
            .replace("__CODE__", json.dumps(code))
            .replace("__ALL__", json.dumps(allb))
            .replace("__MODELS__", json.dumps(present)))


# --- reliability (from manifests, not features.csv) ---------------------------

def load_reliability(data_dir: Path) -> list[dict]:
    """Per (model, generation method): validity/retry stats from batch manifests."""
    from statistics import mean

    agg: dict[tuple, list] = {}
    for f in sorted(data_dir.glob("*/data.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for p in m.get("pieces", []):
            agg.setdefault((p["model"], p.get("mode")), []).append(p)
    rows = []
    for (model, mode), ps in sorted(agg.items()):
        n = len(ps)
        rows.append({
            "model": model, "gen": mode, "n": n,
            "first_ok": sum(bool(p.get("attempts") == 1 and p.get("ok")) for p in ps) / n,
            "fail": sum(not p.get("ok") for p in ps) / n,
            "avg_attempts": round(mean(p.get("attempts", 1) for p in ps), 2),
        })
    return rows


# --- HTML ---------------------------------------------------------------------


def render_html(rows: list[dict], charts: list[tuple[str, str]], out_path: Path,
                reliability: list[dict] | None = None, dists: dict | None = None) -> None:
    summary = summarize(rows)
    n_pieces = len(rows)
    n_models = len({r["model"] for r in rows})
    n_batches = len({r["_batch"] for r in rows})

    # Free-form-only defaults (the purest bias probe).
    ff = [r for r in rows if r["prompt"] == "free-form"]

    # Each table is rendered once per representation (ABC / code-gen / both); the
    # page-wide toggle shows one pane at a time. ABC and code-gen are different runs,
    # so their n and metrics shouldn't be blended by default.
    def table(base_rows, caption):
        panes = "".join(
            f"<div class='rep-pane' data-rep='{rep}'{'' if rep == 'text' else ' hidden'}>"
            f"{_table_html(summarize(_rep_filter(base_rows, rep)), COLUMNS)}</div>"
            for rep, _lbl in REPS)
        return f"<figure><figcaption>{caption}</figcaption>{panes}</figure>"

    key_widget = _key_widget_html(dists) if dists and any(dists.values()) else ""

    rel_section = ""
    if reliability:
        rel_panes = "".join(
            f"<div class='rep-pane' data-rep='{rep}'{'' if rep == 'text' else ' hidden'}>"
            f"{_table_html([x for x in reliability if rep == 'all' or (x.get('gen') == 'codegen') == (rep == 'code')], REL_COLUMNS)}</div>"
            for rep, _lbl in REPS)
        rel_section = (
            "<h2>Reliability <span class='sub'>(format-success &amp; retries, per method)</span></h2>"
            "<figure><figcaption>How often each model produced a valid generation, and how many "
            "tries it took. Code-gen fails loudly (the interpreter rejects bad code → retry); ABC "
            "fails quietly (a lenient syntax gate passes, so 1st-try rates run high). A "
            f"ChatMusician-style format-success view.</figcaption>{rel_panes}</figure>"
        )

    chart_html = "".join(
        f"<figure class='chart'><img src='analysis/{fn}' alt='{html.escape(cap)}'>"
        f"<figcaption>{html.escape(cap)}</figcaption></figure>"
        for fn, cap in charts
    )

    ff_section = (f"<h2>Free-form defaults <span class='sub'>(the purest bias probe)</span></h2>"
                  f"{table(ff, 'What each model reaches for when asked only to express itself.')}"
                  if ff else "")

    doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results — LLM musical inductive biases</title>
<link rel="stylesheet" href="style.css?v=22">
<style>
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
  .sub {{ color: {MUTED}; font-weight: 400; font-size: .8em; }}
  .scope {{ color: {MUTED}; font-size: .9rem; margin: .25rem 0 1.5rem; }}
  .tscroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .9rem; }}
  th, td {{ text-align: right; padding: .35rem .55rem; border-bottom: 1px solid #e7ddd2; }}
  th {{ color: {MUTED}; font-weight: 600; position: relative; }}
  .tip {{ border-bottom: 1px dotted {MUTED}; cursor: help; outline: none; }}
  th .tip:hover::after, th .tip:focus::after {{
    content: attr(data-tip); position: absolute; left: 0; top: 145%; z-index: 30;
    width: 240px; white-space: normal; text-align: left; font-weight: 400;
    font-size: .76rem; line-height: 1.45; color: {BG}; background: {INK};
    padding: .55rem .65rem; border-radius: 7px; box-shadow: 0 6px 20px rgba(0,0,0,.2);
  }}
  th:nth-last-child(-n+3) .tip:hover::after,
  th:nth-last-child(-n+3) .tip:focus::after {{ left: auto; right: 0; }}
  td.m, th:first-child {{ text-align: left; font-weight: 600; }}
  figure {{ margin: 1.5rem 0; }}
  figcaption {{ color: {MUTED}; font-size: .85rem; margin-top: .4rem; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem 2rem; }}
  .chart img {{ width: 100%; border: 1px solid #e7ddd2; border-radius: 8px; }}
  @media (max-width: 760px) {{ .charts {{ grid-template-columns: 1fr; }} }}
  nav.top a {{ color: {ACCENT}; text-decoration: none; font-weight: 600; }}
{KEY_WIDGET_CSS}
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

  <div class="kv-row" style="margin:0 0 1.5rem; align-items:center; gap:10px; position:sticky; top:0; background:{BG}; padding:.5rem 0; z-index:5;">
    <span style="font-size:.9rem; color:{INK}; font-weight:500;">Representation</span>
    <span id="rep-toggle" class="kv-row">
      <button class="kv-btn" data-rep="text" aria-pressed="true">ABC</button>
      <button class="kv-btn" data-rep="code">code-gen</button>
      <button class="kv-btn" data-rep="all">both</button>
    </span>
    <span style="font-size:.8rem; color:{MUTED};">ABC and code-gen are separate runs — "both" blends them.</span>
  </div>

  {ff_section}

  <h2>All pieces, by model</h2>
  {table(rows, 'Aggregated over every prompt for the selected representation. "n" is how many pieces that model contributed.')}

  {key_widget}

  {rel_section}

  <h2>Charts</h2>
  <div class="charts">{chart_html}</div>

  <h2>Methods &amp; references</h2>
  <p class="scope">
    Metrics are standard symbolic-music descriptors, computed with
    <a href="https://arxiv.org/abs/2008.01951">MusPy</a> (Dong et al., ISMIR 2020) —
    scale consistency, pitch-class entropy, polyphony, etc. — and
    <a href="https://www.music21.org/">music21</a> for key (Krumhansl–Schmuckler) and
    tempo. Affect (valence/arousal) follows the
    <a href="https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model">Russell
    circumplex</a> model. <b>Hover any column header</b> for what it measures.
    Project after <a href="https://github.com/sara-fish/llm-musical-self-expression">sara-fish/llm-musical-self-expression</a>;
    representation/evaluation choices follow the LLM-music literature
    (<a href="https://arxiv.org/abs/2402.16153">ChatMusician</a>,
    <a href="https://arxiv.org/abs/2404.06393">MuPT</a>).
  </p>

  <p class="scope" style="margin-top:2rem">Note: the charts below aggregate all
     representations; the toggle above drives the tables and the key-choice chart.</p>
</div>
<script>
(function(){{
  function setRep(rep){{
    var panes=document.querySelectorAll('.rep-pane');
    for(var i=0;i<panes.length;i++) panes[i].hidden = panes[i].getAttribute('data-rep')!==rep;
    var btns=document.querySelectorAll('#rep-toggle button');
    for(var j=0;j<btns.length;j++) btns[j].setAttribute('aria-pressed', btns[j].getAttribute('data-rep')===rep);
    if(window.__kvSetScope) window.__kvSetScope(rep);
  }}
  var btns=document.querySelectorAll('#rep-toggle button');
  for(var k=0;k<btns.length;k++)(function(b){{ b.onclick=function(){{ setRep(b.getAttribute('data-rep')); }}; }})(btns[k]);
  setRep('text');
}})();
</script>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")

