#!/usr/bin/env python3
"""Build a BLIND human-rating page to validate the LLM-judge against perception.

Selects a score-stratified, model-diverse sample of judged free-form pieces, strips
all text (titles / composer notes / comments / voice names), and emits a self-
contained HTML page where a human listens (abcjs synth, blind) and rates each piece
on the same rubric the judge used. The page exports the ratings as JSON; pair it
with score_validation.py to get judge-vs-human Spearman correlations per dimension.

Usage:  python scripts/build_validation.py [--n 25] [--seed 7]
Writes:  docs/validation/rate.html      (give this to raters)
         docs/validation/sample_key.json (id -> identity + judge scores; for scoring)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_music.judge import AFFECT, EMOTION_LABELS, QUALITY, _strip_abc_text  # noqa: E402

RATED = QUALITY + AFFECT  # 1-5 dims shown to the human (no intent — blind)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_judged() -> list[dict]:
    """judge.csv rows (free-form) joined to their ABC from the batch manifests."""
    abc_by_key = {}
    for fn in ROOT.glob("docs/data/2026*/data.json"):
        m = json.loads(fn.read_text(encoding="utf-8"))
        for p in m.get("pieces", []):
            if p.get("ok") and p.get("abc"):
                abc_by_key[(p["model"], p["prompt"], p.get("mode", ""), p.get("title", ""))] = p["abc"]
    rows = []
    for r in csv.DictReader((ROOT / "docs/analysis/judge.csv").open(encoding="utf-8")):
        if r["prompt"] != "free-form":
            continue
        key = (r["model"], r["prompt"], r["mode"], r["title"])
        if key in abc_by_key and _f(r.get("overall")) is not None:
            r["_abc"] = abc_by_key[key]
            rows.append(r)
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Spread the sample across the judge's overall-score range (so correlation has
    variance to work with) and across models (so it isn't one model's profile)."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: _f(r["overall"]))
    bins = min(5, n)
    per = n // bins
    chunk = max(1, len(rows) // bins)
    picked = []
    for b in range(bins):
        lo = b * chunk
        hi = len(rows) if b == bins - 1 else (b + 1) * chunk
        pool = rows[lo:hi]
        rng.shuffle(pool)
        seen = defaultdict(int)
        take = []
        for r in pool:  # prefer model diversity within the bin
            if seen[r["model"]] < 2:
                take.append(r)
                seen[r["model"]] += 1
            if len(take) >= per:
                break
        picked.extend(take)
    # top up to n if integer division left us short
    if len(picked) < n:
        extra = [r for r in rows if r not in picked]
        rng.shuffle(extra)
        picked.extend(extra[: n - len(picked)])
    rng.shuffle(picked)  # blind order — don't leak quality by position
    return picked[:n]


def _playable(abc: str) -> str:
    """Drop %%score / %%staves layout directives — they're purely visual (don't
    affect the audio) and crash abcjs when a model's voice ids in the directive
    (e.g. %%score (V1)) don't match the declared headers (V:1)."""
    return "\n".join(ln for ln in abc.splitlines()
                     if not ln.strip().startswith(("%%score", "%%staves")))


def render_page(sample: list[dict]) -> str:
    dims = [{"key": k, "label": lbl, "lo": lo, "hi": hi} for k, lbl, q, lo, hi in RATED]
    pieces = [{"id": f"P{i + 1:02d}", "abc": _playable(_strip_abc_text(r["_abc"]))}
              for i, r in enumerate(sample)]
    cfg = json.dumps({"dims": dims, "labels": EMOTION_LABELS, "pieces": pieces})
    return _HTML.replace("/*CONFIG*/", cfg)


# The page renders each (text-stripped) ABC to a hidden abcjs visual and plays it
# through the Web-Audio synth — the rater hears blind audio, sees no notation unless
# they ask. Ratings collect into a JSON download.
_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Blind music rating</title>
<script src="https://cdn.jsdelivr.net/npm/abcjs@6.4.4/dist/abcjs-basic-min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/abcjs@6.4.4/abcjs-audio.css"/>
<style>
  body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:0 auto;padding:24px;color:#222}
  h1{font-size:21px} .sub{color:#666}
  .card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0;background:#fafafa}
  .pid{font-weight:700;font-size:16px}
  .dims{display:grid;grid-template-columns:1fr;gap:8px;margin-top:12px}
  .dim{display:grid;grid-template-columns:200px 1fr;align-items:center;gap:8px}
  .dim .lab{font-size:13px} .anchor{color:#999;font-size:11px}
  .scale{display:flex;gap:4px} .scale label{cursor:pointer;font-size:12px;text-align:center}
  .scale input{display:block;margin:0 auto}
  select{font:inherit;padding:3px}
  .bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 0;z-index:5}
  button{font:inherit;padding:8px 16px;border-radius:8px;border:1px solid #888;background:#fff;cursor:pointer}
  button.primary{background:#1a7;color:#fff;border-color:#1a7}
  details{margin-top:8px} pre{font-size:11px;overflow:auto;background:#fff;padding:8px;border:1px solid #eee}
  .hidden{display:none}
</style></head>
<body>
<div class="bar">
  <strong>Blind rating</strong> — <span id="prog">0</span> rated ·
  <button class="primary" onclick="exportRatings()">Download my ratings</button>
</div>
<h1>Rate each piece on the music alone</h1>
<p class="sub">You'll hear short pieces with <b>no titles, composers, or descriptions</b>. Listen, then score each
dimension 1–5 (1 = low/poor, 5 = high/excellent) and pick the dominant emotion. Valence = dark→bright,
arousal = calm→energetic; these aren't quality (a dark piece isn't "worse"). Notation is hidden — open it only if you want.
When finished, click <b>Download my ratings</b> and send me the file.</p>
<div id="list"></div>
<script>
const CFG = /*CONFIG*/;
const visuals = {};
function el(t,a={},...kids){const e=document.createElement(t);for(const k in a){if(k==="class")e.className=a[k];else if(k==="html")e.innerHTML=a[k];else e.setAttribute(k,a[k]);}for(const k of kids)e.append(k);return e;}

CFG.pieces.forEach((pc)=>{
  const card=el("div",{class:"card"});
  card.append(el("div",{class:"pid"},"🎵 "+pc.id));
  const audio=el("div",{}); card.append(audio);
  const hidden=el("div",{class:"hidden"}); card.append(hidden);   // abcjs needs a render target
  // play button
  try{
    const v=ABCJS.renderAbc(hidden,pc.abc,{})[0]; visuals[pc.id]=v;
    const ctrlEl=el("div",{}); audio.append(ctrlEl);
    const sc=new ABCJS.synth.SynthController();
    sc.load(ctrlEl,null,{displayPlay:true,displayProgress:true});
    sc.setTune(v,false,{soundFontUrl:"https://paulrosen.github.io/midi-js-soundfonts/abcjs/"});
  }catch(e){ audio.append(el("p",{},"(audio unavailable for this piece)")); }
  // notation toggle (hidden by default — blind)
  const det=el("details"); det.append(el("summary",{},"show notation"));
  const note=el("div",{}); det.append(note); card.append(det);
  det.addEventListener("toggle",()=>{ if(det.open && !note.dataset.done){ ABCJS.renderAbc(note,pc.abc,{responsive:"resize"}); note.dataset.done="1"; } });
  // rubric
  const dims=el("div",{class:"dims"});
  CFG.dims.forEach((d)=>{
    const row=el("div",{class:"dim"});
    row.append(el("div",{class:"lab"},el("div",{},d.label),el("div",{class:"anchor"},"1="+d.lo+" · 5="+d.hi)));
    const scale=el("div",{class:"scale"});
    for(let s=1;s<=5;s++){
      const id=pc.id+"_"+d.key+"_"+s;
      scale.append(el("label",{for:id},el("input",{type:"radio",name:pc.id+"_"+d.key,id,value:s,onchange:""}),String(s)));
    }
    scale.addEventListener("change",updateProgress);
    row.append(scale); dims.append(row);
  });
  // emotion label
  const erow=el("div",{class:"dim"});
  erow.append(el("div",{class:"lab"},"dominant emotion"));
  const sel=el("select",{name:pc.id+"_emotion_label"});
  sel.append(el("option",{value:""},"—"));
  CFG.labels.forEach((l)=>sel.append(el("option",{value:l},l)));
  erow.append(sel); dims.append(erow);
  card.append(dims);
  document.getElementById("list").append(card);
});

function gather(){
  const out={};
  CFG.pieces.forEach((pc)=>{
    const rec={};
    CFG.dims.forEach((d)=>{const c=document.querySelector(`input[name="${pc.id}_${d.key}"]:checked`);if(c)rec[d.key]=+c.value;});
    const sel=document.querySelector(`select[name="${pc.id}_emotion_label"]`);
    if(sel&&sel.value)rec.emotion_label=sel.value;
    if(Object.keys(rec).length)out[pc.id]=rec;
  });
  return out;
}
function updateProgress(){
  const g=gather(); let done=0;
  CFG.pieces.forEach((pc)=>{ if(g[pc.id]&&CFG.dims.every((d)=>d.key in g[pc.id]))done++; });
  document.getElementById("prog").textContent=done+" / "+CFG.pieces.length;
}
function exportRatings(){
  const blob=new Blob([JSON.stringify(gather(),null,1)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="human_ratings.json"; a.click();
}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_judged()
    if len(rows) < args.n:
        print(f"only {len(rows)} judged ABC pieces available; using all of them")
    sample = stratified_sample(rows, min(args.n, len(rows)), args.seed)

    out_dir = ROOT / "docs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rate.html").write_text(render_page(sample), encoding="utf-8")

    key = {}
    for i, r in enumerate(sample):
        pid = f"P{i + 1:02d}"
        key[pid] = {"model": r["model"], "prompt": r["prompt"], "mode": r["mode"], "title": r["title"],
                    "judge": {k: _f(r.get(k)) for k in [d[0] for d in RATED] + ["overall"]}}
    (out_dir / "sample_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")

    models = defaultdict(int)
    for r in sample:
        models[r["model"]] += 1
    print(f"Wrote {out_dir/'rate.html'} ({len(sample)} blind pieces)")
    print(f"  models: {dict(models)}")
    print(f"  overall spread: {min(_f(r['overall']) for r in sample):.2f} .. {max(_f(r['overall']) for r in sample):.2f}")
    print(f"  key → {out_dir/'sample_key.json'}")


if __name__ == "__main__":
    main()
