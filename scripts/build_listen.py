#!/usr/bin/env python3
"""Build a 'listen & compare' page: hear each judged piece blind, form your own
opinion, then reveal the LLM-judge's panel scores + reasoning to check yourself.

Unlike the validation sheet (which collects YOUR ratings for correlation), this is
just for getting a feel — no data is collected. Pieces play blind (text stripped),
in random order; clicking "reveal the judge's verdict" shows the model, title, the
per-judge score grid, and each judge's reasoning.

Usage:  python scripts/build_listen.py [--n 48] [--seed 3]
Writes: docs/validation/listen.html  (open via the local server)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_validation import _playable  # noqa: E402
from llm_music.judge import _strip_abc_text  # noqa: E402

DIMS = ["coherence", "harmony", "rhythm", "structure", "melody", "emotion",
        "creativity", "naturalness", "valence", "arousal"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load() -> list[dict]:
    abc_by_key = {}
    for fn in ROOT.glob("docs/data/2026*/data.json"):
        m = json.loads(fn.read_text(encoding="utf-8"))
        for p in m.get("pieces", []):
            if p.get("ok") and p.get("abc"):
                abc_by_key[(p["model"], p["prompt"], p.get("mode", ""), p.get("title", ""))] = p["abc"]
    overall = {}
    for r in csv.DictReader((ROOT / "docs/analysis/judge.csv").open(encoding="utf-8")):
        overall[(r["model"], r["prompt"], r["mode"], r["title"])] = _f(r.get("overall"))
    raw = json.loads((ROOT / "docs/analysis/judge_raw.json").read_text(encoding="utf-8"))
    out = []
    for p in raw:
        key = (p["model"], p["prompt"], p.get("mode", ""), p.get("title", ""))
        if key in abc_by_key and overall.get(key) is not None:
            p["_abc"] = abc_by_key[key]
            p["_overall"] = overall[key]
            out.append(p)
    return out


def sample(rows, n, seed):
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["_overall"])
    if n >= len(rows):
        picked = rows[:]
    else:  # even spread across the quality range
        idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
        picked = [rows[i] for i in sorted(set(idx))]
    rng.shuffle(picked)  # blind order
    return picked


def verdict_payload(p):
    judges = list(p["panel"].keys())
    grid = {d: {j: (p["panel"][j].get(d) or {}).get("score") for j in judges} for d in DIMS}
    reasons = {j: {d: (p["panel"][j].get(d) or {}).get("reason", "") for d in DIMS} for j in judges}
    labels = {j: p["panel"][j].get("emotion_label", "") for j in judges}
    return {"model": p["model"], "title": p["title"], "overall": round(p["_overall"], 2),
            "judges": judges, "grid": grid, "reasons": reasons, "labels": labels}


def render(picked):
    pieces = [{"id": f"P{i + 1:02d}", "abc": _playable(_strip_abc_text(r["_abc"])),
               "verdict": verdict_payload(r)} for i, r in enumerate(picked)]
    return _HTML.replace("/*CONFIG*/", json.dumps({"dims": DIMS, "pieces": pieces}))


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Listen & compare to the judge</title>
<script src="https://cdn.jsdelivr.net/npm/abcjs@6.4.4/dist/abcjs-basic-min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/abcjs@6.4.4/abcjs-audio.css"/>
<style>
  body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:0 auto;padding:24px;color:#222}
  h1{font-size:21px} .sub{color:#666}
  .card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:16px 0;background:#fafafa}
  .pid{font-weight:700}
  details{margin-top:10px} summary{cursor:pointer;color:#1a7;font-weight:600}
  .who{margin:8px 0;font-size:14px}
  table{border-collapse:collapse;font-size:12px;margin:8px 0;width:100%}
  th,td{border:1px solid #e2e2e2;padding:3px 6px;text-align:center} th:first-child,td:first-child{text-align:left}
  .ov{font-weight:700}
  .reasons{font-size:12px;color:#444} .reasons b{color:#111}
  .reasons .j{margin:8px 0 4px;font-weight:700;color:#1a7}
  .hidden{display:none}
</style></head>
<body>
<h1>Listen, then check yourself against the judge</h1>
<p class="sub">Each piece plays <b>blind</b> (no title/model). Listen, decide what <i>you</i> think, then open
<b>"reveal the judge's verdict"</b> to see the panel's scores and reasoning. Nothing is recorded — this is just to
get a feel for whether the judge's calls match yours. Order is random.</p>
<div id="list"></div>
<script>
const CFG = /*CONFIG*/;
function el(t,a={},...k){const e=document.createElement(t);for(const x in a){if(x==="class")e.className=a[x];else if(x==="html")e.innerHTML=a[x];else e.setAttribute(x,a[x]);}for(const c of k)e.append(c);return e;}

function verdictHtml(v){
  const wrap=el("div");
  wrap.append(el("div",{class:"who"},el("span",{class:"ov"},"Judge overall "+v.overall+"  ·  "),v.model+" — “"+v.title+"”"));
  // score grid
  const tbl=el("table");
  const head=el("tr"); head.append(el("th",{},"dimension")); v.judges.forEach(j=>head.append(el("th",{},j.replace("-2.5-pro","").replace("-4.8",""))));
  tbl.append(head);
  CFG.dims.forEach(d=>{const tr=el("tr");tr.append(el("td",{},d));v.judges.forEach(j=>{const s=v.grid[d][j];tr.append(el("td",{},s==null?"–":String(s)));});tbl.append(tr);});
  const lr=el("tr"); lr.append(el("td",{},"emotion")); v.judges.forEach(j=>lr.append(el("td",{},v.labels[j]||"–"))); tbl.append(lr);
  wrap.append(tbl);
  // reasoning
  const rs=el("div",{class:"reasons"});
  v.judges.forEach(j=>{
    rs.append(el("div",{class:"j"},j));
    CFG.dims.forEach(d=>{const r=v.reasons[j][d];const s=v.grid[d][j];if(r)rs.append(el("div",{},el("b",{},`${d} ${s==null?"":s}: `),r));});
  });
  wrap.append(rs);
  return wrap;
}

CFG.pieces.forEach(pc=>{
  const card=el("div",{class:"card"});
  card.append(el("div",{class:"pid"},"🎵 "+pc.id));
  const audio=el("div",{}); card.append(audio);
  const hidden=el("div",{class:"hidden"}); card.append(hidden);
  try{
    const vis=ABCJS.renderAbc(hidden,pc.abc,{})[0];
    const ctrl=el("div",{}); audio.append(ctrl);
    const sc=new ABCJS.synth.SynthController();
    sc.load(ctrl,null,{displayPlay:true,displayProgress:true});
    sc.setTune(vis,false,{soundFontUrl:"https://paulrosen.github.io/midi-js-soundfonts/abcjs/"});
  }catch(e){ audio.append(el("p",{},"(audio unavailable)")); }
  const det=el("details"); det.append(el("summary",{},"reveal the judge's verdict"));
  const body=el("div"); det.append(body); card.append(det);
  det.addEventListener("toggle",()=>{ if(det.open && !body.dataset.done){ body.append(verdictHtml(pc.verdict)); body.dataset.done="1"; } });
  document.getElementById("list").append(card);
});
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rows = load()
    picked = sample(rows, min(args.n, len(rows)), args.seed)
    out = ROOT / "docs" / "validation" / "listen.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(picked), encoding="utf-8")
    lo = min(r["_overall"] for r in picked)
    hi = max(r["_overall"] for r in picked)
    print(f"Wrote {out} ({len(picked)} pieces, judge-overall {lo:.2f}..{hi:.2f})")


if __name__ == "__main__":
    main()
