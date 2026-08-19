#!/usr/bin/env python3
"""Regenerate the sparse-codegen (no-toolkit) ablation and build a local viewer
that shows, per piece: the raw music21 code the model wrote, the pass/fail
status, the traceback for failures, and — for successes — the engraved score
(Verovio) and playable audio.

Standalone: does not touch the canonical pipeline. Output goes to
docs/experiments/toolkit_ablation/ (gitignored). Pieces are freshly sampled, so
they differ run-to-run (LLM calls aren't deterministic).

    python scripts/toolkit_ablation_viewer.py
"""

from __future__ import annotations

import html
import json
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from llm_music.generate import SYSTEM_TEMPLATE, _load_prompt, _variant_row  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.modes import MODES  # noqa: E402
from llm_music.modes._common import extract_json  # noqa: E402
from llm_music.render import midi_to_audio  # noqa: E402
from llm_music.sandbox import run_music21_code  # noqa: E402

MODELS = ["fable-5", "gpt-4.1", "deepseek-v4-pro"]
SAMPLES = 3
MODE = "codegen-sparse"
PROMPT = "express-yourself"
SYSTEM_PROMPT = SYSTEM_TEMPLATE.format(variant=_variant_row(PROMPT)["instruction"])
OUT = ROOT / "docs/experiments/toolkit_ablation"


def one_piece(client, base_user, work_dir, max_attempts=5):
    """Return dict with final code/status plus the list of FAILED attempts
    (each with the buggy code and its traceback) so recoveries are visible."""
    mode_mod = MODES[MODE]
    prior_error = None
    last_code, attempts = "", 0
    fails = []  # {code, error} for each attempt that did not run
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        user = mode_mod.build_user_prompt(base_user, prior_error)
        try:
            response = client.complete(SYSTEM_PROMPT, user)
        except Exception as e:
            prior_error = f"API error: {e}"
            fails.append({"code": last_code, "error": prior_error})
            continue
        try:
            obj = extract_json(response)
            last_code = obj.get("code") or last_code
        except ValueError as e:
            prior_error = f"could not parse JSON response: {e}"
            fails.append({"code": last_code, "error": prior_error})
            continue
        code = obj.get("code", "")
        if not isinstance(code, str) or not code.strip():
            prior_error = "response JSON missing non-empty 'code' field"
            fails.append({"code": code, "error": prior_error})
            continue
        last_code = code
        sb = run_music21_code(code, work_dir)
        if sb.ok:
            return {"ok": True, "code": code, "attempts": attempt, "fails": fails,
                    "title": obj.get("title", "Untitled"),
                    "short": obj.get("short_description", ""),
                    "long": obj.get("long_description", ""),
                    "midi": sb.midi_path, "xml": sb.musicxml_path, "error": None}
        prior_error = sb.error
        fails.append({"code": code, "error": sb.error})
    return {"ok": False, "code": last_code, "attempts": attempts, "fails": fails[:-1],
            "title": "", "short": "", "long": "",
            "midi": None, "xml": None, "error": prior_error}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "scores").mkdir(parents=True)
    (OUT / "audio").mkdir(parents=True)
    base_user = _load_prompt(MODES[MODE])
    (OUT / "prompt.txt").write_text(base_user, encoding="utf-8")

    pieces = []
    for model in MODELS:
        client = get_client(model)
        for s in range(SAMPLES):
            print(f"generating {model} #{s} …", flush=True)
            wd = OUT / f"_work_{model}_{s}"
            wd.mkdir(exist_ok=True)
            r = one_piece(client, base_user, wd)
            slug = f"{model}_s{s}"
            if r["ok"] and r["xml"]:
                xml_dst = OUT / "scores" / f"{slug}.musicxml"
                shutil.copy(r["xml"], xml_dst)
                r["xml_rel"] = f"scores/{slug}.musicxml"
                if r["midi"]:
                    mp3 = OUT / "audio" / f"{slug}.mp3"
                    if midi_to_audio(r["midi"], mp3):
                        r["audio_rel"] = f"audio/{slug}.mp3"
            r["model"], r["sample"] = model, s
            pieces.append(r)
            shutil.rmtree(wd, ignore_errors=True)
            print(f"  {'ok' if r['ok'] else 'FAIL'} ({r['attempts']} att)"
                  f"{'' if r['ok'] else ': ' + str(r['error'])[:70]}", flush=True)

    build_html(pieces, base_user)
    n_ok = sum(p["ok"] for p in pieces)
    print(f"\n{n_ok}/{len(pieces)} succeeded → {OUT / 'index.html'}")


def build_html(pieces, base_user):
    esc = html.escape
    blocks = []
    for p in pieces:
        badge = ("<span class='ok'>ran ✓</span>" if p["ok"]
                 else "<span class='fail'>failed ✗</span>")
        head = (f"<h2>{esc(p['model'])} · sample {p['sample']} {badge} "
                f"<span class='att'>{p['attempts']} attempt"
                f"{'s' if p['attempts'] != 1 else ''}</span></h2>")
        meta = ""
        if p["ok"]:
            meta = (f"<p class='title'>“{esc(p['title'])}”</p>"
                    f"<p class='desc'>{esc(p['short'])}</p>")
            if p.get("audio_rel"):
                meta += f"<audio controls src='{p['audio_rel']}'></audio>"
            if p.get("xml_rel"):
                n = len([b for b in blocks])
                meta += f"<div class='score' data-xml='{p['xml_rel']}'>engraving…</div>"
        else:
            meta = ("<p class='errlabel'>traceback (final attempt):</p>"
                    f"<pre class='err'>{esc(str(p['error']))}</pre>")
        label = ("final working code" if p["ok"] else "last (failed) code")
        code = (f"<details{' open' if not p['ok'] else ''}>"
                f"<summary>{label} ({len(p['code'].splitlines())} lines)</summary>"
                f"<pre class='code'>{esc(p['code'])}</pre></details>")
        stumbles = ""
        for i, fa in enumerate(p.get("fails", []), 1):
            errline = str(fa["error"]).strip().splitlines()
            errline = errline[-1] if errline else "?"
            stumbles += (f"<details class='stumble'><summary>failed attempt {i}: "
                         f"{esc(errline[:90])}</summary>"
                         f"<pre class='err'>{esc(str(fa['error']))}</pre>"
                         f"<pre class='code'>{esc(fa['code'])}</pre></details>")
        if stumbles:
            stumbles = (f"<p class='stumblelabel'>{len(p.get('fails', []))} failed "
                        f"attempt(s) before this — the traceback is fed back and the "
                        f"model retries:</p>" + stumbles)
        blocks.append(f"<section class='piece'>{head}{meta}{code}{stumbles}</section>")

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Toolkit ablation — sparse codegen outputs</title>
<script src="https://www.verovio.org/javascript/latest/verovio-toolkit-wasm.js" defer></script>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
        margin: 0 auto; padding: 2rem 1rem; background: #faf9f6; color: #1c1c1c; }}
 h1 {{ font-size: 1.4rem; }}
 .lead {{ color: #666; font-size: .92rem; }}
 .piece {{ border-top: 1px solid #ddd; padding-top: 1.2rem; margin-top: 1.6rem; }}
 h2 {{ font-size: 1.05rem; }}
 .ok {{ color: #2a7; font-weight: 600; }} .fail {{ color: #c33; font-weight: 600; }}
 .att {{ color: #999; font-weight: 400; font-size: .85rem; }}
 .title {{ font-weight: 600; margin: .3rem 0 0; }}
 .desc {{ color: #555; margin: .2rem 0 .6rem; font-style: italic; }}
 audio {{ width: 100%; margin: .4rem 0; }}
 .score {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
          padding: 6px; overflow-x: auto; }}
 .score svg {{ max-width: 100%; height: auto; }}
 details {{ margin: .6rem 0; }} summary {{ cursor: pointer; color: #556; font-size: .9rem; }}
 pre.code {{ background: #1e1e24; color: #d6d6dd; padding: 12px; border-radius: 6px;
            overflow-x: auto; font-size: 12px; line-height: 1.45; }}
 pre.err {{ background: #2a1414; color: #ffb4b4; padding: 12px; border-radius: 6px;
           overflow-x: auto; font-size: 12px; }}
 .errlabel {{ color: #c33; font-size: .85rem; margin: .4rem 0 .2rem; }}
 .stumblelabel {{ color: #a60; font-size: .85rem; margin: .8rem 0 .2rem; }}
 details.stumble {{ margin: .3rem 0; padding-left: .6rem; border-left: 2px solid #e0b070; }}
 details.stumble summary {{ color: #a60; }}
</style></head><body>
<h1>Toolkit ablation — sparse code-gen outputs</h1>
<p class="lead">The three models composing in <code>codegen-sparse</code>: the
canonical code-gen prompt <b>minus</b> the 104-line music21 toolkit doc. For each
of {SAMPLES} samples: the code they wrote, whether it executed, the traceback if
not, and the engraved score + audio if it did. Freshly generated — differs from
the earlier analyzed batch. <a href="prompt.txt">The exact prompt sent.</a></p>
{"".join(blocks)}
<script>
document.addEventListener("DOMContentLoaded", () => {{
  verovio.module.onRuntimeInitialized = async () => {{
    const tk = new verovio.toolkit();
    tk.setOptions({{ scale: 35, pageWidth: 2400, adjustPageHeight: true }});
    for (const el of document.querySelectorAll(".score")) {{
      try {{
        const xml = await (await fetch(el.dataset.xml)).text();
        tk.loadData(xml);
        el.innerHTML = tk.renderToSVG(1);
        const pages = tk.getPageCount();
        if (pages > 1) el.insertAdjacentHTML("beforeend",
          `<p style='color:#999;font-size:.8rem'>page 1 of ${{pages}}</p>`);
      }} catch (e) {{ el.textContent = "score render failed: " + e; }}
    }}
  }};
}});
</script>
</body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
