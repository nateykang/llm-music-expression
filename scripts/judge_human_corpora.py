#!/usr/bin/env python3
"""Judge the human-corpora sample (built by sample_human_corpora.py) with the
extended rubric: the ORIGINAL dimensions unchanged (ecological validity +
comparability with every prior run) plus four additions probing preference and
its mechanism, and a free-text origin guess as a manipulation check.

  original: coherence harmony rhythm structure melody emotion creativity
            naturalness + valence arousal + emotion_label
  added:    enjoyment interest beauty familiarity + origin_guess (unscored)

    python scripts/judge_human_corpora.py --limit 20    # pilot, arm-interleaved
    python scripts/judge_human_corpora.py               # full (~800 x 10 judges)

Resumable (content-keyed checkpoint). Writes docs/analysis/judge_human_raw.json:
  [{arm, id, title, key, mode, panel: {judge: verdict}}]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from llm_music.judge import (  # noqa: E402
    AFFECT, QUALITY, _EMOTION_ALIASES, EMOTION_LABELS, _extract_json,
)
from llm_music.models import get_client  # noqa: E402
from llm_music.retry import backoff_sleep, is_retryable  # noqa: E402

PANEL = ["fable-5", "gpt-5.5", "gemini-2.5-pro", "opus-4.8", "sonnet-4.6",
         "gpt-4.1", "grok-4.3", "deepseek-v4-pro", "qwen3-max", "llama-4-maverick"]

# Additions: hedonic preference + the familiarity mediator. Same anchored,
# reason-before-score style as the original rubric.
EXTRA = [
    ("enjoyment", "Enjoyment",
     "How much did you enjoy this piece?",
     "did not enjoy it", "thoroughly enjoyed it"),
    ("interest", "Interest",
     "How interesting did you find it — did it hold your attention?",
     "dull, attention drifts", "gripping throughout"),
    ("beauty", "Beauty",
     "How beautiful is this piece?",
     "not beautiful", "strikingly beautiful"),
    ("familiarity", "Familiarity",
     "How familiar is this piece's musical style or idiom to you?",
     "completely unfamiliar idiom", "very familiar idiom"),
]
ALL_ITEMS = QUALITY + AFFECT + EXTRA
ALL_KEYS = [k for k, *_ in ALL_ITEMS]

REP_KIND = ("a note listing (key/time/tempo header, then per part — labelled "
            "with its instrument — per bar: Pitch+octave/duration with rests, "
            "dynamics [p]/[f], articulations and technical directions)")

SYSTEM = (
    "You are an expert music critic evaluating short pieces presented in symbolic "
    "notation (a note-by-note listing). Judge ONLY what you can perceive from the "
    "notes — melodic line, rhythm, form, texture, and emotional character. Do not "
    "reward length. Be calibrated and critical: on each 1-5 dimension, 3 = competent "
    "but unremarkable, 5 = genuinely excellent, 1 = a clear failure. For every "
    "dimension write a one-sentence justification and THEN an integer 1-5 using the "
    "anchors. Also name the single dominant emotional character, and give your best "
    "guess at the piece's tradition or origin. Return ONLY one valid JSON object, "
    "no prose."
)


def build_user(rep_text: str) -> str:
    rubric = "\n".join(f"- {k} ({label}): {q} [1 = {lo}; 5 = {hi}]"
                       for k, label, q, lo, hi in ALL_ITEMS)
    schema = ", ".join(f'"{k}": {{"reason": "...", "score": 1-5}}' for k in ALL_KEYS)
    return (
        f"THE MUSIC ({REP_KIND}):\n{rep_text}\n\n"
        f"Rate the piece on each dimension:\n{rubric}\n\n"
        f"Also choose the single dominant emotional character from EXACTLY this list: "
        f"{', '.join(EMOTION_LABELS)}.\n"
        f"Also give your best guess at the piece's tradition/region/era of origin "
        f"(one short phrase; guess even if unsure).\n\n"
        f"Return ONLY this JSON object (integer scores 1-5):\n"
        f'{{{schema}, "emotion_label": "<one label>", "origin_guess": "<short phrase>"}}'
    )


def judge_rep(client, rep_text: str, attempts: int = 3):
    user = build_user(rep_text)
    for a in range(attempts):
        try:
            raw = client.complete(SYSTEM, user, json_mode=True)
            obj = _extract_json(raw)
        except Exception as e:
            if not is_retryable(e):
                print(f"  {client.name}: permanent error, skipping: {e}", flush=True)
                return None
            print(f"  {client.name}: attempt {a + 1}/{attempts}: {e}", flush=True)
            obj = None
        if obj:
            out = {}
            for k in ALL_KEYS:
                v = obj.get(k)
                if isinstance(v, dict) and "score" in v:
                    try:
                        out[k] = {"score": float(v["score"]),
                                  "reason": str(v.get("reason", ""))[:300]}
                    except (TypeError, ValueError):
                        pass
                elif isinstance(v, (int, float)):
                    out[k] = {"score": float(v), "reason": ""}
            lbl = str(obj.get("emotion_label", "")).strip().lower()
            out_lbl = _EMOTION_ALIASES.get(lbl, lbl)
            if out_lbl:
                out["emotion_label"] = out_lbl
            og = str(obj.get("origin_guess", "")).strip()
            if og:
                out["origin_guess"] = og[:120]
            if out:
                return out
        if a < attempts - 1:
            backoff_sleep(a, cap=8.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot on the first N pieces of an arm-interleaved order")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--judges", default=",".join(PANEL))
    args = ap.parse_args()
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    sample = json.loads((ROOT / "docs/analysis/human_corpora_sample.json")
                        .read_text(encoding="utf-8"))
    # arm-interleaved order so --limit pilots cover every arm
    by_arm: dict[str, list] = {}
    for r in sample:
        by_arm.setdefault(r["arm"], []).append(r)
    ordered, i = [], 0
    while any(by_arm.values()):
        for arm in list(by_arm):
            if by_arm[arm]:
                ordered.append(by_arm[arm].pop(0))
        i += 1
    if args.limit:
        ordered = ordered[:args.limit]

    analysis = ROOT / "docs/analysis"
    ckpt_path = analysis / "judge_human_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}
    lock = threading.Lock()
    clients = {j: get_client(j) for j in judges}
    jobs = [(r, j) for r in ordered for j in judges if f"{r['id']}|{j}" not in ckpt]
    print(f"{len(ordered)} pieces × {len(judges)} judges: {len(jobs)} calls to do, "
          f"{len(ckpt)} cached", flush=True)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.replace(ckpt_path)

    def work(job):
        r, j = job
        return f"{r['id']}|{j}", judge_rep(clients[j], r["rep"])

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for fut in as_completed([ex.submit(work, jb) for jb in jobs]):
            k, v = fut.result()
            with lock:
                ckpt[k] = v
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    save()
                if done % 100 == 0 or done == len(jobs):
                    print(f"  [{done}/{len(jobs)}]", flush=True)
    save()

    rows = []
    for r in ordered:
        panel = {j: ckpt[f"{r['id']}|{j}"] for j in judges if ckpt.get(f"{r['id']}|{j}")}
        if panel:
            rows.append({"arm": r["arm"], "id": r["id"], "title": r["title"],
                         "key": r["key"], "mode": r["mode"], "panel": panel})
    out = analysis / "judge_human_raw.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {len(rows)} pieces → {out}")


if __name__ == "__main__":
    main()
