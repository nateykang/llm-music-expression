#!/usr/bin/env python3
"""Key/mode-bias experiment on REAL human music: the full 10-model judge panel
rates the 371 Bach chorales (music21's canonical corpus), blind, from the same
note-listing representation used for code-gen pieces.

Why: on the generated corpus, each judge's preference for major-mode music
tracks its own major-writing rate at r=+0.69 (within-author contrasts). The
chorales test whether that taste transfers to human music with naturally
varied keys (195 major / 176 minor) and zero LLM authorship.

    python scripts/judge_bach.py --limit 20          # pilot: 20 chorales x 10 judges
    python scripts/judge_bach.py                     # full run: 371 x 10 = 3,710 calls
    python scripts/judge_bach.py --judges fable-5    # subset of the panel

Resumable via a content-keyed checkpoint (chorale index + title + judge), same
scheme as judge_corpus. Writes docs/analysis/judge_bach_raw.json:
  [{"index": i, "title": ..., "key": "G major", "panel": {judge: verdict}}]
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
    _extract_json, _score_to_text, _system, build_user,
)
from llm_music.models import get_client  # noqa: E402
from llm_music.retry import backoff_sleep, is_retryable  # noqa: E402

PANEL = ["fable-5", "gpt-5.5", "gemini-2.5-pro", "opus-4.8", "sonnet-4.6",
         "gpt-4.1", "grok-4.3", "deepseek-v4-pro", "qwen3-max", "llama-4-maverick"]

REP_KIND = ("a note listing (key/time/tempo header, then per part — labelled "
            "with its instrument — per bar: Pitch+octave/duration with rests, "
            "dynamics [p]/[f], articulations and technical directions)")


def judge_text(client, rep_text: str, attempts: int = 3):
    """One judge's verdict on one already-rendered representation."""
    user = build_user({"long_description": ""}, REP_KIND, rep_text, include_note=False)
    for a in range(attempts):
        try:
            raw = client.complete(_system(False), user, json_mode=True)
            obj = _extract_json(raw)
        except Exception as e:
            if not is_retryable(e):
                print(f"  {client.name}: permanent error, skipping: {e}", flush=True)
                return None
            print(f"  {client.name}: attempt {a + 1}/{attempts}: {e}", flush=True)
            obj = None
        if obj:
            # normalize exactly like judge_piece does
            from llm_music.judge import AFFECT_KEYS, QUALITY_KEYS, _EMOTION_ALIASES
            out = {}
            for k in QUALITY_KEYS + AFFECT_KEYS:
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
            lbl = _EMOTION_ALIASES.get(lbl, lbl)
            if lbl:
                out["emotion_label"] = lbl
            if out:
                return out
        if a < attempts - 1:
            backoff_sleep(a, cap=8.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N chorales")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--judges", default=",".join(PANEL))
    args = ap.parse_args()
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    from music21.corpus import chorales

    print("rendering chorales to the blind note-listing representation…", flush=True)
    items = []  # (index, title, key_str, rep_text)
    for i, score in enumerate(chorales.Iterator()):
        if args.limit and len(items) >= args.limit:
            break
        try:
            k = score.analyze("key")
            rep = _score_to_text(score, label=f"chorale {i}")
        except Exception as e:
            print(f"  chorale {i}: skipped ({e})", flush=True)
            continue
        title = (score.metadata.title if score.metadata and score.metadata.title
                 else f"chorale-{i}")
        items.append((i, title, f"{k.tonic.name} {k.mode}", rep))
    print(f"{len(items)} chorales rendered; keys balanced "
          f"({sum(1 for it in items if it[2].endswith('major'))} major / "
          f"{sum(1 for it in items if it[2].endswith('minor'))} minor)", flush=True)

    analysis = ROOT / "docs/analysis"
    ckpt_path = analysis / "judge_bach_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}
    lock = threading.Lock()

    clients = {j: get_client(j) for j in judges}
    jobs = [(i, t, key, rep, j) for (i, t, key, rep) in items for j in judges
            if f"{i}|{t}|{j}" not in ckpt]
    n_cached = sum(1 for (i, t, _, _) in items for j in judges if f"{i}|{t}|{j}" in ckpt)
    print(f"{len(jobs)} calls to do, {n_cached} cached "
          f"({args.workers} workers, panel={judges})", flush=True)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.replace(ckpt_path)

    def work(job):
        i, t, key, rep, j = job
        return f"{i}|{t}|{j}", judge_text(clients[j], rep)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for fut in as_completed([ex.submit(work, jb) for jb in jobs]):
            k, verdict = fut.result()
            with lock:
                ckpt[k] = verdict
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    save()
                if done % 100 == 0 or done == len(jobs):
                    print(f"  [{done}/{len(jobs)}]", flush=True)
    save()

    rows = []
    for i, t, key, _rep in items:
        panel = {}
        for j in judges:
            v = ckpt.get(f"{i}|{t}|{j}")
            if v:
                panel[j] = v
        if panel:
            rows.append({"index": i, "title": t, "key": key, "panel": panel})
    out = analysis / "judge_bach_raw.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ok_calls = sum(len(r["panel"]) for r in rows)
    print(f"\nWrote {len(rows)} chorales ({ok_calls} verdicts) → {out}")
    print("Checkpoint kept for incremental extension; delete "
          f"{ckpt_path.name} once you're satisfied the run is complete.")


if __name__ == "__main__":
    main()
