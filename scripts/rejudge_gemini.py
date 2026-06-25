#!/usr/bin/env python3
"""Re-judge gemini on the 200 pilot pieces under the new 600s timeout and merge the
fresh verdicts into judge_allmodels_raw.json — gemini was clipped to 145/200 by the
old 120s cap (it reasons 25-137s per judge call). Replaces gemini's panel entry on
each piece (drops it where the fresh call still fails) so coverage reflects a fair,
un-clipped run. Other judges are untouched.

Usage:  python scripts/rejudge_gemini.py [--workers 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from llm_music.judge import judge_piece  # noqa: E402
from llm_music.models import get_client  # noqa: E402

DATA = ROOT / "docs" / "data"
JUDGE = "gemini-2.5-pro"


def pilot_tasks(limit=200):
    tasks = []
    for bd in sorted(p for p in DATA.iterdir() if p.is_dir()):
        man = bd / "data.json"
        if not man.exists():
            continue
        for pc in json.loads(man.read_text(encoding="utf-8")).get("pieces", []):
            if pc.get("ok") and pc.get("prompt") == "free-form":
                tasks.append((pc, bd))
    return tasks[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    tasks = pilot_tasks(200)
    cli = get_client(JUDGE)
    lock = threading.Lock()
    results = {}
    done = ok = 0

    def work(i):
        pc, bd = tasks[i]
        return i, judge_piece(cli, pc, bd, include_note=False)

    print(f"re-judging {JUDGE} on {len(tasks)} pieces (600s timeout, {args.workers} workers)…",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, i) for i in range(len(tasks))]):
            i, v = fut.result()
            with lock:
                done += 1
                ok += bool(v)
                results[i] = v
                if done % 20 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] {JUDGE} valid={ok}", flush=True)

    # merge into raw json by index (raw is built in the same sorted-task order)
    rawp = ROOT / "docs/analysis/judge_allmodels_raw.json"
    raw = json.loads(rawp.read_text(encoding="utf-8"))
    raw_ff = [p for p in raw if p["prompt"] == "free-form"]
    mism = sum(1 for i, p in enumerate(raw_ff)
               if i >= len(tasks) or p["model"] != tasks[i][0]["model"]
               or p.get("title", "") != tasks[i][0].get("title", ""))
    print(f"alignment check: {mism} mismatches of {len(raw_ff)} (expect 0)")
    if mism:
        print("ABORT: raw json not index-aligned with tasks; not merging.")
        return

    old = sum(1 for p in raw_ff if JUDGE in p.get("panel", {}))
    for i, p in enumerate(raw_ff):
        v = results.get(i)
        if v:
            p["panel"][JUDGE] = v
        else:
            p["panel"].pop(JUDGE, None)
    new = sum(1 for p in raw_ff if JUDGE in p.get("panel", {}))
    rawp.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    print(f"\n{JUDGE} coverage in raw json: {old} -> {new}  (fresh valid: {ok}/{len(tasks)})")
    print("=== REJUDGE GEMINI COMPLETE ===")


if __name__ == "__main__":
    main()
