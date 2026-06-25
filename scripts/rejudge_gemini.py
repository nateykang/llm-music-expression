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

    # Resumable: each verdict is checkpointed as it completes, so an interrupted run
    # (sleep / shutdown) picks up where it left off instead of losing everything.
    ckpt = ROOT / "docs/analysis/rejudge_gemini_ckpt.json"
    results = {int(k): v for k, v in json.loads(ckpt.read_text()).items()} if ckpt.exists() else {}
    todo = [i for i in range(len(tasks)) if i not in results]
    print(f"re-judging {JUDGE}: {len(todo)} to do, {len(results)} cached from checkpoint "
          f"(600s timeout, {args.workers} workers)…", flush=True)

    def save():
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(k): v for k, v in results.items()}))
        tmp.replace(ckpt)

    def work(i):
        return i, judge_piece(cli, tasks[i][0], tasks[i][1], include_note=False)

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, i) for i in todo]):
            i, v = fut.result()
            with lock:
                results[i] = v
                n += 1
                if n % 5 == 0 or n == len(todo):
                    save()
                if n % 20 == 0 or n == len(todo):
                    valid = sum(1 for x in results.values() if x)
                    print(f"  [{len(results)}/{len(tasks)}] {JUDGE} valid={valid}", flush=True)
    save()
    ok = sum(1 for x in results.values() if x)

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
    ckpt.unlink(missing_ok=True)  # merged successfully — clear the checkpoint
    print(f"\n{JUDGE} coverage in raw json: {old} -> {new}  (fresh valid: {ok}/{len(tasks)})")
    print("=== REJUDGE GEMINI COMPLETE ===")


if __name__ == "__main__":
    main()
