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
    ap.add_argument("--only-missing", action="store_true",
                    help="only re-judge pieces gemini is currently absent from (keep existing verdicts)")
    args = ap.parse_args()

    tasks = pilot_tasks(200)
    cli = get_client(JUDGE)
    lock = threading.Lock()

    # Load the all-models raw json up front — needed for the alignment check and, in
    # --only-missing mode, to find which pieces gemini is currently absent from.
    rawp = ROOT / "docs/analysis/judge_allmodels_raw.json"
    raw = json.loads(rawp.read_text(encoding="utf-8"))
    raw_ff = [p for p in raw if p["prompt"] == "free-form"]
    mism = sum(1 for i, p in enumerate(raw_ff)
               if i >= len(tasks) or p["model"] != tasks[i][0]["model"]
               or p.get("title", "") != tasks[i][0].get("title", ""))
    if mism:
        print(f"ABORT: raw json not index-aligned with tasks ({mism} mismatches).")
        return

    # Resumable: each verdict is checkpointed as it completes.
    ckpt = ROOT / "docs/analysis/rejudge_gemini_ckpt.json"
    results = {int(k): v for k, v in json.loads(ckpt.read_text()).items()} if ckpt.exists() else {}

    if args.only_missing:
        target = [i for i, p in enumerate(raw_ff) if JUDGE not in p.get("panel", {})]
        print(f"--only-missing: {len(target)} pieces gemini is currently absent from")
    else:
        target = list(range(len(tasks)))
    todo = [i for i in target if i not in results]
    print(f"re-judging {JUDGE}: {len(todo)} to do, {len(results)} cached "
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
                if n % 10 == 0 or n == len(todo):
                    valid = sum(1 for x in results.values() if x)
                    print(f"  [{n}/{len(todo)}] {JUDGE} valid={valid}", flush=True)
    save()
    ok = sum(1 for x in results.values() if x)

    # merge: --only-missing fills the gaps (never drops existing); full replaces all
    old = sum(1 for p in raw_ff if JUDGE in p.get("panel", {}))
    if args.only_missing:
        for i in target:
            if results.get(i):
                raw_ff[i]["panel"][JUDGE] = results[i]
    else:
        for i, p in enumerate(raw_ff):
            if results.get(i):
                p["panel"][JUDGE] = results[i]
            else:
                p["panel"].pop(JUDGE, None)
    new = sum(1 for p in raw_ff if JUDGE in p.get("panel", {}))
    rawp.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    print(f"\n{JUDGE} coverage in raw json: {old} -> {new}  (fresh valid this run: {ok}/{len(todo)})")
    print("=== REJUDGE GEMINI COMPLETE ===")


if __name__ == "__main__":
    main()
