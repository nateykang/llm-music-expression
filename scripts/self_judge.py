#!/usr/bin/env python3
"""Measure self-enhancement bias: let each panelist judge its OWN pieces (blind),
then compare its self-scores to how the OTHER panelists scored the same pieces.

The blind pilot excluded self-judgments by design; this fills only those gaps
(panelist-authored pieces, scored by their own author) — a small, cheap run — and
reports the per-model self-preference delta = mean(self) - mean(others) on the same
pieces. Positive = the model rates its own work higher than peers do.

Usage:  python scripts/self_judge.py [--limit 200] [--workers 3]
Writes: docs/analysis/self_judge.json  (per-piece self vs others)
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
from llm_music.judge import QUALITY_KEYS, judge_piece  # noqa: E402
from llm_music.models import get_client  # noqa: E402

PANEL = ["gpt-5.5", "gemini-2.5-pro", "opus-4.8"]
DATA = ROOT / "docs" / "data"


def pilot_tasks(prompt: str, limit: int):
    """Reproduce the exact 200-piece pilot selection (sorted batches, ok pieces)."""
    tasks = []
    for bd in sorted(p for p in DATA.iterdir() if p.is_dir()):
        man = bd / "data.json"
        if not man.exists():
            continue
        for pc in json.loads(man.read_text(encoding="utf-8")).get("pieces", []):
            if pc.get("ok") and (not prompt or pc.get("prompt") == prompt):
                tasks.append((pc, bd))
    return tasks[:limit]


def mean_quality(verdict: dict) -> float | None:
    vals = [verdict[k]["score"] for k in QUALITY_KEYS if k in verdict]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--prompt", default="free-form")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    tasks = pilot_tasks(args.prompt, args.limit)
    # the pilot's "others" scores, keyed by piece identity
    raw = json.loads((ROOT / "docs/analysis/judge_raw.json").read_text(encoding="utf-8"))
    others = {(p["model"], p["prompt"], p.get("mode", ""), p.get("title", "")): p["panel"] for p in raw}

    # only panelist-authored pieces need a self-judgment
    jobs = [(pc, bd) for pc, bd in tasks if pc["model"] in PANEL]
    print(f"{len(jobs)} panelist-authored pieces to self-judge (of {len(tasks)} pilot pieces)")
    clients = {m: get_client(m) for m in PANEL}
    lock = threading.Lock()
    rows, done = [], 0

    def work(item):
        pc, bd = item
        return pc, judge_piece(clients[pc["model"]], pc, bd, include_note=False)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            pc, self_v = fut.result()
            with lock:
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(f"  [{done}/{len(jobs)}]", flush=True)
            if not self_v:
                continue
            key = (pc["model"], pc["prompt"], pc.get("mode", ""), pc.get("title", ""))
            panel = others.get(key, {})
            peer = [mean_quality(v) for jn, v in panel.items() if jn != pc["model"]]
            peer = [x for x in peer if x is not None]
            sq = mean_quality(self_v)
            if sq is None or not peer:
                continue
            rows.append({"model": pc["model"], "title": pc["title"],
                         "self": round(sq, 3), "others": round(sum(peer) / len(peer), 3),
                         "self_label": self_v.get("emotion_label", ""),
                         "self_dims": {k: self_v[k]["score"] for k in QUALITY_KEYS if k in self_v}})

    (ROOT / "docs/analysis/self_judge.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print("\n=== SELF-ENHANCEMENT: self-score vs peers on the same pieces ===")
    print(f"{'model':18} {'n':>3} {'self':>6} {'peers':>6} {'delta':>7}")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    for m in PANEL:
        rs = by.get(m, [])
        if not rs:
            print(f"{m:18} {'0':>3}  (no pieces)")
            continue
        s = sum(r["self"] for r in rs) / len(rs)
        o = sum(r["others"] for r in rs) / len(rs)
        print(f"{m:18} {len(rs):>3} {s:>6.2f} {o:>6.2f} {s - o:>+7.2f}")
    print("\n(+delta = the model rates its own pieces HIGHER than peers do = self-enhancement)")


if __name__ == "__main__":
    main()
