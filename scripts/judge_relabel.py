#!/usr/bin/env python3
"""Re-judge the relative-key relabeled pieces (built by relabel_keys.py).

Every variant is note-identical to its original — only the K: label flips to
the relative key (C <-> Am), so each judge's score shift vs its verdict on the
original (already in judge_allmodels_raw.json) measures pure LABEL-driven mode
bias: how much the judge trusts the declared key over the notes.

    python scripts/judge_relabel.py --limit 100       # balanced pilot
    python scripts/judge_relabel.py                   # full set (~370 x 10 judges)

Resumable (content-keyed checkpoint). Writes docs/analysis/judge_relabel_raw.json:
  [{model, mode, title, sample, batch, orig_key, new_key, panel: {judge: verdict}}]
"""

from __future__ import annotations

import argparse
import json
import random
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

from llm_music.judge import judge_piece  # noqa: E402
from llm_music.models import get_client  # noqa: E402

PANEL = ["fable-5", "gpt-5.5", "gemini-2.5-pro", "opus-4.8", "sonnet-4.6",
         "gpt-4.1", "grok-4.3", "deepseek-v4-pro", "qwen3-max", "llama-4-maverick"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot on the first N pieces of a seeded, mode-balanced order")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--judges", default=",".join(PANEL))
    args = ap.parse_args()
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    rows = json.loads((ROOT / "docs/analysis/relabel_experiment.json").read_text(encoding="utf-8"))
    # deterministic mode-balanced order so --limit pilots stay balanced
    rng = random.Random(0)
    minors = [r for r in rows if r["orig_key"].endswith("m")]
    majors = [r for r in rows if not r["orig_key"].endswith("m")]
    rng.shuffle(minors), rng.shuffle(majors)
    ordered = [x for pair in zip(minors, majors) for x in pair]
    ordered += minors[len(majors):] + majors[len(minors):]
    if args.limit:
        ordered = ordered[:args.limit]

    def task_key(r, j):
        return f"{r['batch']}|{r['model']}|{r['mode']}|{r['sample']}|{j}"

    analysis = ROOT / "docs/analysis"
    ckpt_path = analysis / "judge_relabel_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}
    lock = threading.Lock()
    clients = {j: get_client(j) for j in judges}

    jobs = [(r, j) for r in ordered for j in judges if task_key(r, j) not in ckpt]
    print(f"{len(ordered)} relabeled pieces × {len(judges)} judges: "
          f"{len(jobs)} calls to do, {len(ckpt)} cached", flush=True)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.replace(ckpt_path)

    def work(job):
        r, j = job
        # a piece dict as judge_piece expects: the relabeled ABC stands in for
        # the original; batch_dir only matters for the instruments header.
        piece = {"model": r["model"], "prompt": "free-form", "mode": r["mode"],
                 "title": r["title"], "sample": r["sample"], "abc": r["abc"]}
        verdict = judge_piece(clients[j], piece, ROOT / "docs/data" / r["batch"])
        return task_key(r, j), verdict

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

    out_rows = []
    for r in ordered:
        panel = {j: ckpt[task_key(r, j)] for j in judges
                 if ckpt.get(task_key(r, j))}
        if panel:
            out_rows.append({k: r[k] for k in
                             ("model", "mode", "title", "sample", "batch",
                              "orig_key", "new_key")} | {"panel": panel})
    out = analysis / "judge_relabel_raw.json"
    out.write_text(json.dumps(out_rows, indent=1), encoding="utf-8")
    print(f"\nWrote {len(out_rows)} pieces → {out}")
    print("Compare against judge_allmodels_raw.json (same pieces, original labels) "
          "for the paired label-shift analysis.")


if __name__ == "__main__":
    main()
