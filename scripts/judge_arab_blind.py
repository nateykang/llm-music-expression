#!/usr/bin/env python3
"""Re-judge the Arab-Andalusian arm with all quoted text stripped from the reps.

122/150 of the arm's note listings carry quoted Arabic titles/form names (e.g.
"Tawshiya Qaim Wa Nisf") inherited from the source MusicXML — a label leak no
other arm has. This run removes every quoted string so the arm is judged fully
blind; comparing with judge_human_raw.json gives a within-piece labeled-vs-blind
contrast. The 28 pieces with no quoted text are re-judged unchanged and act as
retest controls.

    python scripts/judge_arab_blind.py

Writes docs/analysis/judge_arab_blind_raw.json. Resumable (content-keyed ckpt).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from judge_human_corpora import PANEL, judge_rep  # noqa: E402
from llm_music.models import get_client  # noqa: E402

QUOTED = re.compile(r'\s*"[^"]*"')


def strip_quotes(rep: str) -> str:
    return QUOTED.sub("", rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    sample = json.loads((ROOT / "docs/analysis/human_corpora_sample.json")
                        .read_text(encoding="utf-8"))
    arab = [r for r in sample if r["arm"] == "arab_and"]
    for r in arab:
        r["rep_blind"] = strip_quotes(r["rep"])
    changed = sum(1 for r in arab if r["rep_blind"] != r["rep"])
    print(f"{len(arab)} pieces, {changed} had quoted text stripped", flush=True)

    analysis = ROOT / "docs/analysis"
    ckpt_path = analysis / "judge_arab_blind_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.exists() else {}
    lock = threading.Lock()
    clients = {j: get_client(j) for j in PANEL}
    jobs = [(r, j) for r in arab for j in PANEL if f"{r['id']}|{j}" not in ckpt]
    print(f"{len(jobs)} calls to do, {len(ckpt)} cached", flush=True)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.replace(ckpt_path)

    def work(job):
        r, j = job
        return f"{r['id']}|{j}", judge_rep(clients[j], r["rep_blind"])

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
    for r in arab:
        panel = {j: ckpt[f"{r['id']}|{j}"] for j in PANEL if ckpt.get(f"{r['id']}|{j}")}
        if panel:
            rows.append({"arm": r["arm"], "id": r["id"], "title": r["title"],
                         "key": r["key"], "mode": r["mode"],
                         "stripped": r["rep_blind"] != r["rep"], "panel": panel})
    out = analysis / "judge_arab_blind_raw.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {len(rows)} pieces → {out}")


if __name__ == "__main__":
    main()
