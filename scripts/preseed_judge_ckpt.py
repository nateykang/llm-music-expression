#!/usr/bin/env python3
"""Pre-seed judge_allmodels_ckpt.json from the existing judge_allmodels_raw.json so
a re-run of the full judge only scores NEW pieces (appended batches) rather than
re-judging the whole corpus. Verifies index-alignment first; aborts if misaligned.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"

tasks = []
for bd in sorted(p for p in DATA.iterdir() if p.is_dir()):
    man = bd / "data.json"
    if not man.exists():
        continue
    for pc in json.loads(man.read_text(encoding="utf-8")).get("pieces", []):
        if pc.get("ok") and pc.get("prompt") == "free-form":
            tasks.append((pc["model"], pc.get("title", "")))

rawp = ROOT / "docs/analysis/judge_allmodels_raw.json"
raw = [p for p in json.loads(rawp.read_text(encoding="utf-8")) if p["prompt"] == "free-form"]
mism = sum(1 for i, p in enumerate(raw)
           if i >= len(tasks) or (p["model"], p.get("title", "")) != tasks[i])
print(f"full tasks: {len(tasks)} | existing verdicts: {len(raw)} | alignment mismatches: {mism}")
if mism:
    print("ABORT: misaligned — not pre-seeding (full run would be fresh).")
    raise SystemExit(1)

ckpt = {}
for ti, p in enumerate(raw):
    for jn, v in p["panel"].items():
        ckpt[f"{ti}|{jn}"] = v
(ROOT / "docs/analysis/judge_allmodels_ckpt.json").write_text(json.dumps(ckpt))
print(f"pre-seeded {len(ckpt)} verdicts; ~{len(tasks) * 9 - len(ckpt)} calls remain "
      f"({len(tasks) - len(raw)} new pieces)")
