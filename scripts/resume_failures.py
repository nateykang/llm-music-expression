#!/usr/bin/env python3
"""Re-run only the RECOVERABLE failures in a batch, in place.

Recoverable = transient overloads + JSON-parse failures (e.g. the reasoning-model
empty-content bug, now fixed). Genuine sandbox crashes are left as real failures,
so the reliability metric stays honest. Updates the manifest entry for each cell.

Usage:  python scripts/resume_failures.py docs/data/<batch_dir> [--workers N]
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_music.generate import generate_piece  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.store import append_result, write_manifest  # noqa: E402

RECOVERABLE = re.compile(r"overload|could not parse JSON", re.I)


def main():
    batch = Path(sys.argv[1])
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    m = json.loads((batch / "data.json").read_text(encoding="utf-8"))
    pieces = m["pieces"]
    targets = [(i, p) for i, p in enumerate(pieces)
               if not p.get("ok") and RECOVERABLE.search(str(p.get("error", "")))]
    print(f"{len(targets)} recoverable failures to re-run (of {sum(1 for p in pieces if not p.get('ok'))} total)")
    if not targets:
        return
    clients = {mdl: get_client(mdl) for mdl in {p["model"] for _, p in targets}}
    lock = threading.Lock()
    scratch = Path(tempfile.mkdtemp(prefix="resume_"))

    def work(item):
        i, p = item
        wd = scratch / p["model"] / str(p.get("sample", 0)) / str(i)
        r = generate_piece(clients[p["model"]], p["prompt"], p["mode"], wd, bake_audio=False)
        return i, p, r

    recovered = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, t) for t in targets]):
            i, p, r = fut.result()
            with lock:
                pieces[i] = append_result(batch, r, sample=p.get("sample", 0))
                write_manifest(batch, m["timestamp"], m["models"], m["prompts"], pieces)
                recovered += r.ok
            print(f"  {p['model']:18} s{p.get('sample', 0):<2} -> {'ok' if r.ok else 'FAILED again'}")
    print(f"\nrecovered {recovered}/{len(targets)}")


if __name__ == "__main__":
    main()
