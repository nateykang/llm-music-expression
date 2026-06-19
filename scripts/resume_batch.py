#!/usr/bin/env python3
"""Finish a partially-generated batch: generate only the missing (model, prompt)
cells and append them to the existing batch folder. Usage:

    python scripts/resume_batch.py docs/data/<batch_dir>

The batch's mode is read from its existing pieces. Safe to re-run.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_music.generate import generate_piece  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.store import append_result, write_manifest  # noqa: E402


def resume(batch: Path) -> None:
    m = json.loads((batch / "data.json").read_text(encoding="utf-8"))
    models, prompts, ts = m["models"], m["prompts"], m["timestamp"]
    mode = m["pieces"][0]["mode"] if m["pieces"] else "abc"
    have = {(p["model"], p["prompt"]) for p in m["pieces"]}
    entries = list(m["pieces"])
    todo = [(mo, pr) for mo in models for pr in prompts if (mo, pr) not in have]
    print(f"resuming {batch.name}: {len(have)} done, {len(todo)} to go ({mode})")
    with tempfile.TemporaryDirectory(prefix="llm_resume_") as scratch:
        client = {}
        for mo, pr in todo:
            client.setdefault(mo, get_client(mo))
            print(f"  • {mo} × {pr} …", end="", flush=True)
            r = generate_piece(client[mo], pr, mode, Path(scratch) / mo / pr, max_attempts=5)
            print(f" {'ok: ' + r.title if r.ok else 'FAILED: ' + (r.error or '')[:60]}")
            entries.append(append_result(batch, r))
            write_manifest(batch, ts, models, prompts, entries)
    print(f"done. total {len(entries)}/{len(models) * len(prompts)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    resume(Path(sys.argv[1]))
