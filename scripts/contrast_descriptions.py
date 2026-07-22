#!/usr/bin/env python3
"""What does en route say that post hoc does not (and vice versa)? Per piece,
one rater sees both description arms as anonymous texts A and B (assignment
randomized per piece, deterministically from the piece id) and lists the
substantive statements unique to each, tagged with a fixed category taxonomy.
Aggregated per-arm category rates come out the other end.

    python scripts/contrast_descriptions.py --limit 4   # smoke test
    python scripts/contrast_descriptions.py             # full corpus

Resumable (content-keyed checkpoint). Writes docs/analysis/description_contrast.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from description_corpus import load_pieces, piece_id  # noqa: E402
from llm_music.judge import _extract_json  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.retry import backoff_sleep, is_retryable  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"

CATEGORIES = [
    "intent_or_goal", "process_narrative", "emotional_self_attribution",
    "evaluative_praise", "technical_structure", "programmatic_imagery",
    "admission_of_limitation", "other",
]

SYSTEM = (
    "You compare two descriptions of the same piece of music, labeled A and B. "
    "You never see or hear the music. Identify what each text substantively "
    "says that the other does not — a statement is substantive if it adds a "
    "claim about the music, the writer's intent or process, its quality, or "
    "its emotional content; ignore pure rephrasings of shared content. Return "
    "ONLY one valid JSON object."
)


def build_user(text_a: str, text_b: str) -> str:
    cats = ", ".join(CATEGORIES)
    return (
        f"DESCRIPTION A:\n{text_a}\n\n"
        f"DESCRIPTION B:\n{text_b}\n\n"
        "List the substantive statements unique to each description. Tag every "
        f"statement with exactly one category from: {cats}.\n\n"
        'Return ONLY this JSON object:\n'
        '{"only_a": [{"statement": "...", "category": "..."}], '
        '"only_b": [{"statement": "...", "category": "..."}]}'
    )


def full_text(short: str, long: str) -> str:
    return f"{short}\n{long}"


def a_is_enroute(pid: str) -> bool:
    """Stable per-piece randomization of which arm is shown as A."""
    return hashlib.md5(pid.encode()).digest()[0] % 2 == 0


def _items(lst) -> list[dict] | None:
    if not isinstance(lst, list):
        return None
    out = []
    for it in lst:
        if not isinstance(it, dict) or not isinstance(it.get("statement"), str):
            return None
        cat = it.get("category")
        out.append({"statement": it["statement"],
                    "category": cat if cat in CATEGORIES else "other"})
    return out


def contrast_one(client, piece: dict, attempts: int = 3) -> dict | None:
    ind = piece["independent_description"]
    enroute = full_text(piece["short_description"], piece["long_description"])
    posthoc = full_text(ind["short_description"], ind["long_description"])
    a_enroute = a_is_enroute(piece_id(piece))
    text_a, text_b = (enroute, posthoc) if a_enroute else (posthoc, enroute)
    user = build_user(text_a, text_b)
    for a in range(attempts):
        try:
            raw = client.complete(SYSTEM, user, json_mode=True)
            obj = _extract_json(raw, expect_keys=["only_a", "only_b"])
            if obj is not None:
                only_a, only_b = _items(obj.get("only_a")), _items(obj.get("only_b"))
                if only_a is not None and only_b is not None:
                    return {
                        "only_enroute": only_a if a_enroute else only_b,
                        "only_posthoc": only_b if a_enroute else only_a,
                    }
        except Exception as e:
            if not is_retryable(e):
                print(f"  permanent error, skipping {piece_id(piece)}: {e}",
                      flush=True)
                return None
            print(f"  attempt {a + 1}/{attempts} {piece_id(piece)}: {e}", flush=True)
        if a < attempts - 1:
            backoff_sleep(a, cap=8.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="fable-5")
    args = ap.parse_args()

    pieces = [p for p in load_pieces(ROOT)
              if p.get("independent_description", {})
              and p["independent_description"].get("short_description")]
    if args.limit:
        pieces = pieces[: args.limit]
    ckpt_path = ANALYSIS / "description_contrast_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) \
        if ckpt_path.exists() else {}
    lock = threading.Lock()
    client = get_client(args.model)
    jobs = [p for p in pieces if f"{piece_id(p)}|{args.model}" not in ckpt]
    print(f"{len(pieces)} pieces: {len(jobs)} to contrast, "
          f"{len(pieces) - len(jobs)} cached", flush=True)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.replace(ckpt_path)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(contrast_one, client, p): p for p in jobs}
        for fut in as_completed(futs):
            p = futs[fut]
            with lock:
                ckpt[f"{piece_id(p)}|{args.model}"] = fut.result()
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    save()
                    print(f"  [{done}/{len(jobs)}]", flush=True)
    save()

    rows, counts = [], {"only_enroute": Counter(), "only_posthoc": Counter()}
    for p in pieces:
        res = ckpt.get(f"{piece_id(p)}|{args.model}")
        if not res:
            continue
        rows.append({"id": piece_id(p), "model": p["model"], "mode": p["mode"],
                     "sample": p["sample"], "rater": args.model, **res})
        for side in counts:
            counts[side].update(it["category"] for it in res[side])
    out = {"pieces": rows,
           "category_totals": {side: dict(c.most_common())
                               for side, c in counts.items()}}
    path = ANALYSIS / "description_contrast.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Wrote {len(rows)} pieces -> {path}")
    for side, c in counts.items():
        print(f"  {side}: {dict(c.most_common(5))}")


if __name__ == "__main__":
    main()
