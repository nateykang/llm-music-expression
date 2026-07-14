#!/usr/bin/env python3
"""Plan idea 3 (LLM-graded version) — rate each composer description on
in-context traits the lexicon pass can't see: implied quality, confidence of
tone, concreteness, claimed emotional intensity, self-criticism. Rated from
the TEXT ALONE (the rater never sees the music), so correlating these with
blind panel scores measures whether the composer's framing is calibrated,
and correlating with the noted-minus-blind delta measures what sways judges.

Same anchored reason-before-score protocol as the judge rubric.

    python scripts/rate_description_traits.py --limit 10   # pilot
    python scripts/rate_description_traits.py              # full 651

Resumable (content-keyed checkpoint).
Writes docs/analysis/description_trait_ratings.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import warnings
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

TRAITS = [
    ("implied_quality", "Implied quality",
     "How good a piece does this text imply the composer believes they made?",
     "presents it as slight/flawed", "presents it as an accomplished work"),
    ("confidence", "Confidence of tone",
     "How assured is the voice — does it assert its choices or hedge them?",
     "tentative, heavily hedged", "fully assured, no hedging"),
    ("specificity", "Musical specificity",
     "How much concrete, checkable musical detail does it give (vs vague poetic mood-talk)?",
     "entirely vague/poetic", "dense with concrete musical detail"),
    ("emotional_intensity", "Claimed emotional intensity",
     "How emotionally intense is the experience the text claims the music delivers?",
     "mild, low-stakes", "overwhelming, profound"),
    ("self_criticism", "Self-criticism",
     "Does the text acknowledge limitations, trade-offs, or things that may not have worked?",
     "no limitation acknowledged", "explicitly names weaknesses"),
]
KEYS = [k for k, *_ in TRAITS]

SYSTEM = (
    "You are a careful literary annotator. You are given the title and "
    "self-description a composer wrote about one short piece of music. You "
    "never see or hear the music: rate ONLY the text's rhetorical stance. Be "
    "calibrated: 3 = typical for such blurbs, 1 and 5 = clear extremes. For "
    "every dimension write a one-sentence justification and THEN an integer "
    "1-5 using the anchors. Return ONLY one valid JSON object."
)


def build_user(piece: dict) -> str:
    rubric = "\n".join(f"- {k} ({label}): {q} [1 = {lo}; 5 = {hi}]"
                       for k, label, q, lo, hi in TRAITS)
    schema = ", ".join(f'"{k}": {{"reason": "...", "score": 1-5}}' for k in KEYS)
    return (
        f"TITLE: {piece['title']}\n"
        f"SHORT DESCRIPTION: {piece['short_description']}\n"
        f"LONG DESCRIPTION: {piece['long_description']}\n\n"
        f"Rate the text on each dimension:\n{rubric}\n\n"
        f"Return ONLY this JSON object (integer scores 1-5):\n{{{schema}}}"
    )


def rate_one(client, piece: dict, attempts: int = 3) -> dict | None:
    user = build_user(piece)
    for a in range(attempts):
        try:
            raw = client.complete(SYSTEM, user, json_mode=True)
            obj = _extract_json(raw, expect_keys=KEYS)
        except Exception as e:
            if not is_retryable(e):
                print(f"  permanent error, skipping {piece_id(piece)}: {e}", flush=True)
                return None
            print(f"  attempt {a + 1}/{attempts} {piece_id(piece)}: {e}", flush=True)
            obj = None
        if obj:
            out = {}
            for k in KEYS:
                v = obj.get(k)
                if isinstance(v, dict) and "score" in v:
                    try:
                        out[k] = float(v["score"])
                    except (TypeError, ValueError):
                        pass
                elif isinstance(v, (int, float)):
                    out[k] = float(v)
            if len(out) == len(KEYS):
                return out
        if a < attempts - 1:
            backoff_sleep(a, cap=8.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gpt-5.5")
    args = ap.parse_args()

    pieces = load_pieces(ROOT)
    todo = pieces[: args.limit] if args.limit else pieces
    ckpt_path = ANALYSIS / "description_trait_ratings_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) \
        if ckpt_path.exists() else {}
    lock = threading.Lock()
    client = get_client(args.model)
    jobs = [p for p in todo if f"{piece_id(p)}|{args.model}" not in ckpt]
    print(f"{len(todo)} pieces: {len(jobs)} to rate, {len(todo) - len(jobs)} cached",
          flush=True)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.replace(ckpt_path)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(rate_one, client, p): p for p in jobs}
        for fut in as_completed(futs):
            p = futs[fut]
            with lock:
                ckpt[f"{piece_id(p)}|{args.model}"] = fut.result()
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    save()
                    print(f"  [{done}/{len(jobs)}]", flush=True)
    save()

    rows = [{"id": piece_id(p), "batch": p["batch"], "model": p["model"],
             "mode": p["mode"], "sample": p["sample"], "rater": args.model,
             "ratings": ckpt[f"{piece_id(p)}|{args.model}"]}
            for p in todo if ckpt.get(f"{piece_id(p)}|{args.model}")]
    out = ANALYSIS / "description_trait_ratings.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"Wrote {len(rows)} rating rows -> {out}")


if __name__ == "__main__":
    main()
