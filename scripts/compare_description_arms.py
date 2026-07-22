#!/usr/bin/env python3
"""En-route vs post-hoc valence comparison — Q1 of the description-arms
experiment. Each piece's description pair (the composing call's text vs the
fresh music-only call's text under `independent_description`) is rated blind
by LLM raters on three anchored scales, one call per (piece, arm, rater),
rating the short and long text together. The rater sees the text alone — no
title, model, or arm label. Deterministic measures (VADER, lengths,
first-person rate) ride along for free.

    python scripts/compare_description_arms.py --limit 4   # smoke test
    python scripts/compare_description_arms.py             # full corpus

Resumable (content-keyed checkpoint). Writes docs/analysis/valence_comparison.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
SEED = 20260722
RATERS = ["fable-5", "gpt-5.6-thinking"]

# (key, question, scale max, low anchor, high anchor)
SCALES = [
    ("evaluative_positivity",
     "How positively does the text describe the piece's quality, craft, or success?",
     7, "openly critical", "effusive praise; 4 = purely descriptive with no evaluation"),
    ("weakness_admission",
     "To what extent does the text acknowledge flaws, limitations, or unrealized "
     "ambitions in the piece?",
     5, "none", "extensive"),
    ("affect_valence",
     "Consider the emotions, moods, and imagery the text says the music expresses "
     "or evokes. Rate how negative or positive that described emotional content is. "
     "Ignore the writing quality and ignore any claims about the piece's craft — "
     "rate only the described emotion.",
     9, "deeply negative (grief, despair, dread)",
     "strongly positive (joy, serenity, triumph); 5 = neutral, ambivalent, or "
     "balanced between dark and light"),
]
KEYS = [k for k, *_ in SCALES]

SYSTEM = (
    "You are a careful literary annotator. You are given a short and a long "
    "description of one piece of music. You never see or hear the music: rate "
    "ONLY the text's stance, for each text separately. Be calibrated: use the "
    "full scale, reserving the endpoints for clear extremes. For every "
    "dimension write a one-sentence justification and THEN an integer score "
    "using the anchors. Return ONLY one valid JSON object."
)


def build_user(short: str, long: str) -> str:
    rubric = "\n".join(
        f"- {k}: {q} [1 = {lo}; {mx} = {hi}]" for k, q, mx, lo, hi in SCALES
    )
    schema = ", ".join(f'"{k}": {{"reason": "...", "score": N}}' for k in KEYS)
    return (
        f"SHORT DESCRIPTION: {short}\n"
        f"LONG DESCRIPTION: {long}\n\n"
        f"Rate each text on each dimension:\n{rubric}\n\n"
        f'Return ONLY this JSON object: {{"short": {{{schema}}}, "long": {{{schema}}}}}'
    )


def _scores(obj: dict) -> dict | None:
    out = {}
    for text_key in ("short", "long"):
        section = obj.get(text_key)
        if not isinstance(section, dict):
            return None
        for k, _, mx, _, _ in SCALES:
            v = section.get(k)
            score = v.get("score") if isinstance(v, dict) else v
            try:
                score = float(score)
            except (TypeError, ValueError):
                return None
            if not 1 <= score <= mx:
                return None
            out[f"{text_key}.{k}"] = score
            if isinstance(v, dict) and isinstance(v.get("reason"), str):
                out[f"{text_key}.{k}.reason"] = v["reason"]
    return out


def rate_one(client, short: str, long: str, attempts: int = 3) -> dict | None:
    user = build_user(short, long)
    for a in range(attempts):
        try:
            raw = client.complete(SYSTEM, user, json_mode=True)
            obj = _extract_json(raw, expect_keys=["short", "long"])
            out = _scores(obj) if obj else None
            if out:
                return out
        except Exception as e:
            if not is_retryable(e):
                print(f"  permanent error: {e}", flush=True)
                return None
            print(f"  attempt {a + 1}/{attempts}: {e}", flush=True)
        if a < attempts - 1:
            backoff_sleep(a, cap=8.0)
    return None


_FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'd|me|my|mine|myself)\b")


def text_measures(short: str, long: str) -> dict:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    global _VADER
    if "_VADER" not in globals():
        _VADER = SentimentIntensityAnalyzer()
    out = {}
    for key, text in (("short", short), ("long", long)):
        words = text.split()
        out[f"{key}.words"] = len(words)
        out[f"{key}.vader"] = _VADER.polarity_scores(text)["compound"]
        out[f"{key}.first_person_per_100w"] = (
            100.0 * len(_FIRST_PERSON.findall(text)) / max(1, len(words))
        )
    return out


def arms(piece: dict) -> dict[str, tuple[str, str]] | None:
    ind = piece.get("independent_description")
    if not ind or not ind.get("short_description"):
        return None
    return {
        "enroute": (piece["short_description"], piece["long_description"]),
        "posthoc": (ind["short_description"], ind["long_description"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap pieces (those with both arms) for a pilot")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--raters", default=",".join(RATERS))
    args = ap.parse_args()
    raters = [r.strip() for r in args.raters.split(",") if r.strip()]

    pieces = [p for p in load_pieces(ROOT, include_sparse=True) if arms(p)]
    if args.limit:
        pieces = pieces[: args.limit]
    print(f"{len(pieces)} pieces with both description arms", flush=True)

    ckpt_path = ANALYSIS / "valence_comparison_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) \
        if ckpt_path.exists() else {}
    lock = threading.Lock()
    clients = {r: get_client(r) for r in raters}

    jobs = []
    for p in pieces:
        for arm, (short, long) in arms(p).items():
            for rater in raters:
                if rater == p["model"]:   # never let a model rate its own text
                    continue
                key = f"{piece_id(p)}|{arm}|{rater}"
                if key not in ckpt:
                    jobs.append((key, rater, short, long))
    random.Random(SEED).shuffle(jobs)
    print(f"{len(jobs)} rating calls to make", flush=True)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.replace(ckpt_path)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(rate_one, clients[r], s, l): key
                for key, r, s, l in jobs}
        for fut in as_completed(futs):
            with lock:
                ckpt[futs[fut]] = fut.result()
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    save()
                    print(f"  [{done}/{len(jobs)}]", flush=True)
    save()

    rows = []
    for p in pieces:
        row = {"id": piece_id(p), "batch": p["batch"], "model": p["model"],
               "mode": p["mode"], "sample": p["sample"],
               "describer": p["independent_description"].get("model"),
               "representation": p["independent_description"].get("representation")}
        for arm, (short, long) in arms(p).items():
            row[arm] = {"measures": text_measures(short, long)}
            for rater in raters:
                rating = ckpt.get(f"{piece_id(p)}|{arm}|{rater}")
                if rating:
                    row[arm][rater] = rating
        rows.append(row)
    out = ANALYSIS / "valence_comparison.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"Wrote {len(rows)} pieces -> {out}")


if __name__ == "__main__":
    main()
