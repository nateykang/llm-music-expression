#!/usr/bin/env python3
"""Experiment 1 — blind description->music matching, cross-composer puzzles.

Design: within each generation method (abc / codegen), shuffle each composer's
30 free-form pieces (seeded); round i takes the i-th piece from every composer,
giving an 11-piece puzzle with one piece per composer. Every judge on the panel
sees the round's SCORES (blind representation: composer text/voice names
stripped, as in judge.py) and its DESCRIPTIONS (technical identifiers masked —
key, tempo/BPM, meter, instruments, voice counts) in independently scrambled
order, and must return a one-to-one assignment. 30 rounds x 2 methods x 11
judges; per-piece chance = 1/puzzle_size (11 except late codegen rounds, where
failed generations shrink the pool).

Reads:  part 1 — which composers write descriptions that actually identify
        their music (per-composer matchability, pooled over judges);
        part 2 — self-signature: does a judge match its OWN piece more often
        than the other judges match that same piece?

    python scripts/match_descriptions.py --rounds 1 --judges fable-5   # pilot
    python scripts/match_descriptions.py                               # full
    python scripts/match_descriptions.py --no-mask                     # ablation

Resumable (content-keyed checkpoint). Writes docs/analysis/
description_matching.json (raw assignments) + description_matching_summary.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from description_corpus import load_pieces, piece_id  # noqa: E402
from llm_music.judge import _extract_json, representation  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.retry import backoff_sleep, is_retryable  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"
SEED = 20260710

PANEL = ["fable-5", "gpt-5.5", "gemini-2.5-pro", "opus-4.8", "sonnet-4.6",
         "gpt-4.1", "grok-4.3", "deepseek-v4-pro", "qwen3-max",
         "llama-4-maverick", "opus-4.8-thinking"]

# ------------------------------------------------------------------ masking
# The blind score text keeps its K:/Q:/M: headers and instrument part labels
# (needed to read the music), so the same identifiers must leave the
# descriptions or matching degenerates to header lookup.

_NOTE = r"[A-G](?:[b#♭♯]|-?\s?(?:flat|sharp))?"
_MODES = (r"major|minor|ionian|aeolian|dorian|phrygian|lydian|mixolydian|"
          r"locrian|pentatonic|blues|chromatic")
_INSTRUMENTS = (r"pianos?|violins?|violas?|cellos?|celli|harps?|flutes?|oboes?|"
                r"clarinets?|bassoons?|horns?|trumpets?|trombones?|tubas?|"
                r"guitars?|organs?|harpsichords?|marimbas?|vibraphones?|"
                r"glockenspiels?|celestas?|strings?|woodwinds?|brass|timpani|"
                r"drums?|percussion|synths?|synthesizers?|choirs?|"
                r"contrabass|double bass|string quartet")
_NUMWORD = r"(?:\d+|one|two|three|four|five|six|seven|eight)"

_MASKS = [
    (re.compile(rf"\b(?:in\s+(?:the\s+key\s+of\s+)?)?{_NOTE}[\s-]+(?:{_MODES})\b",
                re.I), "[key]"),
    (re.compile(rf"\bkey\s+of\s+{_NOTE}\b", re.I), "[key]"),
    (re.compile(rf"\b(?:{_MODES})\s+(?:key|mode|tonality|scale)\b", re.I), "[key]"),
    (re.compile(r"\b(?:quarter[- ]note\s*=\s*)?\d{2,3}\s*(?:bpm|beats per minute)\b",
                re.I), "[tempo]"),
    (re.compile(r"[♩♪♫]\s*=\s*\d{2,3}"), "[tempo]"),
    (re.compile(r"\bat\s+(?:around\s+|about\s+|~\s*)?\d{2,3}\b"), "at [tempo]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}(?:\s+(?:time|meter))?\b"), "[meter]"),
    (re.compile(rf"\b{_NUMWORD}[- ](?:voices?|parts?|hands?)\b", re.I), "[voices]"),
    (re.compile(rf"\b(?:{_INSTRUMENTS})\b", re.I), "[instrument]"),
]


def mask_description(text: str) -> str:
    for pat, repl in _MASKS:
        text = pat.sub(repl, text)
    return text


# ------------------------------------------------------------------ puzzles

def build_puzzles(pieces: list[dict]) -> list[dict]:
    """Per method: shuffle within composer, round i = i-th piece per composer."""
    rng = random.Random(SEED)
    puzzles = []
    for method in ("abc", "codegen"):
        groups: dict[str, list] = defaultdict(list)
        for p in pieces:
            if p["mode"] == method:
                groups[p["model"]].append(p)
        for g in groups.values():
            g.sort(key=piece_id)
            rng.shuffle(g)
        n_rounds = max(len(g) for g in groups.values())
        for r in range(n_rounds):
            members = [g[r] for g in groups.values() if r < len(g)]
            if len(members) >= 5:
                puzzles.append({"method": method, "round": r, "pieces": members})
    return puzzles


SYSTEM = (
    "You are an expert musicologist. You are shown N short pieces in symbolic "
    "notation and N composer self-descriptions, in scrambled order. Each "
    "description was written by the composer of exactly one of the pieces. "
    "Some technical identifiers in the descriptions (key, tempo, meter, "
    "instruments, voice counts) are masked as [key], [tempo], [meter], "
    "[instrument], [voices]; match on musical content — melodic contour, "
    "texture, form, development, rhythmic character, emotional arc. Think "
    "carefully, then return ONLY one JSON object mapping every description id "
    "to a distinct score id, e.g. {\"D1\": \"S3\", ...}. Use each score id "
    "exactly once."
)


def build_user(scores: list[tuple[str, str]], descs: list[tuple[str, str]]) -> str:
    parts = ["THE PIECES:\n"]
    parts += [f"--- {sid} ---\n{text}\n" for sid, text in scores]
    parts.append("\nTHE DESCRIPTIONS:\n")
    parts += [f"--- {did} ---\n{text}\n" for did, text in descs]
    parts.append(
        f"\nMatch every description to its piece. Return ONLY the JSON object "
        f'{{"D1": "S?", ..., "D{len(descs)}": "S?"}} using each of '
        f"S1..S{len(scores)} exactly once.")
    return "".join(parts)


def solve(client, user: str, n: int, attempts: int = 3) -> dict | None:
    for a in range(attempts):
        try:
            raw = client.complete(SYSTEM, user, json_mode=True)
            obj = _extract_json(raw, expect_keys=("D1",))
        except Exception as e:
            if not is_retryable(e):
                print(f"  {client.name}: permanent error: {e}", flush=True)
                return None
            print(f"  {client.name}: attempt {a + 1}/{attempts}: {e}", flush=True)
            obj = None
        if obj:
            out = {}
            for i in range(1, n + 1):
                v = str(obj.get(f"D{i}", "")).strip().upper()
                if re.fullmatch(rf"S([1-9]|1[0-9])", v) and 1 <= int(v[1:]) <= n:
                    out[f"D{i}"] = v
            if len(out) >= n - 1:  # tolerate one dropped answer
                return out
        if a < attempts - 1:
            backoff_sleep(a, cap=8.0)
    return None


def run_puzzle(client, puzzle: dict, rep_cache: dict, judge: str,
               mask: bool) -> dict | None:
    members = puzzle["pieces"]
    rng = random.Random(f"{SEED}|{puzzle['method']}|{puzzle['round']}|{judge}")
    score_order = list(members)
    desc_order = list(members)
    rng.shuffle(score_order)
    rng.shuffle(desc_order)
    scores = [(f"S{i+1}", rep_cache[piece_id(p)]) for i, p in enumerate(score_order)]
    descs = []
    for i, p in enumerate(desc_order):
        text = f"{p['short_description']} {p['long_description']}".strip()
        descs.append((f"D{i+1}", mask_description(text) if mask else text))
    ans = solve(client, build_user(scores, descs), len(members))
    if ans is None:
        return None
    sid_of = {piece_id(p): f"S{i+1}" for i, p in enumerate(score_order)}
    composer_of_sid = {f"S{i+1}": p["model"] for i, p in enumerate(score_order)}
    rows = []
    for i, p in enumerate(desc_order):
        pred = ans.get(f"D{i+1}")
        rows.append({"composer": p["model"], "piece": piece_id(p),
                     "predicted_composer": composer_of_sid.get(pred),
                     "correct": bool(pred and pred == sid_of[piece_id(p)])})
    return {"method": puzzle["method"], "round": puzzle["round"], "judge": judge,
            "size": len(members), "results": rows}


# ------------------------------------------------------------------ summary

def summarize(records: list[dict]) -> dict:
    out = {}
    for method in ("abc", "codegen"):
        recs = [r for r in records if r["method"] == method]
        if not recs:
            continue
        flat = [(r["judge"], x) for r in recs for x in r["results"]]
        chance = (sum(1.0 / r["size"] for r in recs for _ in r["results"])
                  / max(1, len(flat)))
        per_judge, per_composer = defaultdict(list), defaultdict(list)
        self_hits, other_hits = defaultdict(list), defaultdict(list)
        confusion = defaultdict(int)
        for judge, x in flat:
            per_judge[judge].append(x["correct"])
            per_composer[x["composer"]].append(x["correct"])
            (self_hits if judge == x["composer"] else other_hits)[
                x["composer"]].append(x["correct"])
            if x["predicted_composer"]:
                confusion[f"{x['composer']}->{x['predicted_composer']}"] += 1

        def acc(v):
            return round(sum(v) / len(v), 3) if v else None

        out[method] = {
            "n_puzzles": len(recs), "n_assignments": len(flat),
            "chance": round(chance, 3),
            "per_judge_acc": {k: acc(v) for k, v in sorted(per_judge.items())},
            "per_composer_matchability": {k: acc(v) for k, v
                                          in sorted(per_composer.items())},
            "self_vs_other": {k: {"self": acc(self_hits.get(k)),
                                  "other": acc(other_hits.get(k)),
                                  "n_self": len(self_hits.get(k, []))}
                              for k in sorted(per_composer)},
            "confusion_top": dict(sorted(confusion.items(),
                                         key=lambda kv: -kv[1])[:40]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=None, help="pilot: first N rounds")
    ap.add_argument("--judges", default=",".join(PANEL))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-mask", action="store_true",
                    help="leave descriptions verbatim (header-leak ablation)")
    args = ap.parse_args()
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    mask = not args.no_mask
    tag = "" if mask else "_nomask"

    pieces = load_pieces(ROOT)
    puzzles = build_puzzles(pieces)
    if args.rounds is not None:
        puzzles = [pz for pz in puzzles if pz["round"] < args.rounds]
    need = {piece_id(p) for pz in puzzles for p in pz["pieces"]}
    print(f"{len(puzzles)} puzzles, {len(need)} distinct pieces, "
          f"judges: {', '.join(judges)}, mask={mask}", flush=True)

    rep_cache_path = ANALYSIS / "matching_rep_cache.json"
    rep_cache = json.loads(rep_cache_path.read_text(encoding="utf-8")) \
        if rep_cache_path.exists() else {}
    todo_reps = [p for pz in puzzles for p in pz["pieces"]
                 if piece_id(p) not in rep_cache]
    seen = set()
    todo_reps = [p for p in todo_reps
                 if not (piece_id(p) in seen or seen.add(piece_id(p)))]
    if todo_reps:
        print(f"building {len(todo_reps)} blind score representations...",
              flush=True)
        for i, p in enumerate(todo_reps):
            _, text = representation(p, ROOT / "docs/data" / p["batch"])
            rep_cache[piece_id(p)] = text or ""
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(todo_reps)}]", flush=True)
        rep_cache_path.write_text(json.dumps(rep_cache), encoding="utf-8")

    ckpt_path = ANALYSIS / f"description_matching{tag}_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) \
        if ckpt_path.exists() else {}
    lock = threading.Lock()
    clients = {j: get_client(j) for j in judges}
    jobs = [(pz, j) for pz in puzzles for j in judges
            if f"{pz['method']}|{pz['round']}|{j}" not in ckpt]
    print(f"{len(jobs)} judge calls to do, {len(ckpt)} cached", flush=True)

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ckpt))
        tmp.replace(ckpt_path)

    def work(job):
        pz, j = job
        return (f"{pz['method']}|{pz['round']}|{j}",
                run_puzzle(clients[j], pz, rep_cache, j, mask))

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for fut in as_completed([ex.submit(work, jb) for jb in jobs]):
            k, v = fut.result()
            with lock:
                ckpt[k] = v
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    save()
                    print(f"  [{done}/{len(jobs)}]", flush=True)
    save()

    records = [v for v in ckpt.values() if v]
    (ANALYSIS / f"description_matching{tag}.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8")
    summary = summarize(records)
    (ANALYSIS / f"description_matching{tag}_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    for method, s in summary.items():
        print(f"\n[{method}] {s['n_puzzles']} puzzles, chance={s['chance']:.1%}")
        print("  judge accuracy:      ",
              {k: v for k, v in s["per_judge_acc"].items()})
        print("  composer matchability:",
              {k: v for k, v in s["per_composer_matchability"].items()})
        print("  self vs other:")
        for k, v in s["self_vs_other"].items():
            if v["n_self"]:
                print(f"    {k:22s} self={v['self']}  other={v['other']}")
    print(f"\nWrote description_matching{tag}.json + summary -> {ANALYSIS}")


if __name__ == "__main__":
    main()
