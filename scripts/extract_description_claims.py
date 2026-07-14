#!/usr/bin/env python3
"""Experiment 2.1 — claimed vs measured: LLM-extract the *verifiable musical
claims* each composer's description makes (key, mode, tempo, texture, voices,
instruments, dynamics, claimed emotion, form), then check them against the
computed features (features.csv) of the very piece they describe.

Corpus: the free-form 30-samples-per-composer batches (abc + codegen, 651 ok
pieces) — see description_corpus.py. Descriptions are used VERBATIM (no strip):
explicit key/tempo mentions are exactly the claims under test.

    python scripts/extract_description_claims.py --limit 10   # pilot
    python scripts/extract_description_claims.py              # full run
    python scripts/extract_description_claims.py --compare-only

Resumable (content-keyed checkpoint). Writes:
  docs/analysis/description_claims.json        raw per-piece claims
  docs/analysis/description_faithfulness.json  per composer x mode scoreboard
"""

from __future__ import annotations

import argparse
import json
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
from llm_music.judge import _extract_json  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.retry import backoff_sleep, is_retryable  # noqa: E402

ANALYSIS = ROOT / "docs/analysis"

MODES = ["major", "minor", "ionian", "aeolian", "dorian", "phrygian", "lydian",
         "mixolydian", "locrian", "chromatic", "atonal"]
_MODE_FOLD = {"ionian": "major", "aeolian": "minor"}
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

SYSTEM = (
    "You are a careful annotator. You are given the title and self-description a "
    "composer wrote about one short piece of music. Extract ONLY what the text "
    "explicitly claims about the music itself. Do not infer from genre or style "
    "conventions; if a property is not stated, use null. Exception: valence and "
    "arousal describe the emotional character the text claims the music has — "
    "rate those from the overall description. Return ONLY one valid JSON object."
)

FIELDS = ["key_tonic", "key_mode", "tempo_bpm", "tempo_character", "meter",
          "texture", "n_voices", "instruments", "dynamics_mentioned",
          "valence", "arousal", "form"]


def build_user(piece: dict) -> str:
    return (
        f"TITLE: {piece['title']}\n"
        f"SHORT DESCRIPTION: {piece['short_description']}\n"
        f"LONG DESCRIPTION: {piece['long_description']}\n\n"
        "Extract the text's explicit claims about the music as JSON with exactly "
        "these fields (null wherever the text does not state it):\n"
        '- "key_tonic": tonic note as written, e.g. "C", "F#", "Bb"\n'
        f'- "key_mode": one of {", ".join(MODES)}\n'
        '- "tempo_bpm": number, only if a BPM/metronome figure is stated\n'
        '- "tempo_character": "slow" | "moderate" | "fast" (from words like adagio,'
        " andante, allegro, 'gentle pace', 'driving')\n"
        '- "meter": time signature as written, e.g. "3/4", "6/8"\n'
        '- "texture": "monophonic" | "homophonic" | "polyphonic" (homophonic = '
        "melody with accompaniment/chords; polyphonic = independent simultaneous "
        "lines, counterpoint)\n"
        '- "n_voices": integer count of voices/parts/hands-as-parts, if stated\n'
        '- "instruments": list of instrument names named as sounding in the piece '
        "(empty list if none)\n"
        '- "dynamics_mentioned": true if the text claims dynamic shaping/contrast '
        "(crescendo, pp/ff, swells), else false\n"
        '- "valence": 1-5, the emotional positivity the text claims (1 = very dark/'
        "sad, 3 = neutral/mixed, 5 = very bright/joyful) — always rate\n"
        '- "arousal": 1-5, the energy the text claims (1 = very calm/still, 5 = '
        "very energetic/intense) — always rate\n"
        '- "form": short label only if a form is claimed, e.g. "ABA", "ternary", '
        '"rondo", "theme and variations", "through-composed"\n\n'
        "Return ONLY the JSON object."
    )


def normalize(obj: dict) -> dict | None:
    """Coerce the model's JSON into the fixed schema; None if unusable."""
    if not isinstance(obj, dict):
        return None
    out = {}
    tonic = obj.get("key_tonic")
    out["key_tonic"] = str(tonic).strip() if tonic else None
    mode = str(obj.get("key_mode") or "").strip().lower() or None
    out["key_mode"] = mode if mode in MODES else None
    try:
        out["tempo_bpm"] = float(obj["tempo_bpm"]) if obj.get("tempo_bpm") else None
    except (TypeError, ValueError):
        out["tempo_bpm"] = None
    tc = str(obj.get("tempo_character") or "").strip().lower() or None
    out["tempo_character"] = tc if tc in ("slow", "moderate", "fast") else None
    meter = obj.get("meter")
    out["meter"] = str(meter).strip() if meter else None
    tx = str(obj.get("texture") or "").strip().lower() or None
    out["texture"] = tx if tx in ("monophonic", "homophonic", "polyphonic") else None
    try:
        out["n_voices"] = int(obj["n_voices"]) if obj.get("n_voices") else None
    except (TypeError, ValueError):
        out["n_voices"] = None
    ins = obj.get("instruments")
    out["instruments"] = [str(i).strip().lower() for i in ins if str(i).strip()] \
        if isinstance(ins, list) else []
    out["dynamics_mentioned"] = bool(obj.get("dynamics_mentioned"))
    for k in ("valence", "arousal"):
        try:
            v = float(obj[k])
            out[k] = min(5.0, max(1.0, v))
        except (TypeError, ValueError, KeyError):
            out[k] = None
    form = obj.get("form")
    out["form"] = str(form).strip() if form else None
    return out


def extract_one(client, piece: dict, attempts: int = 3) -> dict | None:
    user = build_user(piece)
    for a in range(attempts):
        try:
            raw = client.complete(SYSTEM, user, json_mode=True)
            obj = normalize(_extract_json(raw, expect_keys=FIELDS))
        except Exception as e:
            if not is_retryable(e):
                print(f"  permanent error, skipping {piece_id(piece)}: {e}", flush=True)
                return None
            print(f"  attempt {a + 1}/{attempts} {piece_id(piece)}: {e}", flush=True)
            obj = None
        if obj:
            return obj
        if a < attempts - 1:
            backoff_sleep(a, cap=8.0)
    return None


# ---------------------------------------------------------------- comparison

def _pc(tonic: str) -> int | None:
    """Pitch class of a written tonic ('F#', 'Bb', 'D-') — enharmonic-safe."""
    t = (tonic or "").strip()
    if not t or t[0].upper() not in _PC:
        return None
    pc = _PC[t[0].upper()]
    for ch in t[1:]:
        if ch in "#♯":
            pc += 1
        elif ch in "b♭-":
            pc -= 1
    return pc % 12


def _f(row: dict, key: str) -> float | None:
    try:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else None
    except (TypeError, ValueError):
        return None


def _tempo_band_ok(character: str, bpm: float) -> bool:
    return {"slow": bpm <= 95, "moderate": 76 <= bpm <= 144,
            "fast": bpm >= 110}[character]


def compare(claims_rows: list[dict], pieces: list[dict]) -> dict:
    """Score every verifiable claim against the piece's measured features."""
    from scipy.stats import spearmanr

    by_id = {piece_id(p): p for p in pieces}
    groups: dict[tuple, list] = defaultdict(list)
    for row in claims_rows:
        p = by_id.get(row["id"])
        if p and p.get("features"):
            groups[(p["model"], p["mode"])].append((row["claims"], p["features"]))

    def score(pairs: list[tuple]) -> dict:
        n = len(pairs)
        acc: dict[str, list] = defaultdict(list)   # checkable claims -> 0/1
        err: dict[str, list] = defaultdict(list)
        va_c, va_m, ar_c, ar_m = [], [], [], []
        stated = defaultdict(int)
        for c, f in pairs:
            for k in FIELDS:
                v = c.get(k)
                if k == "instruments":
                    v = v or None
                if k == "dynamics_mentioned":
                    v = v or None
                if v is not None:
                    stated[k] += 1
            if c["key_tonic"] and f.get("key_tonic"):
                pc_c, pc_m = _pc(c["key_tonic"]), _pc(f["key_tonic"])
                if pc_c is not None and pc_m is not None:
                    acc["key_tonic"].append(int(pc_c == pc_m))
            cm = _MODE_FOLD.get(c["key_mode"] or "", c["key_mode"])
            if cm in ("major", "minor") and f.get("key_mode") in ("major", "minor"):
                acc["key_mode"].append(int(cm == f["key_mode"]))
            bpm = _f(f, "tempo_bpm")
            if bpm:
                if c["tempo_bpm"]:
                    tol = max(5.0, 0.05 * bpm)
                    acc["tempo_bpm"].append(int(abs(c["tempo_bpm"] - bpm) <= tol))
                    err["tempo_bpm"].append(abs(c["tempo_bpm"] - bpm))
                if c["tempo_character"]:
                    acc["tempo_character"].append(
                        int(_tempo_band_ok(c["tempo_character"], bpm)))
            nv, poly = _f(f, "n_voices"), _f(f, "polyphony")
            multi = (nv or 0) > 1 or (poly or 0) > 1.5
            if c["texture"]:
                claimed_multi = c["texture"] in ("homophonic", "polyphonic")
                acc["texture"].append(int(claimed_multi == multi))
            if c["n_voices"] and nv:
                acc["n_voices"].append(int(c["n_voices"] == nv))
                err["n_voices"].append(abs(c["n_voices"] - nv))
            n_ins = _f(f, "n_instruments")
            if c["instruments"] and n_ins:
                acc["n_instruments"].append(int(len(set(c["instruments"])) == n_ins))
            marks = _f(f, "n_dynamic_marks")
            if c["dynamics_mentioned"] and marks is not None:
                acc["dynamics"].append(int(marks > 0))
            for cl, ms, cs, msr in ((c["valence"], _f(f, "valence"), va_c, va_m),
                                    (c["arousal"], _f(f, "arousal"), ar_c, ar_m)):
                if cl is not None and ms is not None:
                    cs.append(cl)
                    msr.append(ms)
        out = {"n": n,
               "claim_rate": {k: round(stated[k] / n, 3) for k in FIELDS if n},
               "accuracy": {k: {"acc": round(sum(v) / len(v), 3), "n": len(v)}
                            for k, v in sorted(acc.items()) if v},
               "mae": {k: round(sum(v) / len(v), 2) for k, v in err.items() if v}}
        for name, cs, msr in (("valence", va_c, va_m), ("arousal", ar_c, ar_m)):
            if len(cs) >= 10:
                rho, pval = spearmanr(cs, msr)
                if rho == rho:  # skip NaN (zero-variance claims)
                    out[f"{name}_spearman"] = {"rho": round(float(rho), 3),
                                               "p": round(float(pval), 5), "n": len(cs)}
        return out

    result = {"per_composer": {}, "per_mode": {}, "overall": score(
        [pair for pairs in groups.values() for pair in pairs])}
    for (model, mode), pairs in sorted(groups.items()):
        result["per_composer"][f"{model}|{mode}"] = score(pairs)
    by_mode = defaultdict(list)
    for (_, mode), pairs in groups.items():
        by_mode[mode].extend(pairs)
    for mode, pairs in sorted(by_mode.items()):
        result["per_mode"][mode] = score(pairs)
    return result


def print_summary(result: dict) -> None:
    def fmt(s: dict) -> str:
        a = s["accuracy"]
        cells = [f"{k}={v['acc']:.0%}({v['n']})" for k, v in a.items()]
        va = s.get("valence_spearman")
        ar = s.get("arousal_spearman")
        if va:
            cells.append(f"val_rho={va['rho']:+.2f}")
        if ar:
            cells.append(f"aro_rho={ar['rho']:+.2f}")
        return "  ".join(cells)

    print(f"\nOVERALL (n={result['overall']['n']}): {fmt(result['overall'])}")
    for mode, s in result["per_mode"].items():
        print(f"  [{mode}] (n={s['n']}): {fmt(s)}")
    print()
    for key, s in result["per_composer"].items():
        print(f"{key:34s} n={s['n']:3d}  {fmt(s)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="pilot on first N pieces")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gpt-5.5", help="extractor model (registry id)")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip extraction; rescore existing description_claims.json")
    args = ap.parse_args()

    pieces = load_pieces(ROOT)
    claims_path = ANALYSIS / "description_claims.json"

    if args.compare_only:
        claims_rows = json.loads(claims_path.read_text(encoding="utf-8"))
    else:
        todo = pieces[: args.limit] if args.limit else pieces
        ckpt_path = ANALYSIS / "description_claims_ckpt.json"
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) \
            if ckpt_path.exists() else {}
        lock = threading.Lock()
        client = get_client(args.model)
        jobs = [p for p in todo if f"{piece_id(p)}|{args.model}" not in ckpt]
        print(f"{len(todo)} pieces: {len(jobs)} to extract, "
              f"{len(todo) - len(jobs)} cached", flush=True)

        def save():
            tmp = ckpt_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(ckpt))
            tmp.replace(ckpt_path)

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(extract_one, client, p): p for p in jobs}
            for fut in as_completed(futs):
                p = futs[fut]
                with lock:
                    ckpt[f"{piece_id(p)}|{args.model}"] = fut.result()
                    done += 1
                    if done % 25 == 0 or done == len(jobs):
                        save()
                        print(f"  [{done}/{len(jobs)}]", flush=True)
        save()

        claims_rows = []
        for p in todo:
            c = ckpt.get(f"{piece_id(p)}|{args.model}")
            if c:
                claims_rows.append({"id": piece_id(p), "batch": p["batch"],
                                    "model": p["model"], "mode": p["mode"],
                                    "sample": p["sample"], "extractor": args.model,
                                    "claims": c})
        claims_path.write_text(json.dumps(claims_rows, indent=1), encoding="utf-8")
        print(f"Wrote {len(claims_rows)} claim rows -> {claims_path}")

    result = compare(claims_rows, pieces)
    out = ANALYSIS / "description_faithfulness.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print_summary(result)
    print(f"\nWrote scoreboard -> {out}")


if __name__ == "__main__":
    main()
