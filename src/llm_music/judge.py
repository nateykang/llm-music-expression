"""LLM-as-judge: rate each generated piece on a literature-grounded rubric.

Dimensions follow the converged human-study axes — ChatMusician (consistency,
structure), Chu et al. survey (melodiousness, naturalness, creativity, coherence),
and MuSpike (musicality / structure / tonality / harmony / emotion / novelty /
Turing). Protocol follows the LLM-as-judge literature — Zheng et al. (MT-Bench)
and Liu et al. (G-Eval): a short justification *before* each score (chain-of-
thought), anchored 1-5 scales, and a blind panel of diverse frontier judges,
averaged, with each model's own pieces judged by the *other* panelists to defuse
self-enhancement bias. The final `intent` dimension (does the music deliver what
the composer's note claimed?) is our own intent-vs-execution contribution.
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# (key, label, question, low-anchor, high-anchor)
RUBRIC = [
    ("coherence", "Coherence / fluency",
     "Does the music flow naturally, or does it glitch, stall, or lurch between unrelated ideas?",
     "disjointed / glitchy", "flows naturally end to end"),
    ("harmony", "Tonal & harmonic coherence",
     "Is there a clear tonal center and sensible harmony, or is it aimless / accidentally atonal?",
     "aimless or atonal", "clear, purposeful harmony"),
    ("rhythm", "Rhythmic consistency",
     "Does it hold an intentional rhythmic identity, or is the rhythm erratic and pulseless?",
     "erratic, no pulse", "unified, intentional groove"),
    ("structure", "Structure / development",
     "Is there clear form — repetition, contrast, development of ideas — or does it meander?",
     "meanders, no form", "clear form, develops its material"),
    ("melody", "Melodic quality",
     "Is the melodic writing shapely and memorable, or just notes with no line?",
     "shapeless / random", "strong, memorable line"),
    ("emotion", "Emotional expressiveness",
     "Does it convey a clear, intentional emotional character?",
     "flat, no affect", "vivid, intentional emotion"),
    ("creativity", "Creativity / interest",
     "Is it engaging and individual, or generic and formulaic?",
     "formulaic / generic", "novel and compelling"),
    ("naturalness", "Naturalness (Turing)",
     "Could a human composer plausibly have written this?",
     "obviously machine-made", "indistinguishable from human"),
    ("intent", "Intent–execution",
     "Does the music actually deliver what the composer's own note says it intended?",
     "claim unmet by the music", "fully realizes the stated intent"),
]
KEYS = [k for k, *_ in RUBRIC]

JUDGE_SYSTEM = (
    "You are an expert music critic evaluating short solo or small-ensemble pieces "
    "presented in symbolic notation (ABC, or a note-by-note listing). You are given "
    "the notation, the composer's own note about what they intended, and a few "
    "measured facts about the piece. Judge ONLY the music as represented. Do not "
    "reward length or verbosity. Be calibrated and critical: on each dimension, 3 = "
    "competent but unremarkable, 5 = genuinely excellent, 1 = a clear failure. For "
    "every dimension, write a one-sentence justification and THEN an integer 1-5 "
    "using the given anchors. Return ONLY a single valid JSON object, no prose."
)


def _note_tok(n) -> str:
    d = f"{float(n.quarterLength):g}"
    if n.isChord:
        return "[" + ",".join(p.nameWithOctave for p in n.pitches) + "]/" + d
    return n.pitches[0].nameWithOctave + "/" + d


def _score_to_text(score, max_measures: int = 64) -> str:
    """Compact, LLM-readable rendering of a music21 score for code-gen pieces
    (which have no ABC): per part, per bar, notes as Pitch/duration-in-beats."""
    parts = list(score.parts) or [score]
    lines = []
    for pi, part in enumerate(parts):
        measures = list(part.getElementsByClass("Measure"))
        if not measures:
            toks = [_note_tok(n) for n in list(part.recurse().notes)[:400]]
            lines.append(f"Part {pi + 1}: " + " ".join(toks))
            continue
        lines.append(f"Part {pi + 1}:")
        for mi, m in enumerate(measures[:max_measures]):
            toks = [_note_tok(n) for n in m.notes]
            if toks:
                lines.append(f"  m{mi + 1}: " + " ".join(toks))
        if len(measures) > max_measures:
            lines.append(f"  … ({len(measures) - max_measures} more measures)")
    return "\n".join(lines)


def representation(piece: dict, batch_dir: Path) -> tuple[str | None, str | None]:
    """(kind, text) the judge will read: raw ABC, or a note listing for code-gen."""
    if piece.get("abc"):
        return "ABC notation", piece["abc"]
    if piece.get("score"):
        from .analyze import _load
        with tempfile.TemporaryDirectory(prefix="judge_rep_") as td:
            _, score = _load(piece, batch_dir, Path(td))
        if score is not None:
            return ("a note listing (Pitch+octave / duration-in-beats, per bar)",
                    _score_to_text(score))
    return None, None


def _facts(feat: dict | None) -> str:
    if not feat:
        return "(none available)"
    bits = []
    dk = (feat.get("key_declared_tonic") or "") + " " + (feat.get("key_declared_mode") or "")
    if dk.strip():
        bits.append(f"declared key {dk.strip()}")
    if feat.get("key_tonic") and feat.get("key_tonic") != "?":
        bits.append(f"detected key {feat['key_tonic']} {feat.get('key_mode', '')}".strip())
    for label, key in (("tempo", "tempo_bpm"), ("scale-consistency", "scale_consistency"),
                       ("harmonic-motion", "chord_tonal_distance"), ("structureness", "structureness")):
        v = feat.get(key)
        if v not in (None, ""):
            bits.append(f"{label} {v}")
    return "; ".join(bits) if bits else "(none available)"


def build_user(piece: dict, feat: dict | None, rep_kind: str, rep_text: str) -> str:
    note = (piece.get("long_description") or piece.get("short_description") or "").strip()
    rubric = "\n".join(
        f"- {key} ({label}): {q} [1 = {lo}; 5 = {hi}]"
        for key, label, q, lo, hi in RUBRIC)
    schema = ", ".join(f'"{k}": {{"reason": "...", "score": 1-5}}' for k in KEYS)
    return (
        f"PIECE TITLE: {piece.get('title', '(untitled)')}\n\n"
        f"COMPOSER'S NOTE (their stated intent):\n{note or '(none given)'}\n\n"
        f"MEASURED FACTS: {_facts(feat)}\n\n"
        f"THE MUSIC ({rep_kind}):\n{rep_text}\n\n"
        f"Rate the piece on each dimension:\n{rubric}\n\n"
        f"Return ONLY this JSON object (integer scores 1-5):\n{{{schema}}}"
    )


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    # also try the last balanced {...} block
    depth, start = 0, None
    spans = []
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start:i + 1])
    candidates += spans[::-1]
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and any(k in obj for k in KEYS):
                return obj
        except Exception:
            continue
    return None


def judge_piece(client, piece: dict, feat: dict | None, batch_dir: Path) -> dict | None:
    """One judge's verdict on one piece → {key: {score, reason}} or None on failure."""
    rep_kind, rep_text = representation(piece, batch_dir)
    if rep_text is None:
        return None
    user = build_user(piece, feat, rep_kind, rep_text)
    try:
        raw = client.complete(JUDGE_SYSTEM, user)
    except Exception:
        return None
    obj = _extract_json(raw)
    if not obj:
        return None
    out = {}
    for k in KEYS:
        v = obj.get(k)
        if isinstance(v, dict) and "score" in v:
            try:
                out[k] = {"score": float(v["score"]), "reason": str(v.get("reason", ""))[:300]}
            except Exception:
                pass
        elif isinstance(v, (int, float)):
            out[k] = {"score": float(v), "reason": ""}
    return out or None


def _features_index(batch_dir: Path) -> dict:
    """Map (model, prompt, mode, title) → feature row from the batch's features.csv."""
    import csv
    f = batch_dir / "features.csv"
    if not f.exists():
        return {}
    idx = {}
    for r in csv.DictReader(f.open(encoding="utf-8")):
        idx[(r["model"], r["prompt"], r.get("mode", ""), r.get("title", ""))] = r
    return idx


def judge_corpus(data_dir: Path, judge_names: list[str], *, prompt: str | None = None,
                 limit: int | None = None, workers: int = 6, exclude_self: bool = True):
    """Run the judge panel over every successful piece. Each piece is scored by all
    panelists except (optionally) its own generating model; per-dimension scores are
    averaged across the panel. Writes judge.csv (panel means) + judge_raw.json."""
    from .models import get_client

    clients = {name: get_client(name) for name in judge_names}
    lock = threading.Lock()

    # collect (piece, feat, batch_dir) tasks across all batches
    tasks = []
    for batch_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        manifest = batch_dir / "data.json"
        if not manifest.exists():
            continue
        pieces = json.loads(manifest.read_text(encoding="utf-8")).get("pieces", [])
        fidx = _features_index(batch_dir)
        for pc in pieces:
            if not pc.get("ok"):
                continue
            if prompt and pc.get("prompt") != prompt:
                continue
            feat = fidx.get((pc["model"], pc["prompt"], pc.get("mode", ""), pc.get("title", "")))
            tasks.append((pc, feat, batch_dir))
    if limit:
        tasks = tasks[:limit]

    # fan out (piece × panelist), skipping self-judgments
    jobs = []
    for ti, (pc, feat, bd) in enumerate(tasks):
        for jname in judge_names:
            if exclude_self and jname == pc["model"]:
                continue
            jobs.append((ti, jname, pc, feat, bd))

    print(f"Judging {len(tasks)} pieces × panel {judge_names} "
          f"= {len(jobs)} calls ({workers} workers, exclude_self={exclude_self})")
    verdicts: dict[int, dict[str, dict]] = {}

    def work(job):
        ti, jname, pc, feat, bd = job
        return ti, jname, judge_piece(clients[jname], pc, feat, bd)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            ti, jname, verdict = fut.result()
            with lock:
                done += 1
                if verdict:
                    verdicts.setdefault(ti, {})[jname] = verdict
                if done % 20 == 0 or done == len(jobs):
                    print(f"  [{done}/{len(jobs)}]", flush=True)

    # aggregate: per piece, average each dimension across panelists
    rows, raw = [], []
    for ti, (pc, feat, bd) in enumerate(tasks):
        panel = verdicts.get(ti, {})
        if not panel:
            continue
        row = {"model": pc["model"], "prompt": pc["prompt"], "mode": pc.get("mode", ""),
               "title": pc.get("title", ""), "n_judges": len(panel)}
        for k in KEYS:
            scores = [v[k]["score"] for v in panel.values() if k in v]
            row[k] = round(sum(scores) / len(scores), 3) if scores else None
        present = [row[k] for k in KEYS if row[k] is not None]
        row["overall"] = round(sum(present) / len(present), 3) if present else None
        rows.append(row)
        raw.append({"model": pc["model"], "prompt": pc["prompt"], "mode": pc.get("mode", ""),
                    "title": pc.get("title", ""), "panel": panel})

    out_csv = data_dir.parent / "analysis" / "judge.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "prompt", "mode", "title", "n_judges", *KEYS, "overall"])
        w.writeheader()
        w.writerows(rows)
    (data_dir.parent / "analysis" / "judge_raw.json").write_text(
        json.dumps(raw, indent=1), encoding="utf-8")
    print(f"\nWrote {len(rows)} judged pieces → {out_csv}")
    return rows
