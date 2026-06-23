"""LLM-as-judge: rate each generated piece on a literature-grounded rubric.

Dimensions follow the converged human-study axes — ChatMusician (consistency,
structure), Chu et al. survey (melodiousness, naturalness, creativity, coherence),
and MuSpike (musicality / structure / tonality / harmony / emotion / novelty /
Turing). Protocol follows the LLM-as-judge literature — Zheng et al. (MT-Bench)
and Liu et al. (G-Eval): a short justification *before* each score (chain-of-
thought), anchored 1-5 scales, and a blind panel of diverse frontier judges,
averaged, with each model's own pieces judged by the *other* panelists to defuse
self-enhancement bias.

By default the judge is BLIND — it sees ONLY the music, with the composer's note,
title, ABC comments and voice names stripped, so emotion/key are *perceived* from
the notes, not read off text. Run with include_note=True to add the composer's
note (and the intent-execution dimension): blind vs noted is a clean text-bias
experiment. Emotion is characterized three ways: expressiveness (a craft score),
perceived valence + arousal (the Russell/EMOPIA 2-D model, comparable to our
computed proxies), and a single dominant emotion label.
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Craft/quality dimensions (1-5) — these average into the headline "overall".
# (key, label, question, low-anchor, high-anchor)
QUALITY = [
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
     "How vividly and intentionally does it express *some* emotional character (regardless of which)?",
     "flat, no affect", "vivid, strongly expressive"),
    ("creativity", "Creativity / interest",
     "Is it engaging and individual, or generic and formulaic?",
     "formulaic / generic", "novel and compelling"),
    ("naturalness", "Naturalness (Turing)",
     "Could a human composer plausibly have written this?",
     "obviously machine-made", "indistinguishable from human"),
]
# Descriptive emotional-character dimensions (1-5) — NOT quality (a dark piece is
# not "worse"); these map to our computed valence/arousal proxies.
AFFECT = [
    ("valence", "Emotional valence",
     "How positive/bright vs negative/dark is the emotional character?",
     "very dark / negative", "very bright / positive"),
    ("arousal", "Emotional arousal",
     "How calm/still vs energetic/intense is the emotional character?",
     "very calm / still", "very energetic / intense"),
]
# Only meaningful when the composer's note is shown (the noted condition).
INTENT = ("intent", "Intent–execution",
          "Does the music actually deliver what the composer's own note says it intended?",
          "claim unmet by the music", "fully realizes the stated intent")

# Single dominant-emotion vocabulary, spanning the circumplex quadrants.
EMOTION_LABELS = ["joyful", "triumphant", "playful", "serene", "tender", "wistful",
                  "melancholic", "sombre", "tense", "turbulent", "mysterious", "neutral"]

QUALITY_KEYS = [k for k, *_ in QUALITY]
AFFECT_KEYS = [k for k, *_ in AFFECT]
ALL_KEYS = QUALITY_KEYS + AFFECT_KEYS + ["intent"]  # csv column order


def _system(include_note: bool) -> str:
    base = (
        "You are an expert music critic evaluating short solo or small-ensemble "
        "pieces presented in symbolic notation (ABC, or a note-by-note listing). "
        "Judge ONLY what you can perceive from the notes — harmony, melodic line, "
        "rhythm, form, and emotional character. Do not reward length. Be calibrated "
        "and critical: on each 1-5 dimension, 3 = competent but unremarkable, 5 = "
        "genuinely excellent, 1 = a clear failure. For every dimension write a "
        "one-sentence justification and THEN an integer 1-5 using the anchors. Also "
        "name the single dominant emotional character. Return ONLY one valid JSON "
        "object, no prose."
    )
    if include_note:
        base += (" You are also given the composer's own note about what they "
                 "intended; use it only for the intent-execution dimension.")
    return base


def _strip_abc_text(abc: str) -> str:
    """Remove textual/semantic hints so a blind judge sees only the music: title,
    composer, lyrics and other free-text info fields; comment lines and inline
    comments; and voice name= / subname= attributes. Keeps musical directives
    (X K M L Q V P R, %%score) and the notes themselves."""
    out = []
    for line in abc.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("%") and not s.startswith("%%"):  # comment line
            continue
        if re.match(r"^[TCWNOSGHBDFZ]:", s):  # text info fields (keep X K M L Q V P R)
            continue
        if "%" in line and "%%" not in line:  # inline comment
            line = line[:line.index("%")].rstrip()
            if not line.strip():
                continue
        line = re.sub(r'\b(?:name|subname)\s*=\s*"[^"]*"', "", line)  # voice names leak intent
        line = re.sub(r"\b(?:name|subname)\s*=\s*\S+", "", line)
        out.append(line.rstrip())
    return "\n".join(out)


def _note_tok(n) -> str:
    d = f"{float(n.quarterLength):g}"
    if n.isChord:
        return "[" + ",".join(p.nameWithOctave for p in n.pitches) + "]/" + d
    return n.pitches[0].nameWithOctave + "/" + d


def _score_to_text(score, max_measures: int = 64) -> str:
    """Compact, LLM-readable rendering of a music21 score for code-gen pieces
    (which have no ABC): per part (generic index — no instrument names), per bar,
    notes as Pitch+octave / duration-in-beats."""
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
    """(kind, text) the judge reads — always text-stripped so only music remains."""
    if piece.get("abc"):
        return "ABC notation", _strip_abc_text(piece["abc"])
    if piece.get("score"):
        from .analyze import _load
        with tempfile.TemporaryDirectory(prefix="judge_rep_") as td:
            _, score = _load(piece, batch_dir, Path(td))
        if score is not None:
            return ("a note listing (Pitch+octave / duration-in-beats, per bar)",
                    _score_to_text(score))
    return None, None


def build_user(piece: dict, rep_kind: str, rep_text: str, include_note: bool = False) -> str:
    items = QUALITY + AFFECT + ([INTENT] if include_note else [])
    rubric = "\n".join(
        f"- {key} ({label}): {q} [1 = {lo}; 5 = {hi}]"
        for key, label, q, lo, hi in items)
    schema = ", ".join(f'"{k}": {{"reason": "...", "score": 1-5}}' for k, *_ in items)
    note_block = ""
    if include_note:
        note = (piece.get("long_description") or piece.get("short_description") or "").strip()
        note_block = (f"COMPOSER'S NOTE (their stated intent):\n{note or '(none given)'}\n\n")
    return (
        f"{note_block}"
        f"THE MUSIC ({rep_kind}):\n{rep_text}\n\n"
        f"Rate the piece on each dimension:\n{rubric}\n\n"
        f"Also choose the single dominant emotional character from EXACTLY this list: "
        f"{', '.join(EMOTION_LABELS)}.\n\n"
        f"Return ONLY this JSON object (integer scores 1-5):\n"
        f'{{{schema}, "emotion_label": "<one label>"}}'
    )


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    depth, start, spans = 0, None, []
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
            if isinstance(obj, dict) and (any(k in obj for k in QUALITY_KEYS) or "emotion_label" in obj):
                return obj
        except Exception:
            continue
    return None


def judge_piece(client, piece: dict, batch_dir: Path, include_note: bool = False) -> dict | None:
    """One judge's verdict on one piece → {key:{score,reason}, emotion_label:str}."""
    rep_kind, rep_text = representation(piece, batch_dir)
    if rep_text is None:
        return None
    user = build_user(piece, rep_kind, rep_text, include_note)
    try:
        raw = client.complete(_system(include_note), user)
    except Exception:
        return None
    obj = _extract_json(raw)
    if not obj:
        return None
    out = {}
    keys = QUALITY_KEYS + AFFECT_KEYS + (["intent"] if include_note else [])
    for k in keys:
        v = obj.get(k)
        if isinstance(v, dict) and "score" in v:
            try:
                out[k] = {"score": float(v["score"]), "reason": str(v.get("reason", ""))[:300]}
            except Exception:
                pass
        elif isinstance(v, (int, float)):
            out[k] = {"score": float(v), "reason": ""}
    lbl = str(obj.get("emotion_label", "")).strip().lower()
    if lbl:
        out["emotion_label"] = lbl
    return out or None


def _features_index(batch_dir: Path) -> dict:
    import csv
    f = batch_dir / "features.csv"
    if not f.exists():
        return {}
    return {(r["model"], r["prompt"], r.get("mode", ""), r.get("title", "")): r
            for r in csv.DictReader(f.open(encoding="utf-8"))}


def judge_corpus(data_dir: Path, judge_names: list[str], *, prompt: str | None = None,
                 limit: int | None = None, workers: int = 6, exclude_self: bool = True,
                 include_note: bool = False, out_name: str | None = None):
    """Run the judge panel over every successful piece. Each piece is scored by all
    panelists except (optionally) its own generating model; per-dimension scores are
    averaged across the panel, and the dominant emotion label is the panel mode.
    Writes <out_name>.csv (panel means) + <out_name>_raw.json."""
    from .models import get_client

    clients = {name: get_client(name) for name in judge_names}
    lock = threading.Lock()
    out_name = out_name or ("judge_noted" if include_note else "judge")

    tasks = []
    for batch_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        manifest = batch_dir / "data.json"
        if not manifest.exists():
            continue
        pieces = json.loads(manifest.read_text(encoding="utf-8")).get("pieces", [])
        for pc in pieces:
            if not pc.get("ok"):
                continue
            if prompt and pc.get("prompt") != prompt:
                continue
            tasks.append((pc, batch_dir))
    if limit:
        tasks = tasks[:limit]

    jobs = []
    for ti, (pc, bd) in enumerate(tasks):
        for jname in judge_names:
            if exclude_self and jname == pc["model"]:
                continue
            jobs.append((ti, jname, pc, bd))

    print(f"Judging {len(tasks)} pieces × panel {judge_names} = {len(jobs)} calls "
          f"({workers} workers, exclude_self={exclude_self}, include_note={include_note})")
    verdicts: dict[int, dict[str, dict]] = {}

    def work(job):
        ti, jname, pc, bd = job
        return ti, jname, judge_piece(clients[jname], pc, bd, include_note)

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

    score_keys = QUALITY_KEYS + AFFECT_KEYS + (["intent"] if include_note else [])
    rows, raw = [], []
    for ti, (pc, bd) in enumerate(tasks):
        panel = verdicts.get(ti, {})
        if not panel:
            continue
        row = {"model": pc["model"], "prompt": pc["prompt"], "mode": pc.get("mode", ""),
               "title": pc.get("title", ""), "n_judges": len(panel)}
        for k in score_keys:
            scores = [v[k]["score"] for v in panel.values() if k in v]
            row[k] = round(sum(scores) / len(scores), 3) if scores else None
        labels = [v["emotion_label"] for v in panel.values() if v.get("emotion_label")]
        row["emotion_label"] = Counter(labels).most_common(1)[0][0] if labels else ""
        # headline quality = mean of quality dims (+ intent when noted); affect excluded
        qk = QUALITY_KEYS + (["intent"] if include_note else [])
        q = [row[k] for k in qk if row.get(k) is not None]
        row["overall"] = round(sum(q) / len(q), 3) if q else None
        rows.append(row)
        raw.append({"model": pc["model"], "prompt": pc["prompt"], "mode": pc.get("mode", ""),
                    "title": pc.get("title", ""), "panel": panel})

    analysis = data_dir.parent / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    import csv
    out_csv = analysis / f"{out_name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "prompt", "mode", "title", "n_judges",
                                          *ALL_KEYS, "emotion_label", "overall"])
        w.writeheader()
        w.writerows(rows)
    (analysis / f"{out_name}_raw.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    print(f"\nWrote {len(rows)} judged pieces → {out_csv}")
    return rows
