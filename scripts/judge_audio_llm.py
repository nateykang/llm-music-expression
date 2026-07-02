#!/usr/bin/env python3
"""Audio-vs-notation LLM judge — a controlled within-model read-vs-hear experiment.

gemini-2.5-pro and gpt-audio each judge every free-form piece TWICE: once from its
blinded NOTATION (text) and once from its rendered AUDIO (mp3), using the SAME blind
rubric as the notation-judge panel. Lets us ask, per model, whether it rates a piece
differently reading vs. hearing it — and whether hearing pulls it toward Music2Emo.

Resumable: every (piece, judge, modality) verdict is checkpointed. Gemini runs with
thinking disabled (cheap; a judging task needs none). Both judges see no title,
composer note, or instrument names — audio is inherently blind; notation is stripped.

Usage:  python scripts/judge_audio_llm.py [--limit N] [--workers 4]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_music.judge import (  # noqa: E402
    AFFECT_KEYS, QUALITY_KEYS, _extract_json, _system, build_user, representation,
)

KEYS = QUALITY_KEYS + AFFECT_KEYS
NOTE_SYSTEM = _system(False)
AUDIO_SYSTEM = (
    "You are an expert music critic evaluating short solo or small-ensemble pieces "
    "presented as an AUDIO RECORDING. Judge ONLY what you can perceive by listening — "
    "harmony, melodic line, rhythm, form, timbre, and emotional character. Do not reward "
    "length. Be calibrated and critical: on each 1-5 dimension, 3 = competent but "
    "unremarkable, 5 = genuinely excellent, 1 = a clear failure. For every dimension write "
    "a one-sentence justification and THEN an integer 1-5 using the anchors. Also name the "
    "single dominant emotional character. Return ONLY one valid JSON object, no prose."
)
AUDIO_USER_STUB = "(the music is attached to this message as an audio recording — listen to it and judge)"


def parse_verdict(obj):
    if not isinstance(obj, dict):
        return None
    out = {}
    for k in KEYS:
        v = obj.get(k)
        if isinstance(v, dict) and "score" in v:
            try:
                out[k] = {"score": float(v["score"]), "reason": str(v.get("reason", ""))[:200]}
            except (TypeError, ValueError):
                pass
        elif isinstance(v, (int, float)):
            out[k] = {"score": float(v)}
    lbl = str(obj.get("emotion_label", "")).strip().lower()
    if lbl:
        out["emotion_label"] = lbl
    return out or None


def gemini_judge(system, user, audio_bytes, attempts=3):
    from google import genai
    from google.genai import types

    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    contents = []
    if audio_bytes is not None:
        contents.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"))
    contents.append(user)
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=128),  # min (pro can't disable); cost control
    )
    for a in range(attempts):
        try:
            r = c.models.generate_content(model="gemini-2.5-pro", contents=contents, config=cfg)
            v = parse_verdict(_extract_json(r.text))
            if v:
                return v
        except Exception as e:
            # A daily-cap 429 won't clear on retry — bail immediately so we don't
            # burn 3x the request budget against the RPD limit.
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                return None
        time.sleep(min(2 ** a, 8))
    return None


def openai_judge(system, user, audio_bytes, attempts=3):
    # Routed through OpenRouter (same underlying openai/gpt-audio model) so it draws
    # from the OpenRouter balance rather than the primary OpenAI account.
    import openai

    c = openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                      base_url="https://openrouter.ai/api/v1")
    content = [{"type": "text", "text": user}]
    if audio_bytes is not None:
        content.append({"type": "input_audio",
                        "input_audio": {"data": base64.b64encode(audio_bytes).decode(), "format": "mp3"}})
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    for a in range(attempts):
        try:
            r = c.chat.completions.create(model="openai/gpt-audio", messages=msgs)
            v = parse_verdict(_extract_json(r.choices[0].message.content))
            if v:
                return v
        except Exception:
            pass
        time.sleep(min(2 ** a, 8))
    return None


JUDGES = {"gemini-2.5-pro": gemini_judge, "gpt-audio": openai_judge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    analysis = ROOT / "docs/analysis"
    ckpt_path = analysis / "judge_audio_llm_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
    lock = Lock()

    # build the task list: one (piece, batch, judge, modality) per call
    tasks = []
    for bd in sorted((ROOT / "docs/data").glob("2026*")):
        dj = bd / "data.json"
        if not dj.exists():
            continue
        for p in json.loads(dj.read_text())["pieces"]:
            if p.get("prompt") != "free-form" or not p.get("ok"):
                continue
            ar = p.get("audio")
            if not ar or not (bd / ar).exists():
                continue
            pid = f'{p["model"]}|{p.get("mode")}|{p.get("title")}|{p.get("sample")}'
            for judge in JUDGES:
                # gpt-audio requires audio in the request — it can only HEAR, not read.
                # gemini-2.5-pro does both, giving the clean within-model read-vs-hear.
                modalities = ("notation", "audio") if judge == "gemini-2.5-pro" else ("audio",)
                for modality in modalities:
                    key = f"{pid}|{judge}|{modality}"
                    if key not in ckpt:
                        tasks.append((p, bd, judge, modality, key))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(tasks)} verdicts to run ({sum(1 for k in ckpt if ckpt[k])} already cached)", flush=True)

    def run_one(task):
        p, bd, judge, modality, key = task
        try:
            if modality == "notation":
                rep_kind, rep_text = representation(p, bd)
                if rep_text is None:
                    return key, None
                system, user, audio = NOTE_SYSTEM, build_user(p, rep_kind, rep_text), None
            else:
                system, user = AUDIO_SYSTEM, build_user(p, "audio recording", AUDIO_USER_STUB)
                audio = (bd / p["audio"]).read_bytes()
            verdict = JUDGES[judge](system, user, audio)
        except Exception:
            verdict = None
        return key, verdict

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for key, verdict in ex.map(run_one, tasks):
            ckpt[key] = verdict
            done += 1
            if done % 20 == 0:
                with lock:
                    ckpt_path.write_text(json.dumps(ckpt))
                ok = sum(1 for k in ckpt if ckpt[k])
                print(f"  {done}/{len(tasks)} done ({ok} non-null total)", flush=True)
    ckpt_path.write_text(json.dumps(ckpt))

    # materialize a clean results file from the checkpoint
    results = []
    for key, verdict in ckpt.items():
        if not verdict:
            continue
        gm, mode, title, sample, judge, modality = key.split("|")
        rec = {"model": gm, "mode": mode, "title": title, "sample": sample,
               "judge": judge, "modality": modality}
        for k, v in verdict.items():
            rec[k] = v["score"] if isinstance(v, dict) and "score" in v else v
        results.append(rec)
    (analysis / "judge_audio_llm.json").write_text(json.dumps(results, indent=1))
    ok = sum(1 for k in ckpt if ckpt[k])
    print(f"=== AUDIO-LLM JUDGE DONE: {ok}/{len(ckpt)} verdicts, {len(results)} rows ===", flush=True)


if __name__ == "__main__":
    main()
