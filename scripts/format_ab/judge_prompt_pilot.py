"""Two-arm judge-prompt pilot: current system prompt vs composer-persona draft.

Design: 40 stratified pieces (16 native-ABC / 16 codegen / 8 sparse, spread
across models) + 10 duplicate submissions (test-retest) + 4 degenerate anchors.
Each judged under both arms by a 6-judge one-per-lab panel, self-excluded,
native representations, prompt text otherwise identical. Repo untouched.
"""
import json
import logging
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", "/Users/nathanielkang/llm-music-expression"))
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
logging.disable(logging.WARNING)
import llm_music.config  # noqa: E402,F401
from llm_music.judge import _extract_json, _system, build_user, representation  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.retry import backoff_sleep, is_retryable  # noqa: E402

SEED = 20260803
JUDGES = ["opus-4.8", "gpt-5.5", "gemini-3.1-pro", "grok-4.5",
          "deepseek-v4-pro", "qwen3-max"]
WORKERS = 6

ARMS = {
    "current": _system(False),
    "composer": (
        "You are a music composer evaluating pieces presented in symbolic "
        "notation (ABC, or a note-by-note listing). Judge ONLY what you can "
        "perceive from the notes. Do not reward length. Be calibrated and "
        "critical: on each 1-5 dimension, 3 = competent but unremarkable, 5 = "
        "genuinely excellent, 1 = a clear failure. For every dimension write a "
        "one-sentence justification and THEN an integer 1-5 using the anchors. "
        "Also name the single dominant emotional character. Return ONLY one "
        "valid JSON object, no prose."
    ),
}

# --- piece selection ----------------------------------------------------------
by_mode = defaultdict(list)
anchors = []
for manifest in sorted(ROOT.glob("docs/data/2026*/data.json")):
    batch = manifest.parent
    for p in json.loads(manifest.read_text(encoding="utf-8")).get("pieces", []):
        has_art = p.get("abc") or p.get("score")
        if not has_art:
            continue
        if p.get("degenerate"):
            anchors.append((batch, p))
        elif p.get("ok"):
            m = p.get("mode", "")
            if m in ("abc", "codegen", "codegen-sparse"):
                by_mode[m].append((batch, p))

random.seed(SEED)


def stratified(items, n):
    """Spread across models: shuffle within model, round-robin."""
    per = defaultdict(list)
    for it in items:
        per[it[1]["model"]].append(it)
    for v in per.values():
        random.shuffle(v)
    order = sorted(per)
    random.shuffle(order)
    out, i = [], 0
    while len(out) < n and any(per[m] for m in order):
        m = order[i % len(order)]
        if per[m]:
            out.append(per[m].pop())
        i += 1
    return out

sample = (stratified(by_mode["abc"], 16) + stratified(by_mode["codegen"], 16)
          + stratified(by_mode["codegen-sparse"], 8))
random.shuffle(anchors)
anchor_sel = anchors[:4]
dups = sample[:10]
print(f"pieces: {len(sample)} main + {len(dups)} dup + {len(anchor_sel)} anchors; "
      f"modes: {dict((m, sum(1 for _, p in sample if p.get('mode') == m)) for m in ('abc', 'codegen', 'codegen-sparse'))}",
      flush=True)

# --- jobs ---------------------------------------------------------------------
ckpt_path = SCRATCH / "prompt_pilot_ckpt.json"
attempted = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
clients = {j: get_client(j) for j in JUDGES}
lock = threading.Lock()

work_items = ([("main", b, p) for b, p in sample]
              + [("dup", b, p) for b, p in dups]
              + [("anchor", b, p) for b, p in anchor_sel])

reps = {}
jobs = []
for tag, batch, p in work_items:
    pk = f"{batch.name}|{p['model']}|{p['prompt']}|{p.get('sample', 0)}|{tag}"
    if pk.rsplit("|", 1)[0] not in reps:
        reps[pk.rsplit("|", 1)[0]] = representation(p, batch)
    kind, text = reps[pk.rsplit("|", 1)[0]]
    if text is None:
        continue
    user = build_user(p, kind, text, False)
    for arm in ARMS:
        for j in JUDGES:
            if j == p["model"]:
                continue
            key = f"{pk}|{arm}|{j}"
            if key not in attempted:
                jobs.append((key, arm, j, user))
print(f"{len(jobs)} calls to do ({len(attempted)} cached)", flush=True)


def judge_call(system, client, user, attempts=3):
    obj = None
    for a in range(attempts):
        try:
            raw = client.complete(system, user, json_mode=True)
            obj = _extract_json(raw)
        except Exception as e:
            obj = None
            if not is_retryable(e):
                return None
        if obj:
            return obj
        if a < attempts - 1:
            backoff_sleep(a, cap=8.0)
    return None


def work(job):
    key, arm, j, user = job
    return key, judge_call(ARMS[arm], clients[j], user)


def save():
    tmp = ckpt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(attempted))
    tmp.replace(ckpt_path)


done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for fut in as_completed([ex.submit(work, jb) for jb in jobs]):
        key, verdict = fut.result()
        with lock:
            attempted[key] = verdict
            done += 1
            if done % 25 == 0 or done == len(jobs):
                save()
                print(f"  [{done}/{len(jobs)}]", flush=True)
save()

# --- analysis -----------------------------------------------------------------
import statistics as st
QUALITY = ["coherence", "harmony", "rhythm", "structure", "melody",
           "emotion", "creativity", "naturalness"]


def score(v, k):
    x = v.get(k)
    if isinstance(x, dict) and "score" in x:
        try:
            return float(x["score"])
        except Exception:
            return None
    return float(x) if isinstance(x, (int, float)) else None

# rows: key parts = batch|model|prompt|sample|tag|arm|judge
cells = []
for key, v in attempted.items():
    if not v:
        continue
    b, m, pr, s, tag, arm, j = key.split("|")
    row = {"piece": f"{b}|{m}|{pr}|{s}", "tag": tag, "arm": arm, "judge": j}
    for k in QUALITY + ["valence", "arousal"]:
        row[k] = score(v, k)
    q = [row[k] for k in QUALITY if row[k] is not None]
    row["overall"] = sum(q) / len(q) if q else None
    cells.append(row)
(SCRATCH / "prompt_pilot_results.json").write_text(json.dumps(cells, indent=1))

print("\n=== ARM COMPARISON (paired within piece x judge, main+dup) ===")
pair = defaultdict(dict)
for r in cells:
    if r["tag"] in ("main", "dup"):
        pair[(r["piece"], r["tag"], r["judge"])][r["arm"]] = r
for k in QUALITY + ["overall", "valence", "arousal"]:
    ds = [d["composer"][k] - d["current"][k] for d in pair.values()
          if "current" in d and "composer" in d
          and d["current"][k] is not None and d["composer"][k] is not None]
    if ds:
        se = (st.stdev(ds) / len(ds) ** 0.5) if len(ds) > 1 else 0
        print(f"  {k:12} composer-minus-current = {st.mean(ds):+.3f} (SE {se:.3f}, n={len(ds)})")

print("\n=== DISCRIMINATION (SD of panel-mean overall across main pieces) ===")
for arm in ARMS:
    pm = defaultdict(list)
    for r in cells:
        if r["tag"] == "main" and r["arm"] == arm and r["overall"] is not None:
            pm[r["piece"]].append(r["overall"])
    means = [st.mean(v) for v in pm.values() if v]
    print(f"  {arm:10} spread={st.stdev(means):.3f}  mean={st.mean(means):.3f}  (n={len(means)})")

print("\n=== TEST-RETEST (mean |main - dup|, same judge same arm) ===")
for arm in ARMS:
    diffs = []
    idx = {(r["piece"], r["judge"], r["tag"]): r for r in cells if r["arm"] == arm}
    for (pc, j, tag), r in idx.items():
        if tag == "dup" and (pc, j, "main") in idx:
            a, b2 = r, idx[(pc, j, "main")]
            if a["overall"] is not None and b2["overall"] is not None:
                diffs.append(abs(a["overall"] - b2["overall"]))
    if diffs:
        print(f"  {arm:10} mean|retest diff|={st.mean(diffs):.3f}  (n={len(diffs)})")

print("\n=== DEGENERATE ANCHORS (panel-mean overall; must stay LOW) ===")
for arm in ARMS:
    vals = [r["overall"] for r in cells if r["tag"] == "anchor" and r["arm"] == arm
            and r["overall"] is not None]
    if vals:
        print(f"  {arm:10} anchor mean={st.mean(vals):.3f}  (n={len(vals)} judgments)")
