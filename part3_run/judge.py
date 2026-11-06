"""Self-preference matrix extension: Part 2 plus every roster-extension batch.

Env RUN_DIRS lists the generation runs (default part3_run); the last holds this
run's state. Jobs = (new express-yourself pieces × full panel) + (new judges ×
every Part 2 piece). The checkpoint is seeded with the Part 2 verdicts, so the
42 old judges are never re-run on old pieces. Every judge with a Batch API
endpoint gets one batch; each collected batch must pass an integrity gate
(validity, label vocabulary/collapse, reasons on-topic, r vs the Anthropic
panel) before it merges — failures are quarantined and redone live. A realtime
judge_corpus sweep then fills everything else and writes the CSV/raw.
"""
import collections
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from statistics import mean

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", "/Users/nathanielkang/llm-music-expression"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "part2_run"))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from llm_music import batch_api  # noqa: E402
from llm_music.judge import (EMOTION_LABELS, QUALITY_KEYS, _extract_json,  # noqa: E402
                             _system, build_user, judge_corpus, parse_verdict,
                             representation)
from llm_music.models import get_client  # noqa: E402
from reps import wire_reps_cache  # noqa: E402

OLD = ["20260819_201530__models_40_prompts_3",
       "20260819_203512__models_2_prompts_3",
       "20260820_061907__models_42_prompts_3"]
RUN_DIRS = [ROOT / d for d in os.environ.get("RUN_DIRS", "part3_run").split(",")]
NEW = [Path(v["batch"]).name for d in RUN_DIRS for v in
       json.loads((d / "gen_state.json").read_text())["out"].values()]
NO_BATCH = {"opus-4.1", "opus-4.1-thinking"}  # retired snapshot: :batch returns not_found
P3 = RUN_DIRS[-1]
OUT = f"{P3.name.removesuffix('_run')}_selfpref"
DATA = P3 / "data"
CKPT = P3 / "analysis" / f"{OUT}_ckpt.json"
JOBS = P3 / "judge_batches.json"
QUAR = P3 / "judge_quarantine.json"
REPS = P3 / "reps_cache.json"
DEADLINE_H = float(os.environ.get("BATCH_DEADLINE_H", "10"))


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def pk(b, pc):
    return f"{b}|{pc['model']}|{pc['prompt']}|{pc.get('mode', '')}|{pc.get('sample', 0)}"


DATA.mkdir(parents=True, exist_ok=True)
CKPT.parent.mkdir(parents=True, exist_ok=True)
for b in OLD + NEW:
    if not (DATA / b).exists():
        (DATA / b).symlink_to(ROOT / "docs/data" / b)

tasks = [(b, pc) for b in OLD + NEW
         for pc in json.loads((ROOT / "docs/data" / b / "data.json").read_text())["pieces"]
         if pc.get("ok") and pc.get("prompt") == "express-yourself"]
panel = sorted({pc["model"] for _, pc in tasks})
print(f"{len(tasks)} pieces ({sum(b in NEW for b, _ in tasks)} new), panel {len(panel)}",
      flush=True)

if CKPT.exists():
    ck = json.loads(CKPT.read_text())
else:
    ck = {}
    for row in json.loads((ROOT / "part2_run/analysis/part2_selfpref_raw.json").read_text()):
        key = f"{row['batch']}|{row['model']}|{row['prompt']}|{row['mode']}|{row['sample']}"
        for j, v in row["panel"].items():
            ck[f"{key}|{j}"] = v
    save(CKPT, ck)
print(f"checkpoint: {len(json.loads(CKPT.read_text()))} verdicts", flush=True)

reps = json.loads(REPS.read_text()) if REPS.exists() else json.loads(
    (ROOT / "part3_run/reps_cache.json").read_text())
for b, pc in tasks:
    if pk(b, pc) not in reps:
        reps[pk(b, pc)] = list(representation(pc, ROOT / "docs/data" / b))
save(REPS, reps)

# ---- batch legs ----------------------------------------------------------
system = _system(False)
jobs = json.loads(JOBS.read_text()) if JOBS.exists() else {}
or_ok = batch_api.openrouter_batchable()
for j in panel:
    if j in jobs or j in NO_BATCH:
        continue
    client = get_client(j)
    if not batch_api.supports_batch(client, or_ok):
        print(f"{j}: no batch endpoint, realtime", flush=True)
        continue
    reqs, idmap = {}, {}
    for b, pc in tasks:
        key = f"{pk(b, pc)}|{j}"
        kind, text = reps[pk(b, pc)]
        if key in ck or text is None:
            continue
        cid = f"q{len(idmap):06d}"
        idmap[cid] = key
        reqs[cid] = (system, build_user(pc, kind, text), True)
    if not reqs:
        continue
    if os.environ.get("DRY"):
        print(f"DRY {j} [{client.__class__.__name__}]: {len(reqs)} requests", flush=True)
        continue
    try:
        h = batch_api.submit(client, reqs, label=f"{P3.name}_{j}")
    except Exception as e:  # noqa: BLE001
        print(f"SUBMIT FAILED {j}: {e} — left for realtime", flush=True)
        continue
    h.update(idmap=idmap, submitted=time.time())
    jobs[j] = h
    save(JOBS, jobs)
    print(f"SUBMITTED {j}: {len(reqs)} -> {h['id']}", flush=True)

ANTH = [j for j in panel if get_client(j).__class__.__name__ == "AnthropicClient"]
KW = {"harmony": r"harmon|chord|tonal|key|cadence|progression",
      "rhythm": r"rhythm|pulse|meter|tempo|beat|groove",
      "melody": r"melod|line|contour|tune|phrase",
      "structure": r"structur|form|repeat|repetition|section|develop"}


def qmean(v):
    return mean(v[d]["score"] for d in QUALITY_KEYS)


def corr(a, b):
    if len(a) < 10:
        return float("nan")
    ma, mb = mean(a), mean(b)
    n = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    d = (sum((x - ma) ** 2 for x in a) ** .5) * (sum((y - mb) ** 2 for y in b) ** .5)
    return n / d if d else float("nan")


def gate(judge, verdicts, n_sent, ck):
    """Reasons a judge's batch fails integrity ([] = pass)."""
    labels = collections.Counter(v.get("emotion_label") for v in verdicts.values())
    xs, ys, kw_hit, kw_tot = [], [], 0, 0
    for key, v in verdicts.items():
        p = key.rsplit("|", 1)[0]
        ref = [qmean(ck[f"{p}|{a}"]) for a in ANTH if a != judge and ck.get(f"{p}|{a}")]
        if len(ref) >= 5:
            xs.append(qmean(v))
            ys.append(mean(ref))
        for d, pat in KW.items():
            kw_tot += 1
            kw_hit += bool(re.search(pat, v[d]["reason"], re.I))
    n = len(verdicts)
    r = corr(xs, ys)
    m = {"valid": f"{n}/{n_sent}", "r_vs_anth": round(r, 3), "n_r": len(xs),
         "top_label": labels.most_common(1)[0] if n else None,
         "kw": round(kw_hit / kw_tot, 3) if kw_tot else 0}
    bad = []
    if n < 0.92 * n_sent:
        bad.append("valid_rate")
    if n and sum(c for lb, c in labels.items() if lb in EMOTION_LABELS) < 0.95 * n:
        bad.append("label_vocab")
    if n and labels.most_common(1)[0][1] > 0.6 * n:
        bad.append("label_collapse")
    if kw_tot and kw_hit < 0.9 * kw_tot:
        bad.append("kw_align")
    if not r >= 0.35:
        bad.append("r_vs_anth")
    return bad, m


quarantine = json.loads(QUAR.read_text()) if QUAR.exists() else {}


def wait_done(names):
    """Poll until every named batch is terminal (or past the deadline)."""
    left = [j for j in names if not jobs[j].get("collected")]
    while left:
        for j in list(left):
            h = jobs[j]
            try:
                done, summary = batch_api.poll(h)
            except Exception as e:  # noqa: BLE001
                print(f"  poll error {j}: {e}", flush=True)
                continue
            if not done and time.time() - h["submitted"] > DEADLINE_H * 3600:
                cancelled = batch_api.cancel_and_wait(h)
                summary += " — deadline passed, " + ("cancelled" if cancelled
                                                     else "no cancel route")
                done = True
            if done:
                h["summary"] = summary
                left.remove(j)
                yield j
        if left:
            print(f"  waiting on {len(left)}: {left}", flush=True)
            time.sleep(120)


def fetch(j):
    h = jobs[j]
    verdicts = {}
    for cid, (text, meta) in batch_api.results_with_retry(h).items():
        obj = _extract_json(text)
        v = parse_verdict(obj) if obj else None
        if v:
            if meta.get("reasoning"):
                v["reasoning_trace"] = meta["reasoning"][:50000]
            verdicts[h["idmap"][cid]] = v
    return verdicts


def settle(group):
    """Gate each judge in `group` against the checkpoint plus the whole group's
    results (the Anthropic group is its own reference panel), merge passes."""
    ck = json.loads(CKPT.read_text())
    ref = dict(ck)
    for v in group.values():
        ref.update(v)
    for j, verdicts in group.items():
        bad, m = gate(j, verdicts, jobs[j]["n"], ref)
        if bad:
            quarantine[j] = {"why": bad, "metrics": m, "verdicts": verdicts}
            save(QUAR, quarantine)
            print(f"QUARANTINED {j} ({jobs[j]['summary']}): {bad} {m}", flush=True)
        else:
            before = len(ck)
            ck.update(verdicts)
            save(CKPT, ck)
            print(f"MERGED {j} ({jobs[j]['summary']}): {m} — ckpt {before} -> "
                  f"{len(json.loads(CKPT.read_text()))}", flush=True)
        jobs[j]["collected"] = True
        save(JOBS, jobs)


if os.environ.get("DRY"):
    sys.exit(0)
anth_jobs = [j for j in jobs if jobs[j]["provider"] == "anthropic"]
done_anth = list(wait_done(anth_jobs))
settle({j: fetch(j) for j in done_anth})
for j in wait_done([j for j in jobs if j not in anth_jobs]):
    settle({j: fetch(j)})
print("ALL BATCH LEGS COLLECTED", flush=True)
save(CKPT.with_name(f"{OUT}_ckpt.backup.json"), json.loads(CKPT.read_text()))

if os.environ.get("STOP_BEFORE_SWEEP"):
    print("STOPPED BEFORE SWEEP", flush=True)
    sys.exit(0)

# ---- realtime sweep + aggregation -------------------------------------------
wire_reps_cache(ROOT, REPS)
judge_corpus(DATA, panel, prompt="express-yourself", workers=24, exclude_self=False,
             include_note=False, out_name=OUT)
