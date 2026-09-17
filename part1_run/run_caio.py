"""Caio common-set transposition run: 9 model-comparison codegen pieces x 11
shifts x 40 judges (~$85). Batched where the route is proven (Anthropic native;
OpenRouter :batch for Gemini/kimi-k3 with the config-B no-response_format rule
for google/ slugs); everything else realtime; batch failures swept realtime.
One checkpoint: part1_run/caio_ckpt.json, keys batch|model|prompt|codegen|sample|shift|judge.
Request bodies mirror part2_run/batch_{anthropic,openrouter}.py, which mirror the clients.
"""
import json
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402
import httpx  # noqa: E402

import llm_music.judge as J  # noqa: E402
from llm_music.judge import _extract_json, _system, build_user, parse_verdict  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.models.registry import MODEL_REGISTRY  # noqa: E402

MAN = json.loads((ROOT / "part1_run/caio_manifest.json").read_text())
REPS = json.loads((ROOT / "part1_run/caio_reps.json").read_text())
CKPT = ROOT / "part1_run/caio_ckpt.json"
MANIFEST = ROOT / "part1_run/caio_batch_manifest.json"
KIND = ("a note listing: a key/time/tempo header, then each part (labelled with its "
        "instrument) written bar by bar with notes as pitch+octave/duration, rests, "
        "dynamics [p]/[f], articulations, and technical directions")
API = "https://openrouter.ai/api/beta/batches"
H = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
TERMINAL = {"completed", "failed", "expired", "cancelled"}

judges = [j for j in json.loads((ROOT / "docs/analysis/selfpref_v3.json").read_text())["judges_order"]
          if not j.startswith("opus-4.1")]
rep_keys = [f"{MAN['batch']}|{p['model']}|{p['prompt']}|codegen|{p['sample']}|{sh}"
            for p in MAN["picked_flat"] for sh in MAN["shifts"]]
pdict = {f"{MAN['batch']}|{p['model']}|{p['prompt']}|codegen|{p['sample']}": p for p in MAN["picked_flat"]}
ANTH = [j for j in judges if MODEL_REGISTRY[j][0] == "anthropic"]
ORB = [j for j in judges if MODEL_REGISTRY[j][0] == "openrouter"
       and (MODEL_REGISTRY[j][1].startswith("google/") or MODEL_REGISTRY[j][1].startswith("moonshotai/kimi-k3"))]
RT = [j for j in judges if j not in ANTH and j not in ORB]
system = _system(False)
lock = threading.Lock()
ck = json.loads(CKPT.read_text()) if CKPT.exists() else {}
ck = {k: v for k, v in ck.items() if v}


def save():
    tmp = CKPT.with_suffix(".tmp"); tmp.write_text(json.dumps(ck)); tmp.replace(CKPT)


def user_msg(rk):
    stem = rk.rsplit("|", 1)[0]
    return build_user(pdict[stem], KIND, REPS[rk])


manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"anth": [], "or": []}
aclient = anthropic.Anthropic(timeout=600.0, max_retries=3)

# --- submit Anthropic batches ---
have = {m["judge"] for m in manifest["anth"]}
for judge in ANTH:
    if judge in have:
        continue
    jc = get_client(judge)
    reqs, idmap = [], {}
    for i, rk in enumerate(rep_keys):
        key = f"{rk}|{judge}"
        if key in ck:
            continue
        params = {"model": jc.model_id, "max_tokens": jc.max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user_msg(rk)}]}
        if jc.effort:
            params["output_config"] = {"effort": jc.effort}
        if jc.thinking:
            th = dict(jc.thinking)
            if th.get("type") == "adaptive":
                th.setdefault("display", "summarized")
            params["thinking"] = th
        cid = f"c{i:04d}"
        idmap[cid] = key
        reqs.append({"custom_id": cid, "params": params})
    if not reqs:
        continue
    mb = aclient.messages.batches.create(requests=reqs)
    manifest["anth"].append({"id": mb.id, "judge": judge, "n": len(reqs), "idmap": idmap, "collected": False})
    MANIFEST.write_text(json.dumps(manifest))
    print(f"ANTH SUBMITTED {judge}: {len(reqs)} -> {mb.id}", flush=True)

# --- submit OpenRouter batches ---
have = {m["judge"] for m in manifest["or"]}
for judge in ORB:
    if judge in have:
        continue
    jc = get_client(judge)
    slug = jc.model_id
    reqs, idmap = [], {}
    for i, rk in enumerate(rep_keys):
        key = f"{rk}|{judge}"
        if key in ck:
            continue
        body = {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user_msg(rk)}],
                "max_tokens": min(jc.max_output_tokens, 8000 if judge.endswith("-thinking") else 16000)}
        if not slug.startswith("google/"):
            body["response_format"] = {"type": "json_object"}
        if jc.reasoning:
            body["reasoning"] = jc.reasoning
        cid = f"c{i:04d}"
        idmap[cid] = key
        reqs.append({"custom_id": cid, "body": body})
    if not reqs:
        continue
    payload = {"endpoint": "/v1/chat/completions", "model": slug, "requests": reqs}
    r = httpx.post(API, headers=H, content=json.dumps(payload), timeout=600)
    if r.status_code not in (200, 201, 202):
        print(f"OR SUBMIT FAILED {judge}: {r.status_code} {r.text[:200]} — left for realtime sweep", flush=True)
        continue
    manifest["or"].append({"id": r.json()["id"], "judge": judge, "n": len(reqs), "idmap": idmap, "collected": False})
    MANIFEST.write_text(json.dumps(manifest))
    print(f"OR SUBMITTED {judge}: {len(reqs)} -> {manifest['or'][-1]['id']}", flush=True)

# --- realtime leg while batches cook ---
J.representation = lambda piece, bd: (KIND, REPS[piece["_rep_key"]])


def rt_pass(arms, label):
    jobs = []
    for judge in arms:
        for rk in rep_keys:
            key = f"{rk}|{judge}"
            if key not in ck:
                stem = rk.rsplit("|", 1)[0]
                pc = dict(pdict[stem]); pc["_rep_key"] = rk
                jobs.append((key, judge, pc))
    if not jobs:
        return
    print(f"{label}: {len(jobs)} realtime calls", flush=True)
    clients = {}
    done = 0
    def work(job):
        key, judge, pc = job
        with lock:
            c = clients.setdefault(judge, get_client(judge))
        return key, J.judge_piece(c, pc, ROOT / "docs/data" / MAN["batch"])
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "12"))) as ex:
        for fut in as_completed([ex.submit(work, jb) for jb in jobs]):
            key, v = fut.result()
            with lock:
                if v:
                    ck[key] = v
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    save(); print(f"  [{done}/{len(jobs)}]", flush=True)
    save()


rt_pass(RT, "realtime leg")

# --- poll + collect both batch legs ---
pending = {("anth", m["id"]): m for m in manifest["anth"] if not m.get("collected")}
pending.update({("or", m["id"]): m for m in manifest["or"] if not m.get("collected")})
while pending:
    time.sleep(90)
    for (kind_, bid) in list(pending):
        m = pending[(kind_, bid)]
        ok = bad = 0
        try:
            if kind_ == "anth":
                mb = aclient.messages.batches.retrieve(bid)
                if mb.processing_status != "ended":
                    c = mb.request_counts
                    print(f"  anth {m['judge']}: {c.succeeded}/{m['n']} ({c.errored} err)", flush=True)
                    continue
                for res in aclient.messages.batches.results(bid):
                    key = m["idmap"].get(res.custom_id)
                    if key is None or res.result.type != "succeeded":
                        bad += key is not None
                        continue
                    raw = "".join(b.text for b in res.result.message.content if b.type == "text")
                    v = parse_verdict(_extract_json(raw) or {})
                    if v:
                        ck[key] = v; ok += 1
                    else:
                        bad += 1
            else:
                d = httpx.get(f"{API}/{bid}", headers=H, timeout=600).json()
                if d.get("status") not in TERMINAL:
                    rc = d.get("request_counts") or {}
                    print(f"  or {m['judge']}: {d.get('status')} {rc.get('completed', 0)}/{m['n']}", flush=True)
                    continue
                for res in d.get("results") or []:
                    key = m["idmap"].get(res.get("custom_id"))
                    resp = res.get("response") or {}
                    if key is None or res.get("error") or resp.get("status_code") != 200:
                        bad += key is not None
                        continue
                    msg = ((resp.get("body") or {}).get("choices") or [{}])[0].get("message") or {}
                    content = msg.get("content") or msg.get("reasoning") or ""
                    v = parse_verdict(_extract_json(content) or {})
                    if v:
                        ck[key] = v; ok += 1
                    else:
                        bad += 1
        except Exception as e:  # noqa: BLE001
            print(f"  poll error {m['judge']}: {e}", flush=True)
            continue
        with lock:
            save()
        m["collected"] = True
        MANIFEST.write_text(json.dumps(manifest))
        print(f"COLLECTED {kind_} {m['judge']}: {ok} ok, {bad} failed", flush=True)
        del pending[(kind_, bid)]

# --- final realtime sweep for anything a batch dropped ---
rt_pass(judges, "final sweep")
missing = [f"{rk}|{j}" for j in judges for rk in rep_keys if f"{rk}|{j}" not in ck]
print(f"DONE: {len(ck)} verdicts on disk, {len(missing)} missing of {len(judges) * len(rep_keys)}", flush=True)
if missing:
    print("missing:", missing[:10], flush=True)
