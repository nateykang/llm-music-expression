"""OpenRouter beta Batch API leg for the Part 2 run (50% off realtime).

One batch per OpenRouter-routed arm whose model exposes a :batch endpoint
(checked live against the catalog). Inline `requests` array — no file upload.
Bodies mirror OpenRouterClient.complete_full (messages, max_tokens,
response_format json_object, per-arm reasoning config). Results come back
inline on the status GET; verdicts merge into the shared checkpoint through
the common parse_verdict path. Arms without :batch support are skipped and
left for the realtime pass.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", "/Users/nathanielkang/llm-music-expression"))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import httpx  # noqa: E402

from llm_music.judge import (_extract_json, _system, build_user,  # noqa: E402
                             parse_verdict, representation)
from llm_music.models import get_client  # noqa: E402
from llm_music.models.registry import MODEL_REGISTRY  # noqa: E402

BATCHES = [
    "20260819_201530__models_40_prompts_3",
    "20260819_203512__models_2_prompts_3",
    "20260820_061907__models_42_prompts_3",
]
CKPT = ROOT / "part2_run/analysis/part2_selfpref_ckpt.json"
MANIFEST = ROOT / "part2_run/batch_openrouter_manifest.json"
REPS_CACHE = ROOT / "part2_run/reps_cache.json"
API = "https://openrouter.ai/api/beta/batches"
H = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
     "Content-Type": "application/json"}
TERMINAL = {"completed", "failed", "expired", "cancelled"}

panel = sorted({p["model"] for b in BATCHES
                for p in json.loads((ROOT / "docs/data" / b / "data.json").read_text())["pieces"]
                if p.get("ok")})
catalog = {m["id"] for m in httpx.get("https://openrouter.ai/api/v1/models",
                                      timeout=60).json()["data"]}
# Native-OpenAI arms re-routed through OpenRouter's :batch endpoints (the org's
# own Batch API rejects file access); reasoning configs mirror the corpus's
# other gpt-5.x-via-OpenRouter arms.
OR_OVERRIDES = {
    "gpt-5.5": ("openai/gpt-5.5", {"enabled": False}),
    "gpt-5.5-thinking": ("openai/gpt-5.5", {"effort": "high"}),
    "gpt-5.2": ("openai/gpt-5.2", {"enabled": False}),
    "gpt-5.2-thinking": ("openai/gpt-5.2", {"effort": "high"}),
}


def or_slug(n):
    return OR_OVERRIDES[n][0] if n in OR_OVERRIDES else MODEL_REGISTRY[n][1]


candidates = [n for n in panel if n in OR_OVERRIDES or MODEL_REGISTRY[n][0] == "openrouter"]
ARMS = [n for n in candidates if f"{or_slug(n)}:batch" in catalog]
skipped = [n for n in candidates if n not in ARMS]
print(f"batchable OpenRouter arms ({len(ARMS)}): {ARMS}", flush=True)
print(f"no :batch endpoint, left for realtime ({len(skipped)}): {skipped}", flush=True)

tasks = []
for b in BATCHES:
    man = json.loads((ROOT / "docs/data" / b / "data.json").read_text())
    for pc in man["pieces"]:
        if pc.get("ok") and pc.get("prompt") == "express-yourself":
            tasks.append((b, pc))


def piece_key(bname, pc):
    return f"{bname}|{pc['model']}|{pc['prompt']}|{pc.get('mode', '')}|{pc.get('sample', 0)}"


reps = json.loads(REPS_CACHE.read_text()) if REPS_CACHE.exists() else {}
for b, pc in tasks:
    pk = piece_key(b, pc)
    if pk not in reps:
        reps[pk] = list(representation(pc, ROOT / "docs/data" / b))
REPS_CACHE.write_text(json.dumps(reps))
print(f"{len(tasks)} pieces, reps ready", flush=True)

ckpt = json.loads(CKPT.read_text()) if CKPT.exists() else {}
system = _system(False)


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m))


if MANIFEST.exists():
    manifest = json.loads(MANIFEST.read_text())
    print(f"resuming manifest with {len(manifest['batches'])} batches", flush=True)
else:
    manifest = {"batches": []}
CAP_BASE = int(os.environ.get("OR_BATCH_MAX_TOKENS_BASE", "0") or 0)
CAP_THINK = int(os.environ.get("OR_BATCH_MAX_TOKENS_THINK", "0") or 0)
idx = sum(m["n"] for m in manifest["batches"])
have = {m["judge"] for m in manifest["batches"]}
for judge in ARMS:
    if judge in have:
        continue
    if judge in OR_OVERRIDES:
        slug, reasoning, max_out = OR_OVERRIDES[judge][0], OR_OVERRIDES[judge][1], 16000
    else:
        jc = get_client(judge)
        slug, reasoning, max_out = jc.model_id, jc.reasoning, jc.max_output_tokens
    cap = CAP_THINK if judge.endswith("-thinking") else CAP_BASE
    max_tokens = min(max_out, cap) if cap else max_out
    reqs, idmap = [], {}
    for b, pc in tasks:
        pk = piece_key(b, pc)
        key = f"{pk}|{judge}"
        if key in ckpt:
            continue
        kind, text = reps[pk]
        if text is None:
            continue
        body = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": build_user(pc, kind, text)}],
            "max_tokens": max_tokens,
        }
        if not slug.startswith("google/"):
            # Google's batch service derives ONE flat schema per batch from
            # json_object and flattens nested objects (validated 2026-08-31);
            # prompt-only JSON is faithful there. Other backends honor json_object.
            body["response_format"] = {"type": "json_object"}
        if reasoning:
            body["reasoning"] = reasoning
        cid = f"q{idx:06d}"
        idx += 1
        idmap[cid] = key
        reqs.append({"custom_id": cid, "body": body})
    if not reqs:
        print(f"{judge}: nothing to do (all cached)", flush=True)
        continue
    payload = {"endpoint": "/v1/chat/completions", "model": slug,
               "requests": reqs}  # endpoint+model must precede requests
    r = httpx.post(API, headers=H, content=json.dumps(payload), timeout=600)
    if r.status_code not in (200, 201, 202):
        print(f"SUBMIT FAILED {judge}: {r.status_code} {r.text[:200]}", flush=True)
        continue
    bid = r.json()["id"]
    manifest["batches"].append({"id": bid, "judge": judge, "n": len(reqs),
                                "idmap": idmap, "collected": False,
                                "max_tokens": max_tokens})
    save_manifest(manifest)
    print(f"SUBMITTED {judge}: {len(reqs)} requests (max_tokens {max_tokens}) -> {bid}",
          flush=True)

pending = {m["id"]: m for m in manifest["batches"] if not m.get("collected")}
while pending:
    time.sleep(120)
    for bid in list(pending):
        m = pending[bid]
        try:
            d = httpx.get(f"{API}/{bid}", headers=H, timeout=600).json()
        except Exception as e:  # noqa: BLE001
            print(f"  poll error {m['judge']}: {e}", flush=True)
            continue
        st = d.get("status")
        rc = d.get("request_counts") or {}
        if st not in TERMINAL:
            print(f"  {m['judge']}: {st} — {rc.get('completed', 0)}/{rc.get('total', '?')} "
                  f"({rc.get('failed', 0)} failed)", flush=True)
            continue
        ok = bad = 0
        ck = json.loads(CKPT.read_text()) if CKPT.exists() else {}
        for res in d.get("results") or []:
            key = m["idmap"].get(res.get("custom_id"))
            if key is None:
                continue
            resp = res.get("response") or {}
            if res.get("error") or resp.get("status_code") != 200:
                bad += 1
                continue
            msg = ((resp.get("body") or {}).get("choices") or [{}])[0].get("message") or {}
            reasoning = msg.get("reasoning") or None
            if not reasoning and msg.get("reasoning_details"):
                parts = [x.get("text") or x.get("summary") or ""
                         for x in msg["reasoning_details"] if isinstance(x, dict)]
                reasoning = "\n".join(p for p in parts if p) or None
            content = msg.get("content") or ""
            if not content.strip():
                content = reasoning or ""
            obj = _extract_json(content)
            v = parse_verdict(obj) if obj else None
            if v:
                if reasoning:
                    v["reasoning_trace"] = reasoning[:50000]
                ck[key] = v
                ok += 1
            else:
                bad += 1
        tmp = CKPT.with_suffix(".tmp")
        tmp.write_text(json.dumps(ck))
        tmp.replace(CKPT)
        m["collected"] = True
        save_manifest(manifest)
        print(f"COLLECTED {m['judge']} ({st}): {ok} verdicts merged into checkpoint, "
              f"{bad} failed (left for realtime retry)", flush=True)
        del pending[bid]
print("ALL OPENROUTER BATCHES DONE", flush=True)
