"""OpenAI Batch API leg for the Part 2 run (50% off realtime).

One batch per native-OpenAI arm via /v1/responses, mirroring
OpenAIClient.complete_full exactly (instructions/input, max_output_tokens,
reasoning effort + summary=auto for reasoning arms). Results bank into a
sidecar file and merge into the shared checkpoint only once batch_anthropic.py
has exited, avoiding concurrent read-modify-write on the checkpoint.
"""
import io
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", "/Users/nathanielkang/llm-music-expression"))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import openai  # noqa: E402

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
MANIFEST = ROOT / "part2_run/batch_openai_manifest.json"
SIDECAR = ROOT / "part2_run/batch_openai_collected.json"
REPS_CACHE = ROOT / "part2_run/reps_cache.json"

panel = sorted({p["model"] for b in BATCHES
                for p in json.loads((ROOT / "docs/data" / b / "data.json").read_text())["pieces"]
                if p.get("ok")})
ARMS = [n for n in panel if MODEL_REGISTRY[n][0] == "openai"]
print(f"native-OpenAI arms ({len(ARMS)}): {ARMS}", flush=True)

tasks = []
for b in BATCHES:
    man = json.loads((ROOT / "docs/data" / b / "data.json").read_text())
    for pc in man["pieces"]:
        if pc.get("ok") and pc.get("prompt") == "express-yourself":
            tasks.append((b, pc))
print(f"{len(tasks)} express-yourself pieces", flush=True)


def piece_key(bname, pc):
    return f"{bname}|{pc['model']}|{pc['prompt']}|{pc.get('mode', '')}|{pc.get('sample', 0)}"


if REPS_CACHE.exists():
    reps = json.loads(REPS_CACHE.read_text())
    print(f"loaded {len(reps)} cached reps", flush=True)
else:
    reps = {}
for i, (b, pc) in enumerate(tasks):
    pk = piece_key(b, pc)
    if pk not in reps:
        reps[pk] = list(representation(pc, ROOT / "docs/data" / b))
        if (i + 1) % 200 == 0:
            print(f"  reps {i + 1}/{len(tasks)}", flush=True)
REPS_CACHE.write_text(json.dumps(reps))
print("reps ready (cached)", flush=True)

ckpt = json.loads(CKPT.read_text()) if CKPT.exists() else {}
system = _system(False)
client = openai.OpenAI(timeout=600.0, max_retries=3)

if MANIFEST.exists():
    manifest = json.loads(MANIFEST.read_text())
    print(f"resuming manifest with {len(manifest['batches'])} batches", flush=True)
else:
    manifest = {"batches": []}
    idx = 0
    for judge in ARMS:
        jc = get_client(judge)
        lines, idmap = [], {}
        for b, pc in tasks:
            pk = piece_key(b, pc)
            key = f"{pk}|{judge}"
            if key in ckpt:
                continue
            kind, text = reps[pk]
            if text is None:
                continue
            body = {
                "model": jc.model_id,
                "instructions": system,
                "input": build_user(pc, kind, text),
                "max_output_tokens": jc.max_output_tokens,
            }
            if jc.reasoning_effort:
                body["reasoning"] = {"effort": jc.reasoning_effort}
                if jc.reasoning_effort != "none":
                    body["reasoning"]["summary"] = "auto"
            cid = f"q{idx:06d}"
            idx += 1
            idmap[cid] = key
            lines.append(json.dumps({"custom_id": cid, "method": "POST",
                                     "url": "/v1/responses", "body": body}))
        if not lines:
            print(f"{judge}: nothing to do (all cached)", flush=True)
            continue
        buf = io.BytesIO("\n".join(lines).encode())
        buf.name = f"part2_{judge}.jsonl"
        f = client.files.create(file=buf, purpose="batch")
        mb = client.batches.create(input_file_id=f.id, endpoint="/v1/responses",
                                   completion_window="24h")
        manifest["batches"].append({"id": mb.id, "judge": judge, "n": len(lines),
                                    "input_file": f.id, "idmap": idmap,
                                    "collected": False})
        MANIFEST.write_text(json.dumps(manifest))
        print(f"SUBMITTED {judge}: {len(lines)} requests -> {mb.id}", flush=True)


def out_text(body):
    parts = [c.get("text", "") for item in body.get("output", [])
             if item.get("type") == "message"
             for c in item.get("content", []) if c.get("type") == "output_text"]
    return "".join(parts)


def out_reasoning(body):
    parts = [s.get("text", "") for item in body.get("output", [])
             if item.get("type") == "reasoning"
             for s in item.get("summary", []) or []]
    return "\n".join(p for p in parts if p)


sidecar = json.loads(SIDECAR.read_text()) if SIDECAR.exists() else {}
pending = {m["id"]: m for m in manifest["batches"] if not m.get("collected")}
TERMINAL = {"completed", "failed", "expired", "cancelled"}
while pending:
    time.sleep(60)
    for bid in list(pending):
        m = pending[bid]
        try:
            mb = client.batches.retrieve(bid)
        except Exception as e:  # noqa: BLE001
            print(f"  poll error {m['judge']}: {e}", flush=True)
            continue
        rc = mb.request_counts
        if mb.status not in TERMINAL:
            print(f"  {m['judge']}: {mb.status} — {rc.completed}/{rc.total} "
                  f"({rc.failed} failed)", flush=True)
            continue
        ok = bad = 0
        for fid in filter(None, [mb.output_file_id, mb.error_file_id]):
            content = client.files.content(fid).text
            for line in content.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = m["idmap"].get(rec.get("custom_id"))
                if key is None:
                    continue
                resp = rec.get("response") or {}
                body = resp.get("body") or {}
                if resp.get("status_code") != 200 or rec.get("error"):
                    bad += 1
                    continue
                raw = out_text(body)
                obj = _extract_json(raw)
                v = parse_verdict(obj) if obj else None
                if v:
                    trace = out_reasoning(body)
                    if trace:
                        v["reasoning_trace"] = trace[:50000]
                    sidecar[key] = v
                    ok += 1
                else:
                    bad += 1
        tmp = SIDECAR.with_suffix(".tmp")
        tmp.write_text(json.dumps(sidecar))
        tmp.replace(SIDECAR)
        m["collected"] = True
        MANIFEST.write_text(json.dumps(manifest))
        print(f"COLLECTED {m['judge']} ({mb.status}): {ok} verdicts to sidecar, "
              f"{bad} failed (left for realtime retry)", flush=True)
        del pending[bid]

while subprocess.run(["pgrep", "-f", "batch_anthropic.py"],
                     capture_output=True).returncode == 0:
    print("waiting for batch_anthropic.py to exit before merging...", flush=True)
    time.sleep(60)
ck = json.loads(CKPT.read_text()) if CKPT.exists() else {}
before = len(ck)
ck.update(sidecar)
tmp = CKPT.with_suffix(".tmp")
tmp.write_text(json.dumps(ck))
tmp.replace(CKPT)
print(f"MERGED {len(sidecar)} openai verdicts into checkpoint "
      f"({before} -> {len(ck)})", flush=True)
print("ALL OPENAI BATCHES DONE", flush=True)
