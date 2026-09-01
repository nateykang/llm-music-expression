"""Anthropic Message Batches path for the Part 2 run (50% off realtime).

Submits one batch per native-Anthropic judge arm covering every express-yourself
piece not already in the checkpoint, polls until ended, and merges verdicts into
the SAME content-keyed checkpoint the realtime runner uses — so judge_corpus
later treats them as cached. Requests mirror AnthropicClient.complete_full
exactly (model, max_tokens, thinking incl. display=summarized on adaptive,
output_config effort); results go through the shared parse_verdict.
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
import anthropic  # noqa: E402

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
MANIFEST = ROOT / "part2_run/batch_anthropic_manifest.json"

panel = sorted({p["model"] for b in BATCHES
                for p in json.loads((ROOT / "docs/data" / b / "data.json").read_text())["pieces"]
                if p.get("ok")})
ARMS = [n for n in panel if MODEL_REGISTRY[n][0] == "anthropic"]
print(f"native-Anthropic arms ({len(ARMS)}): {ARMS}", flush=True)

tasks = []
for b in BATCHES:
    man = json.loads((ROOT / "docs/data" / b / "data.json").read_text())
    for pc in man["pieces"]:
        if pc.get("ok") and pc.get("prompt") == "express-yourself":
            tasks.append((b, pc))
print(f"{len(tasks)} express-yourself pieces", flush=True)


def piece_key(bname, pc):
    return f"{bname}|{pc['model']}|{pc['prompt']}|{pc.get('mode', '')}|{pc.get('sample', 0)}"


reps = {}
for i, (b, pc) in enumerate(tasks):
    reps[piece_key(b, pc)] = representation(pc, ROOT / "docs/data" / b)
    if (i + 1) % 200 == 0:
        print(f"  reps {i + 1}/{len(tasks)}", flush=True)

ckpt = json.loads(CKPT.read_text()) if CKPT.exists() else {}
system = _system(False)
client = anthropic.Anthropic(timeout=600.0, max_retries=3)

if MANIFEST.exists():
    manifest = json.loads(MANIFEST.read_text())
    print(f"resuming manifest with {len(manifest['batches'])} batches", flush=True)
else:
    manifest = {"batches": []}
    idx = 0
    for judge in ARMS:
        jc = get_client(judge)
        reqs, idmap = [], {}
        for b, pc in tasks:
            pk = piece_key(b, pc)
            key = f"{pk}|{judge}"
            if key in ckpt:
                continue
            kind, text = reps[pk]
            if text is None:
                continue
            params = {
                "model": jc.model_id,
                "max_tokens": jc.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": build_user(pc, kind, text)}],
            }
            if jc.effort:
                params["output_config"] = {"effort": jc.effort}
            if jc.thinking:
                th = dict(jc.thinking)
                if th.get("type") == "adaptive":
                    th.setdefault("display", "summarized")
                params["thinking"] = th
            cid = f"q{idx:06d}"
            idx += 1
            idmap[cid] = key
            reqs.append({"custom_id": cid, "params": params})
        if not reqs:
            print(f"{judge}: nothing to do (all cached)", flush=True)
            continue
        mb = client.messages.batches.create(requests=reqs)
        manifest["batches"].append({"id": mb.id, "judge": judge, "n": len(reqs),
                                    "idmap": idmap, "collected": False})
        MANIFEST.write_text(json.dumps(manifest))
        print(f"SUBMITTED {judge}: {len(reqs)} requests -> {mb.id}", flush=True)

pending = {m["id"]: m for m in manifest["batches"] if not m.get("collected")}
while pending:
    time.sleep(60)
    for bid in list(pending):
        m = pending[bid]
        try:
            mb = client.messages.batches.retrieve(bid)
        except Exception as e:  # noqa: BLE001
            print(f"  poll error {m['judge']}: {e}", flush=True)
            continue
        c = mb.request_counts
        if mb.processing_status != "ended":
            print(f"  {m['judge']}: {c.succeeded} ok / {c.errored} err / {m['n']} total",
                  flush=True)
            continue
        ok = bad = 0
        ck = json.loads(CKPT.read_text()) if CKPT.exists() else {}
        for res in client.messages.batches.results(bid):
            key = m["idmap"].get(res.custom_id)
            if key is None:
                continue
            if res.result.type != "succeeded":
                bad += 1
                continue
            msg = res.result.message
            raw = "".join(blk.text for blk in msg.content if blk.type == "text")
            trace = "\n".join(blk.thinking for blk in msg.content
                              if blk.type == "thinking" and getattr(blk, "thinking", ""))
            obj = _extract_json(raw)
            v = parse_verdict(obj) if obj else None
            if v:
                if trace:
                    v["reasoning_trace"] = trace[:50000]
                ck[key] = v
                ok += 1
            else:
                bad += 1
        tmp = CKPT.with_suffix(".tmp")
        tmp.write_text(json.dumps(ck))
        tmp.replace(CKPT)
        m["collected"] = True
        MANIFEST.write_text(json.dumps(manifest))
        print(f"COLLECTED {m['judge']}: {ok} verdicts merged into checkpoint, "
              f"{bad} failed (left for realtime retry)", flush=True)
        del pending[bid]
print("ALL ANTHROPIC BATCHES DONE", flush=True)
