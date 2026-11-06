"""Roster-extension generation: ARMS × 3 prompts × 20 samples, ABC + codegen —
the v3 corpus protocol. Native-provider first attempts go through the Batch API;
OpenRouter arms run live. The standard attempt loop then parses first attempts
and makes any retries (which carry the prior error) live. Resumable: batch
handles and collected responses are persisted.

Env: RUN_DIR (default part3_run), ARMS (comma-separated), GEN_WORKERS."""
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
from llm_music import batch_api  # noqa: E402
from llm_music.cli import _run_matrix  # noqa: E402
from llm_music.generate import first_attempt_prompts  # noqa: E402
from llm_music.models import get_client  # noqa: E402

RUN = ROOT / os.environ.get("RUN_DIR", "part3_run")
RUN.mkdir(parents=True, exist_ok=True)
ARMS = os.environ.get("ARMS", "gpt-6.1-sol,gpt-6.1-sol-thinking").split(",")
WORKERS = int(os.environ.get("GEN_WORKERS", "8"))
PROMPTS = ["express-yourself", "uniquely-you", "emotional-state"]
MODES = ["abc", "codegen"]
SAMPLES = 20
DEADLINE_H = float(os.environ.get("BATCH_DEADLINE_H", "8"))
STATE = RUN / "gen_state.json"
FIRST = RUN / "gen_first_responses.json"


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


state = json.loads(STATE.read_text()) if STATE.exists() else {"handles": {}, "out": {}}
first = json.loads(FIRST.read_text()) if FIRST.exists() else {}


def cid(mode, m, p, s):
    return f"{mode}|{m}|{p}|{s}"


for m in ARMS:
    if m in state["handles"] or type(get_client(m)).__name__ == "OpenRouterClient":
        continue
    reqs, idmap = {}, {}
    for mode in MODES:
        for p in PROMPTS:
            system, user = first_attempt_prompts(p, mode)
            for s in range(SAMPLES):
                short = f"q{len(idmap):05d}"  # Anthropic custom_ids reject | and .
                idmap[short] = cid(mode, m, p, s)
                reqs[short] = (system, user, True)
    h = batch_api.submit(get_client(m), reqs, label=f"{RUN.name}_gen_{m}")
    h.update(submitted=time.time(), idmap=idmap)
    state["handles"][m] = h
    save(STATE, state)
    print(f"SUBMITTED {m}: {len(reqs)} first attempts -> {h['id']}", flush=True)

for m, h in state["handles"].items():
    if h.get("collected"):
        continue
    while True:
        try:
            done, summary = batch_api.poll(h)
        except Exception as e:  # noqa: BLE001
            print(f"  poll error {m}: {e}", flush=True)
            time.sleep(120)
            continue
        print(f"  {m}: {summary}", flush=True)
        if done:
            break
        if time.time() - h["submitted"] > DEADLINE_H * 3600:
            batch_api.cancel_and_wait(h)
            print(f"  {m}: deadline passed, cancelled — unfinished cells run live",
                  flush=True)
            break
        time.sleep(120)
    got = batch_api.results_with_retry(h)
    first.update({h["idmap"][k]: text for k, (text, _) in got.items()})
    save(FIRST, first)
    n_saved = sum(k.split("|")[1] == m for k in json.loads(FIRST.read_text()))
    h["collected"] = True
    save(STATE, state)
    print(f"COLLECTED {m}: {len(got)}/{h['n']} responses ({n_saved} on disk)", flush=True)

for mode in MODES:
    if state["out"].get(mode, {}).get("done"):
        continue
    fr = {(m, p, s): first[cid(mode, m, p, s)]
          for m in ARMS for p in PROMPTS for s in range(SAMPLES)
          if cid(mode, m, p, s) in first}
    print(f"\n== {mode}: {len(fr)} batched first attempts, "
          f"{len(ARMS) * len(PROMPTS) * SAMPLES - len(fr)} live", flush=True)
    batch, results = _run_matrix(ARMS, PROMPTS, mode, 5, SAMPLES, workers=WORKERS,
                                 first_responses=fr)
    state["out"][mode] = {"batch": str(batch), "done": True}
    save(STATE, state)
    print(f"{mode}: {sum(r.ok for r in results)}/{len(results)} ok -> {batch}", flush=True)
print("GENERATION DONE", flush=True)
