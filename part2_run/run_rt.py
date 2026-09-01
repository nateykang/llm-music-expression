"""Realtime leg for arms without a working batch route: the 14 OpenRouter arms
lacking :batch endpoints plus the 4 native-OpenAI arms. Uses its OWN checkpoint
(part2_rt), pre-seeded from the main one so already-banked keys are skipped,
and purges failed (None) entries on resume so a transient credit/API outage
never becomes a permanent skip. Merge into the main checkpoint afterwards."""
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", "/Users/nathanielkang/llm-music-expression"))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from llm_music.judge import judge_corpus  # noqa: E402
import llm_music.judge as _J  # noqa: E402

# Serve precomputed representations (identical bytes to what the batch legs
# used) instead of re-rendering per judge call — throughput only, the judge
# input is unchanged.
_reps = json.loads((ROOT / "part2_run/reps_cache.json").read_text())
_orig_repr = _J.representation


def _cached_repr(piece, batch_dir):
    key = (f"{batch_dir.name}|{piece['model']}|{piece['prompt']}|"
           f"{piece.get('mode', '')}|{piece.get('sample', 0)}")
    hit = _reps.get(key)
    if hit and hit[1] is not None:
        return hit[0], hit[1]
    return _orig_repr(piece, batch_dir)


_J.representation = _cached_repr
print(f"reps cache wired into judge path ({len(_reps)} entries)", flush=True)

BATCHES = ["20260819_201530__models_40_prompts_3", "20260819_203512__models_2_prompts_3",
           "20260820_061907__models_42_prompts_3"]
ARMS = ["gpt-5", "gpt-5-thinking", "gpt-5.1", "gpt-5.1-thinking", "gpt-5.4", "gpt-5.4-thinking",
        "gpt-5.6", "gpt-5.6-thinking", "grok-4.3", "grok-4.3-thinking", "grok-4.6", "grok-4.6-thinking",
        "kimi-k2", "kimi-k2-thinking", "gpt-5.5", "gpt-5.5-thinking", "gpt-5.2", "gpt-5.2-thinking",
        "opus-4.1", "opus-4.1-thinking"]
data = ROOT / "part2_run/data"
data.mkdir(parents=True, exist_ok=True)
for b in BATCHES:
    link = data / b
    if not link.exists():
        link.symlink_to(ROOT / "docs/data" / b)
analysis = ROOT / "part2_run/analysis"
main_ck, rt_ck = analysis / "part2_selfpref_ckpt.json", analysis / "part2_rt_ckpt.json"
rt = json.loads(rt_ck.read_text()) if rt_ck.exists() else {}
purged = sum(1 for v in rt.values() if not v)
rt = {k: v for k, v in rt.items() if v}
if main_ck.exists():
    for k, v in json.loads(main_ck.read_text()).items():
        if v and k not in rt:
            rt[k] = v
rt_ck.write_text(json.dumps(rt))
print(f"part2_rt checkpoint seeded: {len(rt)} banked keys ({purged} failed entries purged for retry)", flush=True)
judge_corpus(data, ARMS, prompt="express-yourself", workers=int(os.environ.get("WORKERS", "16")),
             exclude_self=False, include_note=False, out_name="part2_rt")
