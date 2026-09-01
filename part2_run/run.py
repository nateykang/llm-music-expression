"""Part 2 self-preference: all 42 corpus models judge every express-yourself
piece (ABC + codegen final batches), self included. Resumable via the standard
content-keyed checkpoint in part2_run/analysis/."""
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

BATCHES = [
    "20260819_201530__models_40_prompts_3",   # ABC arm (40 models)
    "20260819_203512__models_2_prompts_3",    # ABC top-up (grok-4.6 pair)
    "20260820_061907__models_42_prompts_3",   # codegen arm (42 models)
]
data = ROOT / "part2_run/data"
data.mkdir(parents=True, exist_ok=True)
for b in BATCHES:
    link = data / b
    if not link.exists():
        link.symlink_to(ROOT / "docs/data" / b)

judges = sorted({p["model"] for b in BATCHES
                 for p in json.loads(
                     (ROOT / "docs/data" / b / "data.json").read_text())["pieces"]
                 if p.get("ok")})
print(f"panel: {len(judges)} judges: {judges}", flush=True)

judge_corpus(data, judges, prompt="express-yourself", workers=24,
             exclude_self=False, include_note=False, out_name="part2_selfpref")
