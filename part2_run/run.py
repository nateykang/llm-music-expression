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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BATCHES, judges_from  # noqa: E402
data = ROOT / "part2_run/data"
data.mkdir(parents=True, exist_ok=True)
for b in BATCHES:
    link = data / b
    if not link.exists():
        link.symlink_to(ROOT / "docs/data" / b)

judges = judges_from({p["model"] for b in BATCHES
                 for p in json.loads(
                     (ROOT / "docs/data" / b / "data.json").read_text())["pieces"]
                 if p.get("ok")})
print(f"panel: {len(judges)} judges: {judges}", flush=True)

judge_corpus(data, judges, prompt="express-yourself", workers=int(os.environ.get("WORKERS", "24")),
             exclude_self=False, include_note=False, out_name="part2_selfpref")
