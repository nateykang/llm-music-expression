"""Realtime fill for (piece, judge) pairs missing from the final Part 2 raw
file — the handful of verdicts a judge failed three times in the closing
sweep. Patches part2_selfpref_raw.json in place (backup alongside) and
rebuilds the CSV from it with the same aggregation judge_corpus uses.

    python part2_run/fill_missing.py [judge ...]   (default: kimi-k3-thinking)
    python part2_run/fill_missing.py --dry           rebuild the CSV only, to DRY_CSV
"""
import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from llm_music.judge import judge_piece, panel_row, write_judge_csv  # noqa: E402

ANALYSIS = ROOT / "part2_run/analysis"
RAW, CSV = ANALYSIS / "part2_selfpref_raw.json", ANALYSIS / "part2_selfpref.csv"
DATA = ROOT / "docs/data"


def piece_dict(p: dict, manifests: dict) -> dict:
    b = p["batch"]
    if b not in manifests:
        manifests[b] = json.loads((DATA / b / "data.json").read_text(encoding="utf-8"))["pieces"]
    for pc in manifests[b]:
        if pc.get("ok") and (pc["model"], pc["prompt"], pc.get("mode", ""), pc.get("sample", 0)) == \
                (p["model"], p["prompt"], p["mode"], p["sample"]):
            return pc
    raise KeyError(f"piece not in manifest: {p['batch']}|{p['model']}|{p['mode']}|{p['sample']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("judges", nargs="*", default=["kimi-k3-thinking"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--attempts", type=int, default=5)
    args = ap.parse_args()
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    if args.dry:
        out = Path(os.environ.get("DRY_CSV", ANALYSIS / "part2_selfpref.dry.csv"))
        write_judge_csv(out, [panel_row(p, p["batch"], p["panel"]) for p in raw])
        print(f"dry CSV rebuilt from raw → {out}")
        return 0
    from reps import wire_reps_cache
    from llm_music.models import get_client
    wire_reps_cache(ROOT)
    clients, manifests, filled, failed = {}, {}, [], []
    for p in raw:
        for j in args.judges:
            if j in p["panel"]:
                continue
            who = f"{j} on {p['batch']}|{p['model']}|{p['mode']}|{p['sample']}"
            clients.setdefault(j, get_client(j))
            v = judge_piece(clients[j], piece_dict(p, manifests), DATA / p["batch"], attempts=args.attempts)
            if v:
                p["panel"][j] = v
            (filled if v else failed).append(who)
            print(("filled " if v else "FAILED ") + who, flush=True)
            if v:
                print("   scores:", {k: v[k]["score"] for k in v if isinstance(v[k], dict)},
                      "| label:", v.get("emotion_label"), "| trace chars:", len(v.get("reasoning_trace", "")))
    if filled:
        shutil.copy2(RAW, RAW.with_suffix(".json.bak"))
        RAW.write_text(json.dumps(raw, indent=1), encoding="utf-8")
        write_judge_csv(CSV, [panel_row(p, p["batch"], p["panel"]) for p in raw])
    print(f"filled {len(filled)}, failed {len(failed)}; total verdicts "
          f"{sum(len(p['panel']) for p in raw):,} over {len(raw)} pieces")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
