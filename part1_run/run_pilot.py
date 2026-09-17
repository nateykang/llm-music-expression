"""Part 1 pilot: each arm judges its OWN 5 codegen pieces at 11 transpositions
(Sara's design, 2026-09-17). Self-judge only; ~2,200 realtime verdicts (~$40).
Checkpointed like Part 2 (content key + |shift|judge); resumable."""
import json
import os
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import llm_music.judge as J  # noqa: E402
from llm_music.models import get_client  # noqa: E402

MAN = json.loads((ROOT / "part1_run/pilot_manifest.json").read_text())
REPS = json.loads((ROOT / "part1_run/pilot_reps.json").read_text())
KIND = ("a note listing: a key/time/tempo header, then each part (labelled with its "
        "instrument) written bar by bar with notes as pitch+octave/duration, rests, "
        "dynamics [p]/[f], articulations, and technical directions")
J.representation = lambda piece, bd: (KIND, REPS[piece["_rep_key"]])

batch_dir = ROOT / "docs/data" / MAN["batch"]
pieces = {(p["model"], p.get("sample", 0)): p
          for p in json.loads((batch_dir / "data.json").read_text())["pieces"]
          if p["prompt"] == "express-yourself" and p.get("mode") == "codegen"}
ck_path = ROOT / "part1_run/pilot_ckpt.json"
ck = json.loads(ck_path.read_text()) if ck_path.exists() else {}
ck = {k: v for k, v in ck.items() if v}  # retry failures on resume

jobs = []
for arm, picks in MAN["picked"].items():
    for p in picks:
        for sh in MAN["shifts"]:
            key = f"{MAN['batch']}|{arm}|express-yourself|codegen|{p['sample']}|{sh}|{arm}"
            if key in ck:
                continue
            pc = dict(pieces[(arm, p["sample"])])
            pc["_rep_key"] = key.rsplit("|", 1)[0]
            jobs.append((key, arm, pc))
print(f"pilot: {len(jobs)} self-judge calls to do ({len(ck)} cached)", flush=True)
clients, lock, done = {}, threading.Lock(), 0


def work(job):
    key, arm, pc = job
    with lock:
        c = clients.setdefault(arm, get_client(arm))
    return key, J.judge_piece(c, pc, batch_dir)


with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "12"))) as ex:
    for fut in as_completed([ex.submit(work, jb) for jb in jobs]):
        key, v = fut.result()
        with lock:
            ck[key] = v
            done += 1
            if done % 20 == 0 or done == len(jobs):
                tmp = ck_path.with_suffix(".tmp"); tmp.write_text(json.dumps(ck)); tmp.replace(ck_path)
                print(f"  [{done}/{len(jobs)}] failed so far: {sum(1 for x in ck.values() if not x)}", flush=True)
ck_path.with_suffix(".tmp").write_text(json.dumps(ck)); ck_path.with_suffix(".tmp").replace(ck_path)
print(f"done: {sum(1 for x in ck.values() if x)} verdicts, {sum(1 for x in ck.values() if not x)} failed → {ck_path}")
