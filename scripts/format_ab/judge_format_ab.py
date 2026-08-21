"""Representation A/B: judge the SAME pieces as note-listing vs converted ABC.

For every piece whose xml2abc conversion is event-exact, run the standard
judge panel twice — once on the code-gen representation (MusicXML ->
_score_to_text) and once on the converted ABC (the ABC judge path). Paired
within piece, so any score difference is the representation effect alone.

Checkpointed; safe to re-run. Writes nothing under docs/.
"""
import json
import logging
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os
ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", "/Users/nathanielkang/llm-music-expression"))
SCRATCH = Path(__file__).resolve().parent
XML2ABC = SCRATCH / "x2a/xml2abc_177/xml2abc.py"
PY = str(ROOT / ".venv/bin/python")

sys.path.insert(0, str(ROOT / "src"))
logging.disable(logging.WARNING)
import llm_music.config  # noqa: E402,F401  (load_dotenv — required before clients)
from llm_music.judge import judge_piece  # noqa: E402
from llm_music.models import get_client  # noqa: E402

JUDGES = ["gpt-5.5", "gemini-2.5-pro", "opus-4.8"]
WORKERS = 6

# --- select the event-exact pieces --------------------------------------------
verify = json.loads((SCRATCH / "xml2abc_verify.json").read_text())
exact = {(v["batch"], v["model"], v["prompt"], v.get("sample", 0))
         for v in verify if v["status"] == "ok"
         and v["recall"] == 1.0 and v["precision"] == 1.0}

targets = []  # (batch_dir, piece_dict)
for manifest in sorted(ROOT.glob("docs/data/2026*/data.json")):
    batch = manifest.parent
    for p in json.loads(manifest.read_text(encoding="utf-8")).get("pieces", []):
        if (p.get("ok") and p.get("mode") in ("codegen", "codegen-sparse")
                and p.get("score")
                and (batch.name, p["model"], p["prompt"], p.get("sample", 0)) in exact):
            targets.append((batch, p))
print(f"{len(targets)} event-exact pieces", flush=True)


def convert(xml_path: Path) -> str:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([PY, str(XML2ABC), "-m", "2", "-d", "8", "-n", "999999",
                        "-o", td, str(xml_path)],
                       capture_output=True, text=True, timeout=120)
        produced = list(Path(td).glob("*.abc"))
        abc = produced[0].read_text(encoding="utf-8") if produced else ""
    return re.sub(r"(^V:[^\n]*?) transpose=-?\d+", r"\1", abc, flags=re.M)


# --- build jobs ---------------------------------------------------------------
ckpt_path = SCRATCH / "judge_format_ab_ckpt.json"
attempted = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
clients = {j: get_client(j) for j in JUDGES}
lock = threading.Lock()

jobs = []
for batch, p in targets:
    key_base = f"{batch.name}|{p['model']}|{p['prompt']}|{p.get('sample', 0)}"
    abc = None
    for rep in ("listing", "abc"):
        for j in JUDGES:
            if j == p["model"]:
                continue  # exclude_self, standard protocol
            key = f"{key_base}|{rep}|{j}"
            if key in attempted:
                continue
            if rep == "abc" and abc is None:
                abc = convert(batch / p["score"])
            if rep == "listing":
                piece = {"model": p["model"], "prompt": p["prompt"],
                         "sample": p.get("sample", 0), "ok": True,
                         "score": p["score"]}
            else:
                if not abc:
                    attempted[key] = None
                    continue
                piece = {"model": p["model"], "prompt": p["prompt"],
                         "sample": p.get("sample", 0), "ok": True, "abc": abc}
            jobs.append((key, j, piece, batch))

n_cached = len(attempted)
print(f"{len(jobs)} judge calls to do ({n_cached} cached)", flush=True)


def work(job):
    key, jname, piece, batch = job
    return key, judge_piece(clients[jname], piece, batch, include_note=False)


def save():
    tmp = ckpt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(attempted))
    tmp.replace(ckpt_path)


done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for fut in as_completed([ex.submit(work, jb) for jb in jobs]):
        key, verdict = fut.result()
        with lock:
            attempted[key] = verdict
            done += 1
            if done % 25 == 0 or done == len(jobs):
                save()
                print(f"  [{done}/{len(jobs)}]", flush=True)
save()

# --- aggregate ----------------------------------------------------------------
from collections import defaultdict
QUALITY = ["coherence", "harmony", "rhythm", "structure", "melody",
           "emotion", "creativity", "naturalness"]
panel = defaultdict(dict)  # (piece_key, rep) -> {judge: verdict}
for key, v in attempted.items():
    if not v:
        continue
    b, m, pr, s, rep, j = key.split("|")
    panel[(f"{b}|{m}|{pr}|{s}", rep)][j] = v

rows = []
for (pk, rep), verdicts in panel.items():
    row = {"piece": pk, "rep": rep, "n_judges": len(verdicts)}
    for k in QUALITY + ["valence", "arousal"]:
        scores = [v[k]["score"] for v in verdicts.values() if k in v]
        row[k] = sum(scores) / len(scores) if scores else None
    q = [row[k] for k in QUALITY if row[k] is not None]
    row["overall"] = sum(q) / len(q) if q else None
    rows.append(row)
(SCRATCH / "judge_format_ab.json").write_text(json.dumps(rows, indent=1))

# paired comparison
by_piece = defaultdict(dict)
for r in rows:
    by_piece[r["piece"]][r["rep"]] = r
paired = [(d["listing"], d["abc"]) for d in by_piece.values()
          if "listing" in d and "abc" in d
          and d["listing"]["overall"] is not None and d["abc"]["overall"] is not None]
print(f"\n=== paired pieces: {len(paired)} ===")
import statistics as st
for k in QUALITY + ["overall", "valence", "arousal"]:
    ds = [a[k] - b[k] for a, b in paired if a[k] is not None and b[k] is not None]
    if not ds:
        continue
    mean = st.mean(ds)
    sd = st.stdev(ds) if len(ds) > 1 else 0.0
    se = sd / (len(ds) ** 0.5) if ds else 0.0
    print(f"  {k:12} listing-minus-abc = {mean:+.3f}  (SE {se:.3f}, n={len(ds)})")
