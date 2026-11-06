"""Write a run's restorable state into a backup directory: small files as they
are, audio-free corpus metadata, and verdicts gzipped per judge so a push only
carries the judges that changed. Part 2 verdicts live elsewhere and are skipped."""
import gzip
import json
import shutil
import sys
from pathlib import Path

root, run, dest = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
src = root / run
dest.mkdir(parents=True, exist_ok=True)


def gz_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, sort_keys=True).encode()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(data)
    if not path.exists() or path.read_bytes() != tmp.read_bytes():
        tmp.replace(path)
    else:
        tmp.unlink()


for f in src.iterdir():
    if f.is_file() and f.suffix in {".log", ".json", ".jsonl"}:
        if f.stat().st_size < 50_000_000:
            shutil.copy2(f, dest / f.name)
        else:
            gz_json(dest / f"{f.name}.gz", json.loads(f.read_text()))

state = src / "gen_state.json"
if state.exists():
    for v in json.loads(state.read_text())["out"].values():
        batch = root / "docs/data" / Path(v["batch"]).name
        if batch.exists():
            out = dest / "corpus" / batch.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(batch / "data.json", out / "data.json")
            if (batch / "scores").exists():
                shutil.copytree(batch / "scores", out / "scores", dirs_exist_ok=True)

ckpts = list((src / "analysis").glob("*_ckpt.json"))
if ckpts:
    ck = json.loads(ckpts[0].read_text())
    seen = root / "part2_run/analysis/part2_selfpref_raw.json"
    old = set()
    for row in json.loads(seen.read_text()):
        key = f"{row['batch']}|{row['model']}|{row['prompt']}|{row['mode']}|{row['sample']}"
        old.update(f"{key}|{j}" for j in row["panel"])
    shards = {}
    for k, v in ck.items():
        if k not in old:
            shards.setdefault(k.rsplit("|", 1)[1], {})[k] = v
    for j, d in shards.items():
        gz_json(dest / "verdicts" / f"{j}.json.gz", d)
    print(f"{len(ck)} verdicts in checkpoint, {sum(map(len, shards.values()))} new, "
          f"{len(shards)} judges backed up", flush=True)
