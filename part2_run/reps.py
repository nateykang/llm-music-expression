"""Serve precomputed piece representations (identical bytes to what the batch
legs used) to the realtime judge path — throughput only; the judge input is
unchanged. Falls through to the live renderer on a cache miss."""
import json
from pathlib import Path

import llm_music.judge as _J


def wire_reps_cache(root: Path) -> int:
    reps = json.loads((root / "part2_run/reps_cache.json").read_text())
    orig = _J.representation

    def cached(piece, batch_dir):
        key = (f"{batch_dir.name}|{piece['model']}|{piece['prompt']}|"
               f"{piece.get('mode', '')}|{piece.get('sample', 0)}")
        hit = reps.get(key)
        if hit and hit[1] is not None:
            return hit[0], hit[1]
        return orig(piece, batch_dir)

    _J.representation = cached
    print(f"reps cache wired into judge path ({len(reps)} entries)", flush=True)
    return len(reps)
