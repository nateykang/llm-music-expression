"""Command-line interface: `llm-music run` (single) and `llm-music batch` (matrix)."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from .generate import generate_piece
from .models import get_client, list_models
from .modes import MODES
from .store import append_result, open_batch, write_manifest


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _split(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def _run_matrix(models: list[str], prompts: list[str], mode: str, max_attempts: int):
    # The batch folder + manifest are created up front and rewritten after every
    # piece, so an interrupted run still leaves a valid, viewable partial batch.
    ts = _timestamp()
    batch = open_batch(ts, models, prompts)
    print(f"  → writing to {batch}")
    results, entries = [], []
    with tempfile.TemporaryDirectory(prefix="llm_music_batch_") as scratch:
        for m in models:
            client = get_client(m)
            for p in prompts:
                work = Path(scratch) / m / p
                print(f"  • {m} × {p} ({mode}) …", end="", flush=True)
                r = generate_piece(client, p, mode, work, max_attempts=max_attempts)
                if r.ok:
                    audio = "audio" if r.audio_path else "no-audio"
                    print(f" ok ({r.attempts} attempt(s), {audio}): {r.title!r}")
                else:
                    print(f" FAILED after {r.attempts}: {r.error}")
                results.append(r)
                entries.append(append_result(batch, r))
                write_manifest(batch, ts, models, prompts, entries)
    return batch, results


def cmd_run(args) -> int:
    models, prompts = [args.model], [args.prompt]
    batch, results = _run_matrix(models, prompts, args.mode, args.max_attempts)
    print(f"\nWrote batch: {batch}")
    return 0 if all(r.ok for r in results) else 1


def cmd_batch(args) -> int:
    models, prompts = _split(args.models), _split(args.prompts)
    if not models or not prompts:
        print("error: --models and --prompts must be non-empty", file=sys.stderr)
        return 2
    print(f"Batch: {len(models)} model(s) × {len(prompts)} prompt(s) [{args.mode}]")
    batch, results = _run_matrix(models, prompts, args.mode, args.max_attempts)
    n_ok = sum(r.ok for r in results)
    print(f"\nWrote batch: {batch}  ({n_ok}/{len(results)} succeeded)")
    return 0 if n_ok == len(results) else 1


def cmd_models(_args) -> int:
    print("Registered models:")
    for name in list_models():
        print(f"  {name}")
    return 0


def cmd_analyze(args) -> int:
    from collections import Counter
    from statistics import mean

    from .analyze import analyze_batch, write_csv

    batch = Path(args.batch)
    rows = analyze_batch(batch)
    if not rows:
        print(f"no analyzable pieces in {batch}")
        return 1
    out = batch / "features.csv"
    write_csv(rows, out)
    print(f"Wrote {len(rows)} rows → {out}\n")

    # Inductive-bias readout: per-model defaults (free-form is the purest probe).
    ff = [r for r in rows if r["prompt"] == args.summary_prompt] or rows
    scope = args.summary_prompt if any(r["prompt"] == args.summary_prompt for r in rows) else "all prompts"
    print(f"=== Per-model defaults ({scope}) ===")
    by_model: dict[str, list] = {}
    for r in ff:
        by_model.setdefault(r["model"], []).append(r)
    for model, rs in sorted(by_model.items()):
        minor = sum(r["key_mode"] == "minor" for r in rs) / len(rs)
        keys = Counter(f"{r['key_tonic']} {r['key_mode']}" for r in rs).most_common(2)
        print(
            f"  {model:14} n={len(rs):2d}  minor={minor:.0%}  "
            f"valence={mean(r['valence'] for r in rs):+.2f}  "
            f"arousal={mean(r['arousal'] for r in rs):.2f}  "
            f"tempo={mean(r['tempo_bpm'] for r in rs):3.0f}  "
            f"scale_consist={mean(r['scale_consistency'] for r in rs if r['scale_consistency'] is not None):.2f}  "
            f"top_keys={keys}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-music", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=list(MODES), default="codegen")
    common.add_argument("--max-attempts", type=int, default=5)

    pr = sub.add_parser("run", parents=[common], help="generate one model × prompt")
    pr.add_argument("--model", required=True)
    pr.add_argument("--prompt", default="free-form")
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("batch", parents=[common], help="generate a model × prompt matrix")
    pb.add_argument("--models", required=True, help="comma-separated friendly ids")
    pb.add_argument("--prompts", default="free-form", help="comma-separated prompt names")
    pb.set_defaults(func=cmd_batch)

    pm = sub.add_parser("models", help="list registered models")
    pm.set_defaults(func=cmd_models)

    pa = sub.add_parser("analyze", help="extract standard metrics from a batch → features.csv")
    pa.add_argument("batch", help="path to a docs/data/<batch> folder")
    pa.add_argument("--summary-prompt", default="free-form",
                    help="prompt to base the per-model bias readout on")
    pa.set_defaults(func=cmd_analyze)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
