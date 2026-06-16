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
from .store import write_results


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _split(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def _run_matrix(models: list[str], prompts: list[str], mode: str, max_attempts: int):
    results = []
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
        ts = _timestamp()
        batch = write_results(results, ts, models, prompts)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-music", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=list(MODES), default="codegen")
    common.add_argument("--max-attempts", type=int, default=5)

    pr = sub.add_parser("run", parents=[common], help="generate one model × prompt")
    pr.add_argument("--model", required=True)
    pr.add_argument("--prompt", default="freeform")
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("batch", parents=[common], help="generate a model × prompt matrix")
    pb.add_argument("--models", required=True, help="comma-separated friendly ids")
    pb.add_argument("--prompts", default="freeform", help="comma-separated prompt names")
    pb.set_defaults(func=cmd_batch)

    pm = sub.add_parser("models", help="list registered models")
    pm.set_defaults(func=cmd_models)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
