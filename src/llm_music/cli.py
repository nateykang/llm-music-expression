"""Command-line interface: `llm-music run` (single) and `llm-music batch` (matrix)."""

from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .generate import generate_piece
from .models import get_client, list_models
from .modes import MODES
from .store import append_result, open_batch, write_manifest


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _split(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def _run_matrix(models: list[str], prompts: list[str], mode: str, max_attempts: int,
                samples: int = 1, workers: int = 6, bake_audio: bool = True):
    # The batch folder + manifest are created up front and rewritten after every
    # piece, so an interrupted run still leaves a valid, viewable partial batch.
    ts = _timestamp()
    batch = open_batch(ts, models, prompts)
    print(f"  → writing to {batch}")

    # Generation is network-bound API calls, so we fan out across independent cells
    # with a thread pool. Clients are created once per model and shared (the SDKs are
    # thread-safe for concurrent requests). Cells are ordered sample-major so the
    # first `len(models)` in flight hit *different* providers — spreads rate limits.
    clients = {m: get_client(m) for m in models}
    cells = [(m, p, s) for s in range(samples) for p in prompts for m in models]
    total = len(cells)
    results, entries = [], []
    lock = threading.Lock()

    with tempfile.TemporaryDirectory(prefix="llm_music_batch_") as scratch:
        def work_cell(cell):
            m, p, s = cell
            wd = Path(scratch) / m / p / str(s)
            return cell, generate_piece(clients[m], p, mode, wd,
                                        max_attempts=max_attempts, bake_audio=bake_audio)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for fut in as_completed([ex.submit(work_cell, c) for c in cells]):
                (m, p, s), r = fut.result()
                with lock:
                    results.append(r)
                    entries.append(append_result(batch, r, sample=s))
                    write_manifest(batch, ts, models, prompts, entries)
                    n = len(results)
                tag = f" #{s + 1}" if samples > 1 else ""
                info = (f"ok ({r.attempts} att): {r.title!r}" if r.ok
                        else f"FAILED after {r.attempts}: {r.error}")
                print(f"  [{n}/{total}] {m} × {p}{tag} … {info}", flush=True)
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
    n_cells = len(models) * len(prompts) * args.samples
    print(f"Batch: {len(models)} model(s) × {len(prompts)} prompt(s) × {args.samples} "
          f"sample(s) = {n_cells} [{args.mode}], {args.workers} workers")
    batch, results = _run_matrix(models, prompts, args.mode, args.max_attempts,
                                 args.samples, args.workers, bake_audio=not args.no_audio)
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
        modes = [r["key_mode_best"] for r in rs if r.get("key_mode_best")]
        minor = (sum(m == "minor" for m in modes) / len(modes)) if modes else 0
        matches = [r["mode_match"] for r in rs if r.get("mode_match") not in (None, "")]
        match = (sum(int(x) for x in matches) / len(matches)) if matches else None
        keys = Counter(f"{r['key_declared_tonic'] or r['key_tonic']} {m}"
                       for r, m in zip(rs, (r.get("key_mode_best") or "?" for r in rs))).most_common(2)
        scales = [r["scale_consistency"] for r in rs if r["scale_consistency"] is not None]
        print(
            f"  {model:16} n={len(rs):2d}  minor={minor:.0%}  "
            f"mode_match={'—' if match is None else f'{match:.0%}'}  "
            f"valence={mean(r['valence'] for r in rs):+.2f}  "
            f"tempo={mean(r['tempo_bpm'] for r in rs):3.0f}  "
            f"scale_consist={(sum(scales)/len(scales) if scales else 0):.2f}  "
            f"top_keys={keys}"
        )
    return 0


def cmd_report(args) -> int:
    from .report import (key_distributions, load_features, load_reliability,
                         make_charts, make_key_chart, render_html)

    data_dir = Path(args.data_dir)
    rows = load_features(data_dir)
    if not rows:
        print(f"No features.csv found under {data_dir}. Run `llm-music analyze <batch>` first.")
        return 1
    analysis = data_dir.parent / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    charts = make_charts(rows, analysis)
    dists = key_distributions(rows)
    charts.append(make_key_chart(dists["all"], analysis))
    reliability = load_reliability(data_dir)

    # Bach-chorale reference (human functional-harmony baseline) — cached, since
    # computing the metric panel on the chorales is slow.
    import json as _json

    from .analyze import bach_reference
    bach_cache = analysis / "bach_reference.json"
    if bach_cache.exists():
        bach_rows = _json.loads(bach_cache.read_text(encoding="utf-8"))
    else:
        print("computing Bach-chorale reference (first run, ~1-2 min)…")
        bach_rows = bach_reference()
        bach_cache.write_text(_json.dumps(bach_rows), encoding="utf-8")

    out = data_dir.parent / "results.html"
    render_html(rows, charts, out, reliability, dists, bach_rows)
    print(f"Wrote dashboard → {out}  ({len(rows)} pieces, {len(charts)} charts)")
    return 0


def cmd_judge(args) -> int:
    from .judge import judge_corpus

    judges = _split(args.judges)
    if not judges:
        print("error: --judges must be non-empty", file=sys.stderr)
        return 2
    judge_corpus(Path(args.data_dir), judges, prompt=args.prompt or None,
                 limit=args.limit, workers=args.workers,
                 exclude_self=not args.no_exclude_self)
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
    pb.add_argument("--samples", type=int, default=1,
                    help="repeats per model×prompt cell (for sampling distributions)")
    pb.add_argument("--workers", type=int, default=6,
                    help="concurrent generations (network-bound; raise to go faster)")
    pb.add_argument("--no-audio", action="store_true",
                    help="skip audio baking (for large sampling runs — keeps the site lean)")
    pb.set_defaults(func=cmd_batch)

    pm = sub.add_parser("models", help="list registered models")
    pm.set_defaults(func=cmd_models)

    pa = sub.add_parser("analyze", help="extract standard metrics from a batch → features.csv")
    pa.add_argument("batch", help="path to a docs/data/<batch> folder")
    pa.add_argument("--summary-prompt", default="free-form",
                    help="prompt to base the per-model bias readout on")
    pa.set_defaults(func=cmd_analyze)

    pj = sub.add_parser("judge", help="run the LLM-judge panel over the corpus → judge.csv")
    pj.add_argument("--judges", default="gpt-5.5,gemini-2.5-pro,opus-4.8",
                    help="comma-separated panelist model ids (frontier; diverse)")
    pj.add_argument("--prompt", default="free-form",
                    help="restrict to one prompt (default free-form; '' for all)")
    pj.add_argument("--limit", type=int, default=None, help="cap number of pieces (for a pilot)")
    pj.add_argument("--workers", type=int, default=6, help="concurrent judge calls")
    pj.add_argument("--no-exclude-self", action="store_true",
                    help="let a model judge its own pieces (default: exclude, to defuse self-bias)")
    pj.add_argument("--data-dir", default="docs/data")
    pj.set_defaults(func=cmd_judge)

    prp = sub.add_parser("report", help="build the analysis dashboard (results.html + charts)")
    prp.add_argument("--data-dir", default="docs/data",
                     help="folder holding the batch subfolders (default: docs/data)")
    prp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
