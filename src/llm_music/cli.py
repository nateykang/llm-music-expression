"""Command-line interface: `llm-music run` (single) and `llm-music batch` (matrix)."""

from __future__ import annotations

import argparse
import json
import logging
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
                samples: int = 1, workers: int = 6, bake_audio: bool = True,
                independent_description: bool = False):
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
                                        max_attempts=max_attempts, bake_audio=bake_audio,
                                        independent_description=independent_description)

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
    batch, results = _run_matrix(models, prompts, args.mode, args.max_attempts,
                                 independent_description=args.independent_description)
    print(f"\nWrote batch: {batch}")
    return 0 if all(r.ok and not r.independent_description_error for r in results) else 1


def cmd_batch(args) -> int:
    models, prompts = _split(args.models), _split(args.prompts)
    if not models or not prompts:
        print("error: --models and --prompts must be non-empty", file=sys.stderr)
        return 2
    n_cells = len(models) * len(prompts) * args.samples
    print(f"Batch: {len(models)} model(s) × {len(prompts)} prompt(s) × {args.samples} "
          f"sample(s) = {n_cells} [{args.mode}], {args.workers} workers")
    batch, results = _run_matrix(models, prompts, args.mode, args.max_attempts,
                                 args.samples, args.workers, bake_audio=not args.no_audio,
                                 independent_description=args.independent_description)
    n_ok = sum(r.ok for r in results)
    n_description_failed = sum(bool(r.independent_description_error) for r in results)
    print(f"\nWrote batch: {batch}  ({n_ok}/{len(results)} succeeded)")
    if n_description_failed:
        print(f"warning: {n_description_failed} independent description(s) failed")
    return 0 if n_ok == len(results) and not n_description_failed else 1


def cmd_models(_args) -> int:
    print("Registered models:")
    for name in list_models():
        print(f"  {name}")
    return 0


def cmd_redescribe(args) -> int:
    """Apply the same music-only description call to an existing batch."""
    from .describe import describe_music, install_description, music_from_entry

    batch = Path(args.batch)
    manifest_path = batch / "data.json"
    if not manifest_path.is_file():
        print(f"error: no data.json under {batch}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = []
    for index, entry in enumerate(manifest.get("pieces", [])):
        if not entry.get("ok"):
            continue
        if args.author and entry.get("model") != args.author:
            continue
        if args.prompt and entry.get("prompt") != args.prompt:
            continue
        if args.sample is not None and entry.get("sample", 0) != args.sample:
            continue
        if entry.get("independent_description") and not args.force:
            continue
        selected.append((index, entry))
        if args.limit is not None and len(selected) >= args.limit:
            break
    if not selected:
        print("No matching undescribed pieces.")
        return 0

    clients = {}
    completed = failed = 0
    # Sequential processing plus a write after every response makes this safely
    # resumable without a separate checkpoint file.
    for index, entry in selected:
        model = args.model or entry["model"]
        try:
            music, representation = music_from_entry(entry, batch)
            if model not in clients:
                clients[model] = get_client(model)
            client = clients[model]
            result = describe_music(client, music, representation,
                                    max_attempts=args.max_attempts)
        except Exception as exc:
            result = None
            entry["independent_description_error"] = str(exc)
        if result and result.ok:
            install_description(entry, result, model, representation)
            completed += 1
            status = "ok"
        else:
            if result:
                entry["independent_description_error"] = result.error
            failed += 1
            status = f"FAILED: {entry['independent_description_error']}"
        checkpoint = manifest_path.with_suffix(".json.tmp")
        checkpoint.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        checkpoint.replace(manifest_path)
        print(f"  [{completed + failed}/{len(selected)}] piece {index} "
              f"{entry.get('model')} × {entry.get('prompt')} … {status}", flush=True)
    print(f"Updated {manifest_path}: {completed} described, {failed} failed")
    return 0 if not failed else 1


def cmd_analyze(args) -> int:
    from collections import Counter
    from statistics import mean

    from .analyze import analyze_batch, write_csv

    if args.all:
        import json as _json

        base = Path(args.data_dir)
        batches = sorted(p for p in base.iterdir() if p.is_dir() and (p / "data.json").exists())
        # experiment batches (e.g. the sparse-toolkit ablation) never enter the
        # corpus feature tables — they report in their own results.html section
        batches = [b for b in batches
                   if not _json.loads((b / "data.json").read_text(encoding="utf-8")).get("experiment")]
        if not batches:
            print(f"no batches under {base}")
            return 1
        total = 0
        for b in batches:
            rows = analyze_batch(b)
            if rows:
                write_csv(rows, b / "features.csv")
            total += len(rows)
            print(f"  {b.name}: {len(rows)} rows", flush=True)
        print(f"Wrote features for {total} pieces across {len(batches)} batches")
        return 0
    if not args.batch:
        print("error: give a batch path or use --all", file=sys.stderr)
        return 2
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

    from .analyze import FEATURES_VERSION

    data_dir = Path(args.data_dir)
    rows = load_features(data_dir)
    if not rows:
        print(f"No features.csv found under {data_dir}. Run `llm-music analyze <batch>` first.")
        return 1
    # Batches whose features.csv predates the current metric definitions would
    # silently blend old and new numbers — flag them instead.
    stale = sorted({r["_batch"] for r in rows
                    if str(r.get("features_version") or "") != str(FEATURES_VERSION)})
    if stale:
        print(f"warning: {len(stale)} batch(es) analyzed with an older feature version "
              f"(current v{FEATURES_VERSION}) — run `llm-music analyze --all`:")
        for b in stale:
            print(f"  {b}")
    unanalyzed = sorted(p.name for p in data_dir.iterdir()
                        if p.is_dir() and (p / "data.json").exists()
                        and not (p / "features.csv").exists())
    if unanalyzed:
        print(f"warning: {len(unanalyzed)} batch(es) have no features.csv (their pieces are "
              f"missing from the dashboard): {', '.join(unanalyzed)}")
    analysis = data_dir.parent / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    charts = make_charts(rows, analysis)
    dists = key_distributions(rows)
    charts.append(make_key_chart(dists["all"], analysis))
    reliability = load_reliability(data_dir)

    # Bach-chorale reference (human functional-harmony baseline) — cached, since
    # computing the metric panel on the chorales is slow. The cache is keyed to
    # FEATURES_VERSION so metric changes invalidate it instead of pinning the
    # reference row to whatever the code did when the cache was first written.
    import json as _json

    from .analyze import bach_reference
    bach_cache = analysis / "bach_reference.json"
    bach_rows = None
    if bach_cache.exists():
        cached = _json.loads(bach_cache.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("version") == FEATURES_VERSION:
            bach_rows = cached["rows"]
    if bach_rows is None:
        print(f"computing Bach-chorale reference (feature version v{FEATURES_VERSION}, ~1-2 min)…")
        bach_rows = bach_reference()
        bach_cache.write_text(_json.dumps({"version": FEATURES_VERSION, "rows": bach_rows}),
                              encoding="utf-8")

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
                 exclude_self=not args.no_exclude_self,
                 include_note=args.include_note, out_name=args.out_name)
    return 0


def cmd_judge_report(args) -> int:
    from .judge_report import render_judge_html

    data_dir = Path(args.data_dir)
    analysis = data_dir.parent / "analysis"
    out = render_judge_html(analysis, data_dir, data_dir.parent / "judge.html")
    print(f"Wrote judge page → {out}")
    return 0


def cmd_audio_report(args) -> int:
    from .audio_report import render_audio_html

    data_dir = Path(args.data_dir)
    analysis = data_dir.parent / "analysis"
    out = render_audio_html(analysis, data_dir, data_dir.parent / "audio.html")
    print(f"Wrote audio page → {out}")
    return 0


def cmd_embed_report(args) -> int:
    from .embed_report import render_selfpref_html

    data_dir = Path(args.data_dir)
    analysis = data_dir.parent / "analysis"
    out = render_selfpref_html(analysis, data_dir, data_dir.parent / "selfpref.html")
    print(f"Wrote self-preference page → {out}")
    return 0


def cmd_multimodal_report(_args) -> int:
    from .multimodal_report import write_multimodal_report

    out = write_multimodal_report()
    print(f"Wrote multimodal page → {out}")
    return 0


def cmd_genre_report(args) -> int:
    from .genre_report import render_genre_html

    data_dir = Path(args.data_dir)
    analysis = data_dir.parent / "analysis"
    out = render_genre_html(analysis, data_dir, data_dir.parent / "genre.html")
    print(f"Wrote genre page → {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-music", description=__doc__)
    p.add_argument("-q", "--quiet", action="store_true",
                   help="only log warnings and errors")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=list(MODES), default="codegen")
    common.add_argument("--max-attempts", type=int, default=5)
    common.add_argument("--independent-description", action="store_true",
                        help="replace the composing call's notes with a fresh, music-only "
                             "description from the same model (original notes are retained)")

    pr = sub.add_parser("run", parents=[common], help="generate one model × prompt")
    pr.add_argument("--model", required=True)
    pr.add_argument("--prompt", default="express-yourself")
    pr.set_defaults(func=cmd_run)

    pb = sub.add_parser("batch", parents=[common], help="generate a model × prompt matrix")
    pb.add_argument("--models", required=True, help="comma-separated friendly ids")
    pb.add_argument("--prompts", default="express-yourself",
                    help="comma-separated variant ids (see prompts/variants.csv)")
    pb.add_argument("--samples", type=int, default=1,
                    help="repeats per model×prompt cell (for sampling distributions)")
    pb.add_argument("--workers", type=int, default=6,
                    help="concurrent generations (network-bound; raise to go faster)")
    pb.add_argument("--no-audio", action="store_true",
                    help="skip audio baking (for large sampling runs — keeps the site lean)")
    pb.set_defaults(func=cmd_batch)

    pm = sub.add_parser("models", help="list registered models")
    pm.set_defaults(func=cmd_models)

    pd = sub.add_parser(
        "redescribe", help="generate fresh music-only descriptions for an existing batch"
    )
    pd.add_argument("batch", help="path to a docs/data/<batch> folder")
    pd.add_argument("--model", default=None,
                    help="describer model (default: each piece's author model)")
    pd.add_argument("--author", default=None, help="restrict to one author model")
    pd.add_argument("--prompt", default=None, help="restrict to one generation prompt")
    pd.add_argument("--sample", type=int, default=None, help="restrict to one sample index")
    pd.add_argument("--limit", type=int, default=None, help="cap pieces for a pilot")
    pd.add_argument("--max-attempts", type=int, default=3)
    pd.add_argument("--force", action="store_true",
                    help="regenerate descriptions that already have this treatment")
    pd.set_defaults(func=cmd_redescribe)

    pa = sub.add_parser("analyze", help="extract standard metrics from a batch → features.csv")
    pa.add_argument("batch", nargs="?", help="path to a docs/data/<batch> folder")
    pa.add_argument("--all", action="store_true",
                    help="re-analyze every batch under --data-dir (run after metric changes)")
    pa.add_argument("--data-dir", default="docs/data")
    pa.add_argument("--summary-prompt", default="free-form",
                    help="prompt to base the per-model bias readout on")
    pa.set_defaults(func=cmd_analyze)

    pj = sub.add_parser("judge", help="run the LLM-judge panel over the corpus → judge.csv")
    pj.add_argument("--judges", default="gpt-5.5,gemini-2.5-pro,opus-4.8",
                    help="comma-separated panelist model ids (frontier; diverse)")
    pj.add_argument("--prompt", default="",
                    help="restrict to one prompt id (default: judge all prompts)")
    pj.add_argument("--limit", type=int, default=None, help="cap number of pieces (for a pilot)")
    pj.add_argument("--workers", type=int, default=6, help="concurrent judge calls")
    pj.add_argument("--no-exclude-self", action="store_true",
                    help="let a model judge its own pieces (default: exclude, to defuse self-bias)")
    pj.add_argument("--include-note", action="store_true",
                    help="show the composer's note + add intent dimension (noted condition; "
                         "default is blind music-only — run both to test for text bias)")
    pj.add_argument("--out-name", default=None,
                    help="output basename under analysis/ (default: judge / judge_noted)")
    pj.add_argument("--data-dir", default="docs/data")
    pj.set_defaults(func=cmd_judge)

    pjr = sub.add_parser("judge-report", help="build the LLM-judge analysis page (judge.html)")
    pjr.add_argument("--data-dir", default="docs/data")
    pjr.set_defaults(func=cmd_judge_report)

    par = sub.add_parser("audio-report", help="build the audio-emotion analysis page (audio.html)")
    par.add_argument("--data-dir", default="docs/data")
    par.set_defaults(func=cmd_audio_report)

    pe = sub.add_parser("embed-report",
                        help="build the style-space & self-preference page (selfpref.html)")
    pe.add_argument("--data-dir", default="docs/data")
    pe.set_defaults(func=cmd_embed_report)

    pg = sub.add_parser("genre-report",
                        help="build the human-corpora genre-bias page (genre.html)")
    pg.add_argument("--data-dir", default="docs/data")
    pg.set_defaults(func=cmd_genre_report)

    pmm = sub.add_parser("multimodal-report",
                         help="build the description↔music page (multimodal.html)")
    pmm.set_defaults(func=cmd_multimodal_report)

    prp = sub.add_parser("report", help="build the analysis dashboard (results.html + charts)")
    prp.add_argument("--data-dir", default="docs/data",
                     help="folder holding the batch subfolders (default: docs/data)")
    prp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
