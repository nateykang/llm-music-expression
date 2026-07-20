#!/usr/bin/env python3
"""Per-model report on a sparse-codegen batch: successes /5, attempts, error
types, and two kinds of comment behavior in the saved code — deepseek-style
idea-iteration comments (thinking out loud / self-correction) and fable-style
expressive comments (narrative or poetic language).

    python scripts/analyze_sparse_batch.py [batch_dir]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

ERROR_CLASSES = [
    ("api-hallucination", r"AttributeError: (module 'music21|'[A-Za-z]+' object)"),
    ("music21 type misuse", r"TypeError|music21\..*Exception|StreamException"),
    ("python syntax", r"SyntaxError"),
    ("json escaping", r"line continuation character|could not parse JSON"),
    ("degenerate score", r"Degenerate score"),
    ("export failure", r"MusicXMLExport|midi"),
    ("api error", r"API error"),
]

ITER_RE = re.compile(
    r"#\s*(actually|wait\b|hmm|adjust(?:ing)?|instead|let'?s (?:try|use|make|adjust)|"
    r"we'?ll (?:modify|use|adjust|replace)|oops|correction|scratch that|"
    r"on second thought|better:|no[,:] |re-?do|fix(?:ing)? the|simplif)",
    re.I)

EXPRESSIVE_WORDS = re.compile(
    r"\b(warmth|stillness|breath|breathes?|glow|shimmer|dusk|dawn|twilight|"
    r"tide|drift(?:ing)?|dream|memory|memories|longing|yearn|ache|tender|"
    r"gentle|whisper|echoes?|fades?|dissolv\w+|settles?|wander|distant|"
    r"horizon|solitude|melanchol\w+|grief|sorrow|luminous|starlight|ember|"
    r"hearth|rain|snowfall|ocean|river|moonlight|hush|weightless|unfolds?|"
    r"blooms?|exhale|inhale|heartbeat|farewell|homecoming|wistful|serene|"
    r"radiant|fragile|eternity|fleeting|question(?:ing)? the (?:dark|silence|light))\b",
    re.I)

CODEY = re.compile(r"[=(){}\[\]]|import |return |def |quarterLength|noqa|type: ignore")


def comments_of(code: str):
    out = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            out.append(("block", s.lstrip("# ").strip()))
        elif "#" in ln and not re.search(r"['\"].*#.*['\"]", ln):
            out.append(("inline", ln.split("#", 1)[1].strip()))
    return [(k, t) for k, t in out if t]


def classify_error(err: str) -> str:
    tail = err.strip().splitlines()[-1] if err.strip() else ""
    for name, pat in ERROR_CLASSES:
        if re.search(pat, err):
            return name
    return f"other ({tail[:40]})"


def main():
    batch = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        (ROOT / "docs/data").glob("*/"), key=lambda p: p.stat().st_mtime)[-1]
    d = json.loads((batch / "data.json").read_text(encoding="utf-8"))
    ps = d["pieces"]
    print(f"batch: {batch.name}  ({len(ps)} pieces)\n")

    by = defaultdict(list)
    for p in ps:
        by[p["model"]].append(p)

    print(f"{'model':<22}{'ok':>5}{'attempts':>16}{'iter-cmts':>10}{'expr-cmts':>10}")
    examples_iter, examples_expr = defaultdict(list), defaultdict(list)
    err_types = defaultdict(lambda: defaultdict(int))
    for m in sorted(by):
        rows = by[m]
        ok = sum(1 for p in rows if p.get("ok"))
        atts = [p.get("attempts", 0) for p in rows]
        n_iter = n_expr = 0
        for p in rows:
            # final code + every failed draft (drafts often carry the
            # thinking-out-loud comments that get cleaned up on retry)
            sources = [("final", p.get("code") or "")]
            sources += [("draft", fa.get("code") or "")
                        for fa in (p.get("failed_attempts") or [])]
            for tag, code in sources:
                for kind, text in comments_of(code):
                    if ITER_RE.search("# " + text):
                        n_iter += 1
                        if len(examples_iter[m]) < 4:
                            examples_iter[m].append(f"[{tag}] {text[:100]}")
                    if EXPRESSIVE_WORDS.search(text) and not CODEY.search(text):
                        n_expr += 1
                        if len(examples_expr[m]) < 4:
                            examples_expr[m].append(f"[{tag}] {text[:100]}")
            errs = ([p.get("error")] if p.get("error") else []) \
                + [fa["error"] for fa in (p.get("failed_attempts") or []) if fa.get("error")] \
                + list(p.get("errors") or [])
            for err in dict.fromkeys(errs):  # dedupe final-vs-trail duplicates
                err_types[m][classify_error(err)] += 1
        print(f"{m:<22}{f'{ok}/{len(rows)}':>5}{str(atts):>16}{n_iter:>10}{n_expr:>10}")

    print("\n== error types (failed pieces' final errors + any recorded retries) ==")
    for m in sorted(err_types):
        parts = ", ".join(f"{k}×{v}" for k, v in sorted(err_types[m].items(), key=lambda t: -t[1]))
        print(f"  {m:<22}{parts}")

    print("\n== deepseek-style iteration comments (thinking out loud) ==")
    for m in sorted(examples_iter):
        print(f"  {m}:")
        for e in examples_iter[m]:
            print(f"    # {e}")

    print("\n== fable-style expressive comments ==")
    for m in sorted(examples_expr):
        print(f"  {m}:")
        for e in examples_expr[m]:
            print(f"    # {e}")


if __name__ == "__main__":
    main()
