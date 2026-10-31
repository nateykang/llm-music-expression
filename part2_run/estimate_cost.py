"""Cost estimate for a Part 2 extension: generation of the new arms' pieces plus
the judging cells the extension adds (old panel x new pieces, new arms x every
express-yourself piece). Per-judge table, so expensive judges are visible
before deciding what to drop (PART2_SKIP_JUDGES).

Inputs, all local except prices:
  * judge prompt sizes are rebuilt from the corpus with the judge path's own
    representation() + build_user() (cached in part2_run/reps_cache.json if present);
  * the checkpoint, if present, removes cells already banked;
  * prices: Anthropic list prices below; everything else from OpenRouter's live
    /models catalog (native-OpenAI arms priced as their openai/<id> slug).
    Batch-capable legs get the 50% batch discount.

Output tokens are the big unknown. Defaults are assumptions (--out-base,
--out-think); `--measure K` replaces them with K live judge calls per judge
(probe_arms.measure, the judge request byte-for-byte) banked in
part2_run/cost_sample.json and reused on later runs.

    python part2_run/estimate_cost.py                   # assumptions only
    python part2_run/estimate_cost.py --measure 2       # + ~2 real calls per judge
"""
import argparse
import json
import os
import statistics as st
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)
from config import BASE_BATCHES, BATCHES, SKIP_JUDGES  # noqa: E402
from llm_music.judge import _system, build_user, representation  # noqa: E402
from llm_music.models.registry import MODEL_REGISTRY  # noqa: E402
from probe_arms import NEW_ARMS  # noqa: E402

# $ per 1M tokens (input, output), first-party list prices.
ANTHROPIC_PRICES = {
    "claude-fable-5-1": (10.0, 50.0), "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-5-20251101": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0), "claude-sonnet-4-5-20250929": (3.0, 15.0),
}
# Proxy arms whose stored pieces stand in for the not-yet-generated new arms.
GEN_PROXY = {"fable-5.1": "fable-5", "opus-5.5": "opus-4.8", "gpt-6-astra": "gpt-5.6"}
CKPT = ROOT / "part2_run/analysis/part2_selfpref_ckpt.json"
SAMPLE = ROOT / "part2_run/cost_sample.json"
REPS_CACHE = ROOT / "part2_run/reps_cache.json"


def piece_key(b, pc):
    return f"{b}|{pc['model']}|{pc['prompt']}|{pc.get('mode', '')}|{pc.get('sample', 0)}"


def load_pieces(batches):
    out = []
    for b in batches:
        f = ROOT / "docs/data" / b / "data.json"
        if not f.exists():
            continue
        for pc in json.loads(f.read_text())["pieces"]:
            if pc.get("ok") and pc["prompt"] == "express-yourself":
                out.append((b, pc))
    return out


def prompt_chars(pieces):
    cache = json.loads(REPS_CACHE.read_text()) if REPS_CACHE.exists() else {}
    sys_len, out = len(_system(False)), {}
    for b, pc in pieces:
        pk = piece_key(b, pc)
        kind, text = cache.get(pk) or representation(pc, ROOT / "docs/data" / b)
        if text is not None:
            out[pk] = sys_len + len(build_user(pc, kind, text))
    return out


def catalog():
    try:
        import httpx
        data = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    except Exception as e:  # noqa: BLE001
        print(f"(OpenRouter catalog unavailable: {e}; non-Anthropic prices unknown)")
        return {}
    return {m["id"]: m for m in data}


def price(arm, cat):
    """(in $/M, out $/M, batch discount factor, route) or None."""
    provider, model_id = MODEL_REGISTRY[arm][:2]
    if provider == "anthropic":
        p = ANTHROPIC_PRICES.get(model_id)
        return (*p, 0.5, "anthropic batch") if p else None
    slug = model_id if provider == "openrouter" else f"openai/{model_id}"
    m = cat.get(slug)
    if not m:
        return None
    pi, po = float(m["pricing"]["prompt"]) * 1e6, float(m["pricing"]["completion"]) * 1e6
    batch = f"{slug}:batch" in cat
    return pi, po, 0.5 if batch else 1.0, "openrouter batch" if batch else "realtime"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-arms", default=",".join(NEW_ARMS))
    ap.add_argument("--planned-new-pieces", type=int, default=40,
                    help="per new arm, used while its batch isn't generated yet")
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument("--out-base", type=int, default=900, help="assumed output tokens, base arm")
    ap.add_argument("--out-think", type=int, default=4000, help="assumed output tokens, thinking arm")
    ap.add_argument("--gen-reason-base", type=int, default=2000)
    ap.add_argument("--gen-reason-think", type=int, default=10000)
    ap.add_argument("--measure", type=int, default=0, help="K live judge calls per judge")
    a = ap.parse_args()
    cpt = a.chars_per_token
    new_arms = [x for x in a.new_arms.split(",") if x]

    old = load_pieces(BASE_BATCHES)
    extra = load_pieces([b for b in BATCHES if b not in BASE_BATCHES])
    sizes = prompt_chars(old + extra)
    mode_mean = {m: st.mean(v for (b, pc) in old if pc["mode"] == m
                            for v in [sizes.get(piece_key(b, pc))] if v)
                 for m in ("abc", "codegen")}
    old_panel = sorted({pc["model"] for _, pc in old})
    judges = [j for j in sorted(set(old_panel) | set(new_arms)) if j not in SKIP_JUDGES]
    generated = {pc["model"] for _, pc in extra}
    # planned pieces for new arms with no batch on disk yet: half ABC, half codegen
    planned = [(arm, m) for arm in new_arms if arm not in generated
               for m in ("abc", "codegen") for _ in range(a.planned_new_pieces // 2)]
    ck = json.loads(CKPT.read_text()) if CKPT.exists() else None
    print(f"pieces: {len(old)} base + {len(extra)} extension on disk + {len(planned)} planned; "
          f"judges: {len(judges)}; checkpoint: {'yes' if ck is not None else 'NO (old x old assumed done)'}")

    # measured output tokens per judge
    sample = json.loads(SAMPLE.read_text()) if SAMPLE.exists() else {}
    if a.measure:
        from probe_arms import measure
        probe = [x for x in old if x[1]["model"] == "opus-4.8" and x[1]["sample"] < a.measure]
        for j in judges:
            have = sample.setdefault(j, [])
            for b, pc in probe[:max(0, a.measure * 2 - len(have))]:
                kind, text = representation(pc, ROOT / "docs/data" / b)
                try:
                    r = measure(j, _system(False), build_user(pc, kind, text))
                    have.append(r)
                    print(f"  measured {j}: in={r['input_tokens']} out={r['output_tokens']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  measure {j} failed: {e}", flush=True)
            SAMPLE.write_text(json.dumps(sample, indent=1))

    def out_tokens(j):
        vals = [r["output_tokens"] for r in sample.get(j, []) if r.get("output_tokens")]
        if vals:
            return st.mean(vals), "measured"
        return (a.out_think if j.endswith("-thinking") else a.out_base), "assumed"

    cat = catalog()
    cells = defaultdict(lambda: [0, 0.0])        # judge -> [n, input chars]
    for j in judges:
        for b, pc in old + extra:
            pk = piece_key(b, pc)
            if pk not in sizes:
                continue
            if ck is not None:
                if f"{pk}|{j}" in ck:
                    continue
            elif j in old_panel and b in BASE_BATCHES:
                continue
            cells[j][0] += 1
            cells[j][1] += sizes[pk]
        for arm, m in planned:
            cells[j][0] += 1
            cells[j][1] += mode_mean[m]

    print(f"\n{'judge':28} {'cells':>6} {'in Mtok':>8} {'out/cell':>9} {'route':17} {'$':>9}")
    total, unknown = 0.0, []
    rows = []
    for j in judges:
        n, chars = cells[j]
        if not n:
            continue
        intok = chars / cpt
        ot, src = out_tokens(j)
        p = price(j, cat)
        if p is None:
            unknown.append(j)
            rows.append((None, j, n, intok, ot, src, "price ?"))
            continue
        pi, po, disc, route = p
        cost = (intok * pi + n * ot * po) / 1e6 * disc
        total += cost
        rows.append((cost, j, n, intok, ot, src, route))
    for cost, j, n, intok, ot, src, route in sorted(rows, key=lambda r: -(r[0] or 0)):
        c = f"{cost:9.2f}" if cost is not None else "        ?"
        print(f"{j:28} {n:6d} {intok / 1e6:8.2f} {ot:7.0f}{src[0]:>2} {route:17} {c}")
    print(f"judging total: ${total:,.2f}" + (f"  (+ unpriced: {unknown})" if unknown else ""))

    # generation: new arms x 40 pieces, realtime CLI, proxy arms' piece sizes + attempts
    gen_total = 0.0
    print(f"\n{'generation arm':28} {'pieces':>6} {'attempts':>8} {'out tok/att':>11} {'$':>9}")
    allp = []
    for b in BASE_BATCHES:
        allp += json.loads((ROOT / "docs/data" / b / "data.json").read_text())["pieces"]
    for arm in new_arms:
        base = arm[:-9] if arm.endswith("-thinking") else arm
        proxy = GEN_PROXY.get(base, base) + arm[len(base):]
        ps = [p for p in allp if p["model"] == proxy and p["prompt"] == "express-yourself"]
        p = price(arm, cat)
        if not ps or p is None:
            print(f"{arm:28} (no proxy pieces or price)")
            continue
        pi, po = p[0], p[1]
        att = st.mean(p_.get("attempts") or 1 for p_ in ps)
        inc = st.mean(len(p_.get("system_prompt") or "") + len(p_.get("prompt_text") or "") for p_ in ps)
        outc = st.mean(len(p_.get("abc") or p_.get("code") or "") + len(p_.get("long_description") or "")
                       + len(p_.get("short_description") or "") for p_ in ps)
        reason = a.gen_reason_think if arm.endswith("-thinking") else a.gen_reason_base
        ot = outc / cpt + reason
        n = a.planned_new_pieces
        cost = n * att * (inc / cpt * pi + ot * po) / 1e6
        gen_total += cost
        print(f"{arm:28} {n:6d} {att:8.2f} {ot:11.0f} {cost:9.2f}")
    print(f"generation total: ${gen_total:,.2f}")
    print(f"\nGRAND TOTAL ≈ ${total + gen_total:,.2f}  (output tokens marked 'a' are assumptions; "
          f"run --measure for real numbers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
