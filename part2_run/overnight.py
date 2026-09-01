"""Overnight OpenRouter batch pipeline: validate each arm on 50 pieces, run the
full batch only for arms that pass, quarantine anything that degrades at scale.
Sequential validations, pipelined full batches, single writer to the checkpoint.
"""
import json, os, re, sys, time, warnings, collections
from statistics import mean
from pathlib import Path

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", "/Users/nathanielkang/llm-music-expression"))
sys.path.insert(0, str(ROOT / "src")); warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")  # noqa: E402
import httpx  # noqa: E402
from llm_music.judge import (_system, build_user, _extract_json, parse_verdict,  # noqa: E402
                             EMOTION_LABELS)
from llm_music.models import get_client  # noqa: E402
from llm_music.models.registry import MODEL_REGISTRY  # noqa: E402

API = "https://openrouter.ai/api/beta/batches"
H = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
CKPT = ROOT / "part2_run/analysis/part2_selfpref_ckpt.json"
STATE = ROOT / "part2_run/overnight_state.json"
LOG = ROOT / "part2_run/overnight.log"
Q = ["coherence","harmony","rhythm","structure","melody","emotion","creativity","naturalness","valence","arousal","topline"]
Q8 = Q[:8]
ABC, TOP, CG = ("20260819_201530__models_40_prompts_3", "20260819_203512__models_2_prompts_3",
                "20260820_061907__models_42_prompts_3")
TERMINAL = {"completed", "failed", "expired", "cancelled"}
ORDER = ["gemini-3.7-flash","gemini-3.7-flash-thinking","gemini-3-flash","gemini-3-flash-thinking",
         "gemini-3.6-flash","gemini-3.6-flash-thinking","gemini-3.5-flash","gemini-3.5-flash-thinking",
         "gemini-3.1-pro","gemini-3.1-pro-thinking","kimi-k3","kimi-k3-thinking",
         "gpt-5.2","gpt-5.2-thinking","gpt-5.5","gpt-5.5-thinking","opus-4.1","opus-4.1-thinking"]
OR_OVERRIDES = {"gpt-5.5": ("openai/gpt-5.5", {"enabled": False}), "gpt-5.5-thinking": ("openai/gpt-5.5", {"effort": "high"}),
                "gpt-5.2": ("openai/gpt-5.2", {"enabled": False}), "gpt-5.2-thinking": ("openai/gpt-5.2", {"effort": "high"})}
CAP_BASE = int(os.environ.get("OR_CAP_BASE", "8000"))
CAP_THINK = int(os.environ.get("OR_CAP_THINK", "16000"))
VAL_TIMEOUT, TOTAL_HOURS = 45 * 60, 10 * 3600

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def arm_cfg(arm):
    if arm in OR_OVERRIDES: slug, reasoning = OR_OVERRIDES[arm]
    else:
        jc = get_client(arm); slug, reasoning = jc.model_id, jc.reasoning
    return slug, reasoning, (CAP_THINK if arm.endswith("-thinking") else CAP_BASE)

catalog = {m["id"]: m for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]}
def out_price(slug): return float(catalog[slug]["pricing"]["completion"]) * 1e6
def balance():
    c = httpx.get("https://openrouter.ai/api/v1/credits", headers=H, timeout=30).json()["data"]
    return c["total_credits"] - c["total_usage"]

reps = json.load(open(ROOT / "part2_run/reps_cache.json"))
def pk(b, pc): return f"{b}|{pc['model']}|{pc['prompt']}|{pc.get('mode','')}|{pc.get('sample',0)}"
tasks = []
for b in (ABC, TOP, CG):
    for pc in json.loads((ROOT / "docs/data" / b / "data.json").read_text())["pieces"]:
        if pc.get("ok") and pc.get("prompt") == "express-yourself" and reps.get(pk(b, pc), [None, None])[1]:
            tasks.append((b, pc))
val_pieces = [t for t in tasks if t[0] == ABC][:25] + [t for t in tasks if t[0] == CG][:25]
system = _system(False)

def load_ckpt(): return json.loads(CKPT.read_text())
def save_ckpt(ck):
    tmp = CKPT.with_suffix(".tmp"); tmp.write_text(json.dumps(ck)); tmp.replace(CKPT)

def body_for(slug, reasoning, pc, kind, text, max_tokens):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": build_user(pc, kind, text)}],
            "max_tokens": max_tokens}
    if not slug.startswith("google/"): body["response_format"] = {"type": "json_object"}
    if reasoning: body["reasoning"] = reasoning
    return body

def submit(arm, pieces_, ck):
    slug, reasoning, mt = arm_cfg(arm)
    reqs, idmap = [], {}
    for i, (b, pc) in enumerate(pieces_):
        key = f"{pk(b, pc)}|{arm}"
        if key in ck: continue
        kind, text = reps[pk(b, pc)]
        cid = f"q{i:06d}"; idmap[cid] = key
        reqs.append({"custom_id": cid, "body": body_for(slug, reasoning, pc, kind, text, mt)})
    if not reqs: return None, idmap, 0.0
    hold = len(reqs) * mt * out_price(slug) / 1e6 / 2
    r = httpx.post(API, headers=H, content=json.dumps({"endpoint": "/v1/chat/completions", "model": slug, "requests": reqs}), timeout=900)
    if r.status_code not in (200, 201, 202):
        return {"error": f"{r.status_code} {r.text[:160]}"}, idmap, hold
    return {"id": r.json()["id"]}, idmap, hold

def poll(bid):
    try: return httpx.get(f"{API}/{bid}", headers=H, timeout=600).json()
    except Exception as e: return {"status": "poll_error", "error": str(e)[:100]}

ANTH = [j for j in {k.rsplit("|", 1)[1] for k in load_ckpt()} if not j.startswith("gemini")]
panel_cache = {}
def panel_mean(pkey, ck):
    if pkey not in panel_cache:
        vals = [mean(ck[f"{pkey}|{a}"][d]["score"] for d in Q8) for a in ANTH if f"{pkey}|{a}" in ck]
        panel_cache[pkey] = mean(vals) if len(vals) >= 5 else None
    return panel_cache[pkey]
KW = {"harmony": r"harmon|chord|tonal|key|cadence|progression", "rhythm": r"rhythm|pulse|meter|tempo|beat|groove",
      "melody": r"melod|line|contour|tune|phrase", "structure": r"structur|form|repeat|repetition|section|develop"}
def corr(a, b):
    if len(a) < 10: return float("nan")
    ma, mb = mean(a), mean(b); n = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    da = (sum((x-ma)**2 for x in a)**.5) * (sum((y-mb)**2 for y in b)**.5)
    return n / da if da else float("nan")

def collect(d, idmap, ck):
    """results -> (verdicts dict, metrics dict)"""
    out, n_results, api_err, invalid = {}, 0, 0, 0
    labels, xs, ys, kw_hit, kw_tot, traces, reps_seen = collections.Counter(), [], [], 0, 0, 0, set()
    for res in d.get("results") or []:
        key = idmap.get(res.get("custom_id"))
        if not key: continue
        n_results += 1
        if res.get("error"): api_err += 1; continue
        body = (res.get("response") or {}).get("body") or {}
        msg = ((body.get("choices") or [{}])[0].get("message") or {})
        content = msg.get("content") or ""
        obj = _extract_json(content); v = parse_verdict(obj) if obj else None
        if not v: invalid += 1; continue
        if msg.get("reasoning"): v["reasoning_trace"] = msg["reasoning"][:50000]; traces += 1
        out[key] = v; labels[v.get("emotion_label")] += 1
        pkey = key.rsplit("|", 1)[0]; reps_seen.add("codegen" if pkey.startswith(CG) else "abc")
        pm = panel_mean(pkey, ck)
        if pm is not None: xs.append(mean(v[dd]["score"] for dd in Q8)); ys.append(pm)
        for dd, pat in KW.items(): kw_tot += 1; kw_hit += bool(re.search(pat, v[dd]["reason"], re.I))
    n_ok = len(out); tot = max(n_results, 1)
    m = {"n_results": n_results, "valid": n_ok, "valid_rate": n_ok / tot, "api_errors": api_err, "invalid": invalid,
         "label_in_vocab": (sum(c for l, c in labels.items() if l in EMOTION_LABELS) / n_ok) if n_ok else 0,
         "top_label_share": (labels.most_common(1)[0][1] / n_ok) if n_ok else 0,
         "top_label": labels.most_common(1)[0][0] if n_ok else None,
         "kw_align": (kw_hit / kw_tot) if kw_tot else 0, "r_vs_anth": corr(xs, ys), "n_r": len(xs),
         "mean_overall": mean(xs) if xs else None, "trace_rate": (traces / n_ok) if n_ok else 0, "reps": sorted(reps_seen)}
    return out, m

def verdict_check(m, full=False):
    reasons = []
    if m["valid_rate"] < (0.90 if full else 0.92): reasons.append(f"valid_rate {m['valid_rate']:.2f}")
    if m["label_in_vocab"] < 0.95: reasons.append(f"label_in_vocab {m['label_in_vocab']:.2f}")
    if m["top_label_share"] > 0.60: reasons.append(f"label collapse {m['top_label']} {m['top_label_share']:.2f}")
    if m["kw_align"] < 0.90: reasons.append(f"kw_align {m['kw_align']:.2f}")
    if not (m["r_vs_anth"] >= 0.35): reasons.append(f"r_vs_anth {m['r_vs_anth']:.2f}")
    if not full and set(m["reps"]) != {"abc", "codegen"}: reasons.append(f"reps {m['reps']}")
    return reasons

state = json.loads(STATE.read_text()) if STATE.exists() else {"arms": {}, "full_pending": {}}
def save_state(): STATE.write_text(json.dumps(state, indent=1))
def fmt(m): return (f"valid {m['valid']}/{m['n_results']} r={m['r_vs_anth']:.2f} mean={m['mean_overall'] and round(m['mean_overall'],2)} "
                    f"labels {m['top_label']}:{m['top_label_share']:.2f} kw={m['kw_align']:.2f} traces={m['trace_rate']:.2f} reps={m['reps']}")

def try_full_submit(arm):
    ck = load_ckpt(); slug, reasoning, mt = arm_cfg(arm)
    n_todo = sum(1 for b, pc in tasks if f"{pk(b, pc)}|{arm}" not in ck)
    hold = n_todo * mt * out_price(slug) / 1e6 / 2
    bal = balance()
    RESERVE = int(os.environ.get("OR_CREDIT_RESERVE", "150"))
    if bal < hold + RESERVE:
        state["arms"][arm]["full"] = "credit_blocked"; state["arms"][arm]["hold_needed"] = round(hold, 2)
        log(f"CREDIT-BLOCKED {arm}: full batch needs ~${hold:.0f} hold, balance ${bal:.0f} — will retry as holds release"); save_state(); return
    res, idmap, hold = submit(arm, tasks, ck)
    if res is None: state["arms"][arm]["full"] = "nothing_to_do"; save_state(); return
    if "error" in res:
        state["arms"][arm]["full"] = "submit_error"; state["arms"][arm]["full_error"] = res["error"]
        log(f"FULL SUBMIT ERROR {arm}: {res['error']}"); save_state(); return
    state["arms"][arm]["full"] = "submitted"; state["arms"][arm]["full_id"] = res["id"]
    state["full_pending"][res["id"]] = {"arm": arm, "idmap": idmap, "n": len(idmap)}
    log(f"FULL SUBMITTED {arm}: {len(idmap)} requests (hold ~${hold:.0f}) -> {res['id']}"); save_state()

def service_full_pending():
    """Poll every pending full batch; collect, check, merge or quarantine."""
    for bid in list(state["full_pending"]):
        info = state["full_pending"][bid]; arm = info["arm"]
        d = poll(bid); st = d.get("status")
        if st not in TERMINAL: continue
        ck = load_ckpt(); out, m = collect(d, info["idmap"], ck)
        reasons = verdict_check(m, full=True)
        if reasons:
            (ROOT / f"part2_run/quarantine_{arm}.json").write_text(json.dumps(out))
            state["arms"][arm]["full"] = "quarantined"; state["arms"][arm]["full_metrics"] = m; state["arms"][arm]["full_reasons"] = reasons
            log(f"FULL QUARANTINED {arm} ({st}): {fmt(m)} | reasons: {reasons}")
        else:
            ck.update(out); save_ckpt(ck)
            state["arms"][arm]["full"] = "merged"; state["arms"][arm]["full_metrics"] = m
            log(f"FULL MERGED {arm} ({st}): {fmt(m)} | cost ${(d.get('usage') or {}).get('cost')}")
        del state["full_pending"][bid]; save_state()
        for other, s in state["arms"].items():  # holds released -> retry credit-blocked arms
            if s.get("validation") == "pass" and s.get("full") == "credit_blocked": try_full_submit(other)

def main():
    t_start = time.time()
    log(f"OVERNIGHT START: {len(ORDER)} arms, {len(tasks)} pieces, balance ${balance():.2f}")
    for arm in ORDER:
        s = state["arms"].setdefault(arm, {})
        if s.get("validation") == "pass" and s.get("full") in (None, "credit_blocked", "submit_error"):
            log(f"retry full submit {arm} (previous: {s.get('full')})"); try_full_submit(arm); continue
        if s.get("validation") in ("pass", "fail", "timeout", "error"):
            log(f"skip {arm}: validation already {s['validation']}"); continue
        ck = load_ckpt()
        if s.get("validation_id") and s.get("validation_idmap"):
            res, idmap = {"id": s["validation_id"]}, s["validation_idmap"]
            log(f"VALIDATION REUSE {arm}: polling {res['id']}")
        else:
            res, idmap, _ = submit(arm, val_pieces, ck)
            if res is None: s["validation"] = "pass"; s["note"] = "validation pieces already banked"; save_state(); try_full_submit(arm); continue
            if "error" in res: s["validation"] = "error"; s["error"] = res["error"]; log(f"VALIDATION SUBMIT ERROR {arm}: {res['error']}"); save_state(); continue
            s["validation_id"] = res["id"]; s["validation_idmap"] = idmap; save_state(); log(f"VALIDATION SUBMITTED {arm}: {len(idmap)} pieces -> {res['id']}")
        t0 = time.time(); d = None
        while time.time() - t0 < VAL_TIMEOUT:
            time.sleep(30); service_full_pending()
            d = poll(res["id"])
            if d.get("status") in TERMINAL: break
        if not d or d.get("status") not in TERMINAL:
            s["validation"] = "timeout"; log(f"VALIDATION TIMEOUT {arm} (45 min) — skipped"); save_state(); continue
        ck = load_ckpt(); out, m = collect(d, idmap, ck); reasons = verdict_check(m)
        s["validation_metrics"] = m
        if reasons:
            s["validation"] = "fail"; s["reasons"] = reasons
            (ROOT / f"part2_run/validation_fail_{arm}.json").write_text(json.dumps(out))
            log(f"VALIDATION FAIL {arm} ({d.get('status')}): {fmt(m)} | reasons: {reasons} — full batch NOT run"); save_state(); continue
        ck.update(out); save_ckpt(ck); s["validation"] = "pass"
        log(f"VALIDATION PASS {arm}: {fmt(m)} | cost ${(d.get('usage') or {}).get('cost')} — 50 merged, submitting full batch")
        save_state(); try_full_submit(arm)

    log("all validations done; draining full batches")
    while state["full_pending"] and time.time() - t_start < TOTAL_HOURS:
        time.sleep(60); service_full_pending()
    if state["full_pending"]: log(f"TIME LIMIT: still pending {[(v['arm'], k) for k, v in state['full_pending'].items()]} (ids saved in state)")
    n = len(load_ckpt())
    log(f"OVERNIGHT DONE: checkpoint {n} verdicts | " + ", ".join(f"{a}: val={s.get('validation')} full={s.get('full')}" for a, s in state["arms"].items()))


if __name__ == '__main__':
    main()
