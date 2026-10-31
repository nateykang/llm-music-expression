"""Live probe: one real judge call per arm, with token usage.

Run before any full extension run. For each arm it judges one fixed corpus
piece with exactly the request the judge path sends (same system prompt, user
message, model id, max tokens, thinking/effort/reasoning config) and reports:

  * whether the model id resolves and a verdict parses,
  * input / output / reasoning tokens (and OpenRouter's billed cost if returned),
  * for +/- pairs, whether the base arm really reasons less than the thinking arm.

`measure()` is also what estimate_cost.py --measure uses, so the cost sample and
the probe come from the same code path.

    python part2_run/probe_arms.py fable-5.1 fable-5.1-thinking opus-5.5 ...
    python part2_run/probe_arms.py --new          # the six extension arms
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from llm_music.judge import _extract_json, _system, build_user, parse_verdict, representation  # noqa: E402
from llm_music.models import get_client  # noqa: E402
from llm_music.models.registry import MODEL_REGISTRY  # noqa: E402

NEW_ARMS = ["fable-5.1", "fable-5.1-thinking", "opus-5.5", "opus-5.5-thinking",
            "gpt-6-astra", "gpt-6-astra-thinking", "gemini-4-argon", "gemini-4-argon-thinking"]
PROBE_BATCH = "20260819_201530__models_40_prompts_3"


def probe_piece(batch: str = PROBE_BATCH, model: str = "opus-4.8", sample: int = 0):
    man = json.loads((ROOT / "docs/data" / batch / "data.json").read_text())
    for pc in man["pieces"]:
        if (pc.get("ok") and pc["model"] == model and pc["prompt"] == "express-yourself"
                and pc.get("sample", 0) == sample):
            return pc, ROOT / "docs/data" / batch
    raise KeyError(f"no probe piece {batch}|{model}|{sample}")


def _get(obj, *path):
    for p in path:
        if obj is None:
            return None
        obj = obj.get(p) if isinstance(obj, dict) else getattr(obj, p, None)
    return obj


def measure(name: str, system: str, user: str) -> dict:
    """One judge call through the arm's provider with usage captured."""
    provider = MODEL_REGISTRY[name][0]
    jc = get_client(name)
    t0 = time.time()
    out = {"arm": name, "provider": provider, "model_id": jc.model_id}
    if provider == "anthropic":
        client = jc._ensure_client()
        kw = dict(model=jc.model_id, max_tokens=jc.max_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
        if jc.effort:
            kw["extra_body"] = {"output_config": {"effort": jc.effort}}
        if jc.thinking:
            th = dict(jc.thinking)
            if th.get("type") == "adaptive":
                th.setdefault("display", "summarized")
            kw["thinking"] = th
        with client.messages.stream(**kw) as s:
            msg = s.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        out.update(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                   reasoning_tokens=None, stop_reason=msg.stop_reason)
    elif provider == "openai":
        client = jc._ensure_client()
        kw = {}
        if jc.reasoning_effort:
            kw["reasoning"] = {"effort": jc.reasoning_effort}
        resp = client.responses.create(model=jc.model_id, instructions=system, input=user,
                                       max_output_tokens=jc.max_output_tokens, **kw)
        text = resp.output_text or ""
        u = resp.usage
        out.update(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                   reasoning_tokens=_get(u, "output_tokens_details", "reasoning_tokens"),
                   stop_reason=getattr(resp, "status", None))
    elif provider == "openrouter":
        client = jc._ensure_client()
        kw = dict(model=jc.model_id, max_tokens=jc.max_output_tokens,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}],
                  response_format={"type": "json_object"})
        extra = {"usage": {"include": True}}
        if jc.reasoning:
            extra["reasoning"] = jc.reasoning
        kw["extra_body"] = extra
        resp = client.chat.completions.create(**kw)
        ch = resp.choices[0] if resp.choices else None
        text = (ch.message.content if ch and ch.message else "") or ""
        u = resp.usage
        ud = u.model_dump() if hasattr(u, "model_dump") else dict(u or {})
        out.update(input_tokens=ud.get("prompt_tokens"), output_tokens=ud.get("completion_tokens"),
                   reasoning_tokens=_get(ud, "completion_tokens_details", "reasoning_tokens"),
                   billed_cost=ud.get("cost"), served_model=getattr(resp, "model", None),
                   stop_reason=ch.finish_reason if ch else None)
    else:
        raise ValueError(provider)
    obj = _extract_json(text)
    out["verdict_ok"] = bool(obj and parse_verdict(obj))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--new", action="store_true", help="probe the six extension arms")
    a = ap.parse_args()
    arms = (NEW_ARMS if a.new else []) + a.arms
    if not arms:
        ap.error("name arms or pass --new")
    pc, bdir = probe_piece()
    kind, text = representation(pc, bdir)
    system, user = _system(False), build_user(pc, kind, text)
    rows = []
    for arm in arms:
        try:
            r = measure(arm, system, user)
        except Exception as e:  # noqa: BLE001 — a probe reports, never crashes the sweep
            r = {"arm": arm, "error": f"{type(e).__name__}: {e}"[:400]}
        rows.append(r)
        print(json.dumps(r), flush=True)
    # +/- pair sanity: the base arm must reason (or emit) clearly less.
    by = {r["arm"]: r for r in rows}
    for arm in arms:
        if arm.endswith("-thinking") and arm[:-9] in by and "error" not in by[arm]:
            lo, hi = by[arm[:-9]], by[arm]
            if "error" in lo:
                continue
            k = "reasoning_tokens" if hi.get("reasoning_tokens") is not None else "output_tokens"
            print(f"pair {arm[:-9]}: {k} base={lo.get(k)} thinking={hi.get(k)}"
                  + ("  <-- CHECK: base not below thinking" if (lo.get(k) or 0) >= (hi.get(k) or 0) else ""))
    bad = [r["arm"] for r in rows if "error" in r or not r.get("verdict_ok")]
    print("ALL OK" if not bad else f"FAILED: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
