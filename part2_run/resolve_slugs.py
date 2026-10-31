"""Resolve the provider ids of newly released models against the live catalogs
and pin them in registry.py, before anything is generated.

For each family it lists every OpenRouter id (and native OpenAI id, for GPT)
whose name contains all the family's tokens. Exactly one OpenRouter text model
-> that slug is written into the registry (both arms of the pair). Zero or
several -> nothing is written and the script exits 1 with the candidates, so a
human picks. OpenRouter is preferred over native OpenAI because the judging
leg's 50% batch route goes through OpenRouter's :batch endpoints (the native
org's Batch API rejects file access), and one route per arm keeps generation
and judging on the same endpoint.

    python part2_run/resolve_slugs.py
"""
import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(os.environ.get("LLM_MUSIC_ROOT", Path(__file__).resolve().parents[1]))
REG = ROOT / "src/llm_music/models/registry.py"
FAMILIES = {  # registry base id -> tokens every candidate id must contain
    "gpt-6-astra": ("gpt-6", "astra"),
    "gemini-4-argon": ("gemini-4", "argon"),
}
SKIP = re.compile(r"(image|audio|tts|realtime|embed|search|transcribe|:free|:batch|:online)")


def main() -> int:
    cat = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    ids = {m["id"]: m for m in cat}
    native = set()
    if os.environ.get("OPENAI_API_KEY"):
        try:
            r = httpx.get("https://api.openai.com/v1/models", timeout=60,
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY'].strip()}"})
            native = {m["id"] for m in r.json().get("data", [])}
        except Exception as e:  # noqa: BLE001 — type only: messages can echo the auth header
            print(f"(native OpenAI model list unavailable: {type(e).__name__})")
    src = REG.read_text()
    bad = False
    for arm, toks in FAMILIES.items():
        cands = sorted(i for i in ids if all(t in i.lower() for t in toks) and not SKIP.search(i))
        nat = sorted(i for i in native if all(t in i.lower() for t in toks))
        print(f"{arm}: openrouter={cands} native_openai={nat or '-'}")
        for c in cands:
            m = ids[c]
            print(f"    {c}: ${float(m['pricing']['prompt']) * 1e6:.2f}/"
                  f"${float(m['pricing']['completion']) * 1e6:.2f} per M, "
                  f"batch={'yes' if c + ':batch' in ids else 'no'}, "
                  f"reasoning={'reasoning' in (m.get('supported_parameters') or [])}")
        cur = re.search(rf'"{re.escape(arm)}": \("openrouter", "([^"]+)"', src).group(1)
        if len(cands) > 1 and cur in cands:
            # The registry's own slug is an exact catalog id: keep it rather than
            # guess between it and suffixed variants (e.g. -pro).
            print(f"  -> {arm}: registry slug {cur} is in the catalog; keeping it over {[c for c in cands if c != cur]}")
            continue
        if len(cands) != 1:
            print(f"  !! {arm}: need exactly one OpenRouter candidate, got {len(cands)} — "
                  f"set the slug in registry.py by hand")
            bad = True
            continue
        if cur != cands[0]:
            src = src.replace(f'"openrouter", "{cur}"', f'"openrouter", "{cands[0]}"')
            print(f"  -> pinned {arm} (+ -thinking) to {cands[0]} (was {cur})")
        else:
            print(f"  -> {arm} slug confirmed: {cur}")
    REG.write_text(src)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
