"""Provider Batch APIs (50% off realtime) behind one interface.

Request bodies mirror each adapter's realtime `complete_full` call exactly, and
results come back as the same (text, meta) pair, so a batched response is
interchangeable with a live one. One batch = one model (OpenAI and OpenRouter
require it). Handles are plain dicts so callers can persist them and resume.

  handle = submit(client, {custom_id: (system, user, json_mode), ...})
  done, summary = poll(handle)
  results(handle) -> {custom_id: (text, meta)}   # successful requests only
"""

from __future__ import annotations

import io
import json
import time
import os

import httpx

_OR_API = "https://openrouter.ai/api/beta/batches"
_OR_TERMINAL = {"completed", "failed", "expired", "cancelled"}
_OPENAI_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _provider(client) -> str:
    return {"OpenAIClient": "openai", "AnthropicClient": "anthropic",
            "OpenRouterClient": "openrouter"}[type(client).__name__]


def openrouter_batchable() -> set[str]:
    """OpenRouter slugs that currently expose a :batch endpoint."""
    data = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    return {m["id"].removesuffix(":batch") for m in data if m["id"].endswith(":batch")}


def supports_batch(client, or_batchable: set[str] | None = None) -> bool:
    prov = _provider(client)
    if prov != "openrouter":
        return True
    return client.model_id in (or_batchable if or_batchable is not None
                               else openrouter_batchable())


def _openai_sdk():
    import openai
    return openai.OpenAI(timeout=600.0, max_retries=3)


def _anthropic_sdk():
    import anthropic
    return anthropic.Anthropic(timeout=600.0, max_retries=3)


def _or_headers():
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"}


def _openai_body(c, system, user):
    body = {"model": c.model_id, "instructions": system, "input": user,
            "max_output_tokens": c.max_output_tokens}
    if c.reasoning_effort:
        body["reasoning"] = {"effort": c.reasoning_effort}
        if c.reasoning_effort != "none":
            body["reasoning"]["summary"] = "auto"
    return body


def _anthropic_params(c, system, user):
    params = {"model": c.model_id, "max_tokens": c.max_tokens, "system": system,
              "messages": [{"role": "user", "content": user}]}
    if c.effort:
        params["output_config"] = {"effort": c.effort}
    if c.thinking:
        th = dict(c.thinking)
        if th.get("type") == "adaptive":
            th.setdefault("display", "summarized")
        params["thinking"] = th
    return params


def _openrouter_body(c, system, user, json_mode):
    body = {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": c.max_output_tokens}
    if json_mode and not c.model_id.startswith("google/"):
        # Google's batch backend derives ONE flat schema per batch from
        # json_object and flattens nested objects (diagnosed 2026-08-31);
        # prompt-only JSON is faithful there.
        body["response_format"] = {"type": "json_object"}
    if c.reasoning:
        body["reasoning"] = c.reasoning
    return body


def submit(client, requests: dict[str, tuple[str, str, bool]], label: str = "batch") -> dict:
    prov = _provider(client)
    handle = {"provider": prov, "model": client.name, "n": len(requests)}
    if prov == "openai":
        lines = [json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/responses",
                             "body": _openai_body(client, s, u)})
                 for cid, (s, u, _) in requests.items()]
        buf = io.BytesIO("\n".join(lines).encode())
        buf.name = f"{label}.jsonl"
        sdk = _openai_sdk()
        f = sdk.files.create(file=buf, purpose="batch")
        b = sdk.batches.create(input_file_id=f.id, endpoint="/v1/responses",
                               completion_window="24h")
        handle["id"] = b.id
    elif prov == "anthropic":
        b = _anthropic_sdk().messages.batches.create(requests=[
            {"custom_id": cid, "params": _anthropic_params(client, s, u)}
            for cid, (s, u, _) in requests.items()])
        handle["id"] = b.id
    else:
        payload = {"endpoint": "/v1/chat/completions", "model": client.model_id,
                   "requests": [{"custom_id": cid, "body": _openrouter_body(client, s, u, j)}
                                for cid, (s, u, j) in requests.items()]}
        r = httpx.post(_OR_API, headers=_or_headers(), content=json.dumps(payload),
                       timeout=600)
        r.raise_for_status()
        handle["id"] = r.json()["id"]
    return handle


def poll(handle: dict) -> tuple[bool, str]:
    prov = handle["provider"]
    if prov == "openai":
        b = _openai_sdk().batches.retrieve(handle["id"])
        rc = b.request_counts
        return (b.status in _OPENAI_TERMINAL,
                f"{b.status} {rc.completed}/{rc.total} ({rc.failed} failed)")
    if prov == "anthropic":
        b = _anthropic_sdk().messages.batches.retrieve(handle["id"])
        c = b.request_counts
        return (b.processing_status == "ended",
                f"{b.processing_status} {c.succeeded} ok / {c.errored} err / {handle['n']}")
    d = httpx.get(f"{_OR_API}/{handle['id']}", headers=_or_headers(), timeout=600).json()
    rc = d.get("request_counts") or {}
    return (d.get("status") in _OR_TERMINAL,
            f"{d.get('status')} {rc.get('completed', 0)}/{rc.get('total', '?')} "
            f"({rc.get('failed', 0)} failed)")


def cancel_and_wait(handle: dict, timeout: float = 900.0) -> bool:
    """Cancel, then wait for the batch to go terminal: finished results only
    become readable once cancellation has settled."""
    cancelled = cancel(handle)
    deadline = time.time() + timeout
    while cancelled and time.time() < deadline:
        try:
            if poll(handle)[0]:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(30)
    return cancelled


def cancel(handle: dict) -> bool:
    """Cancel a running batch; False where the provider has no cancel route."""
    prov = handle["provider"]
    if prov == "openai":
        _openai_sdk().batches.cancel(handle["id"])
        return True
    if prov == "anthropic":
        _anthropic_sdk().messages.batches.cancel(handle["id"])
        return True
    return False


def results_with_retry(handle: dict, attempts: int = 6, wait: float = 120.0) -> dict:
    """`results`, retried across transient network errors on large downloads."""
    for attempt in range(attempts):
        try:
            return results(handle)
        except Exception as e:  # noqa: BLE001
            print(f"  results error {handle['model']}: {e}", flush=True)
            if attempt == attempts - 1:
                raise
            time.sleep(wait)
    return {}


def results(handle: dict) -> dict[str, tuple[str, dict]]:
    prov = handle["provider"]
    out: dict[str, tuple[str, dict]] = {}
    if prov == "openai":
        sdk = _openai_sdk()
        b = sdk.batches.retrieve(handle["id"])
        for fid in filter(None, [b.output_file_id]):
            for line in sdk.files.content(fid).text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                resp = rec.get("response") or {}
                if rec.get("error") or resp.get("status_code") != 200:
                    continue
                body = resp.get("body") or {}
                text = "".join(c.get("text", "") for item in body.get("output", [])
                               if item.get("type") == "message"
                               for c in item.get("content", [])
                               if c.get("type") == "output_text")
                summary = "\n".join(s.get("text", "") for item in body.get("output", [])
                                    if item.get("type") == "reasoning"
                                    for s in item.get("summary", []) or [] if s.get("text"))
                out[rec["custom_id"]] = (text, {"reasoning": summary} if summary else {})
    elif prov == "anthropic":
        sdk = _anthropic_sdk()  # held: a collected client closes the result stream
        for res in sdk.messages.batches.results(handle["id"]):
            if res.result.type != "succeeded":
                continue
            msg = res.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            trace = "\n".join(b.thinking for b in msg.content
                              if b.type == "thinking" and getattr(b, "thinking", ""))
            meta = {"stop_reason": msg.stop_reason}
            if trace:
                meta["reasoning"] = trace
            out[res.custom_id] = (text, meta)
    else:
        d = httpx.get(f"{_OR_API}/{handle['id']}", headers=_or_headers(), timeout=600).json()
        for res in d.get("results") or []:
            resp = res.get("response") or {}
            if res.get("error") or resp.get("status_code") != 200:
                continue
            msg = ((resp.get("body") or {}).get("choices") or [{}])[0].get("message") or {}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or None
            if not reasoning and msg.get("reasoning_details"):
                parts = [x.get("text") or x.get("summary") or ""
                         for x in msg["reasoning_details"] if isinstance(x, dict)]
                reasoning = "\n".join(p for p in parts if p) or None
            content = msg.get("content") or ""
            if not content.strip():
                content = reasoning or ""
            out[res["custom_id"]] = (content, {"reasoning": reasoning} if reasoning else {})
    return out
