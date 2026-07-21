"""Comparison mode: fan one prompt out to many (model × method) cells, each an
INDEPENDENT generation — no awareness of the other cells. That isolation is
the point: an expert judging the grid isn't seeing outputs that were
conditioned on each other.

A comparison is conversational per cell: every follow-up composer message goes
to ALL cells, and each cell continues its own private thread (its history is
just its own prior work — never another cell's). Round r, cell i lands at
``pieces/v{r*n_cells + i + 1}/`` so the existing piece-file route serves it
unchanged and the grid rebuilds from the event log like any other session.

This reuses the studio's provider backends, render tools, and sandbox so a
comparison cell and a chat turn produce identical artifacts (score + audio +
analysis).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import secrets
import shutil

from starlette.concurrency import run_in_threadpool

from .agent import (
    BACKENDS,
    MAX_TURN_STEPS,
    REQUIRED_KEYS,
    TOOLS_BY_MODE,
    _model_spec,
    _strip_thinking,
    system_prompt,
)
from .pieces import render_abc_version, render_codegen_version
from .sessions import SessionStore

log = logging.getLogger(__name__)

# A human can't meaningfully compare more than a handful of results side by
# side; this also bounds the fan-out (worst case MAX_MODELS × both methods
# cells). Enforced in app.py — rejected, never silently truncated.
MAX_MODELS = 5

# Synthetic user turn injected when a cell chats instead of rendering. Also the
# marker fork_cell_to_chat uses to keep these out of the rebuilt chat transcript.
NUDGE = "Compose the piece now by calling the render tool."


def build_cells(models: list[str], modes: list[str]) -> list[dict]:
    """The (model × mode) grid, deduped. Order is stable so cell index ->
    version is deterministic across a rerun."""
    seen, cells = set(), []
    for mode in modes:
        for model in models:
            key = (model, mode)
            if key in seen:
                continue
            seen.add(key)
            cells.append({"model": model, "mode": mode})
    return cells


def _comparison_tools(mode: str) -> list:
    """Comparison cells render into a grid where chat commentary is invisible,
    so the render tool grows a structured one-line intent shown beside the
    score. Chat turns keep the plain tools — there the commentary IS the chat."""
    tools = copy.deepcopy(TOOLS_BY_MODE[mode])
    for t in tools:
        t["input_schema"]["properties"]["short_description"] = {
            "type": "string",
            "description": "A single sentence describing your musical intent with this piece.",
        }
        t["input_schema"]["required"].append("short_description")
    return tools


async def _one_render(pieces_dir, mode, inp, version, extra):
    if mode == "codegen":
        return await run_in_threadpool(
            render_codegen_version, pieces_dir, inp.get("code", ""),
            inp.get("title", "Untitled"), inp.get("note", ""), version, extra)
    return await run_in_threadpool(
        render_abc_version, pieces_dir, inp.get("abc", ""),
        inp.get("title", "Untitled"), inp.get("note", ""), version, extra)


async def run_cell(store: SessionStore, comp_id: str, user_text: str,
                   model: str, mode: str, index: int, version: int,
                   round_idx: int) -> dict:
    """One cell's turn for one round. Continues the cell's own conversation
    (round 0 starts it). Returns a cell result (a piece event on success, or an
    error cell). Never raises — a broken cell must not sink the whole grid.
    Every outcome, including failures, is appended to the event log so a
    reopened grid shows it instead of an eternal 'composing…'."""
    base = {"type": "cell", "index": index, "round": round_idx, "model": model,
            "mode": mode, "version": version}

    def fail(err: str) -> dict:
        ev = {**base, "ok": False, "error": err}
        store.append_event(comp_id, ev)
        return ev

    try:
        provider, model_id, options = _model_spec(model)
    except (KeyError, ValueError) as e:
        return fail(str(e))
    if not os.environ.get(REQUIRED_KEYS[provider]):
        return fail(f"server missing {REQUIRED_KEYS[provider]}")

    tools = _comparison_tools(mode)
    backend = BACKENDS[provider]
    messages = _strip_thinking(store.cell_messages(comp_id, index))
    messages.append({"role": "user", "content": user_text})
    pieces_dir = store.pieces_dir(comp_id)
    extra = {"model": model, "prompt": user_text, "round": round_idx}

    def save() -> None:
        store.save_cell_messages(comp_id, index, messages)

    last_err = "model produced no render"
    for _ in range(MAX_TURN_STEPS):
        try:
            final = None
            async for ev in backend(model_id, options, system_prompt(mode), tools, messages):
                if ev.get("type") == "_result":
                    final = ev
            if final is None:
                save()
                return fail("no response from model")
        except Exception as e:  # noqa: BLE001 — isolate provider/network failure to this cell
            log.warning("compare cell %s (%s) failed: %s", index, model, e)
            save()
            return fail(f"model error: {e}")

        blocks = final["blocks"]
        if not blocks:
            save()
            return fail("model returned nothing")
        messages.append({"role": "assistant", "content": blocks})
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        if not tool_uses:
            # The model chatted instead of rendering — nudge once via a synthetic
            # user turn, else give up (no per-cell human to reply).
            last_err = "model responded with text instead of composing"
            messages.append({"role": "user", "content": NUDGE})
            continue

        tu = tool_uses[0]
        desc = (tu["input"].get("short_description") or "").strip()
        result = await _one_render(pieces_dir, mode, tu["input"], version,
                                   {**extra, "short_description": desc})
        if result.get("ok"):
            # Close the tool cycle in the saved history so the next round's
            # request replays cleanly (a dangling tool_use is rejected).
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": tu["id"],
                "content": json.dumps(result, ensure_ascii=False),
                "is_error": False}]})
            save()
            piece = {**base, "ok": True, "title": result["title"],
                     "note": result["note"], "short_description": desc,
                     "files": result["files"], "analysis": result["analysis"]}
            store.append_event(comp_id, piece)
            return piece
        last_err = result.get("error") or last_err
        # Every tool_use in the assistant turn needs a matching tool_result or
        # the next request is rejected; extras beyond the first weren't run.
        results = [{"type": "tool_result", "tool_use_id": tu["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                    "is_error": True}]
        for extra_tu in tool_uses[1:]:
            results.append({
                "type": "tool_result", "tool_use_id": extra_tu["id"],
                "content": json.dumps({"ok": False, "error":
                                       "only the first render call per response is executed"},
                                      ensure_ascii=False),
                "is_error": True})
        messages.append({"role": "user", "content": results})

    save()
    return fail(last_err)


async def run_round(store: SessionStore, comp_id: str, text: str,
                    cells: list[dict], round_idx: int):
    """Run one composer message across every cell concurrently; yield each
    result as it finishes so the grid fills in progressively. Independence
    means order of completion is arbitrary."""
    store.append_event(comp_id, {"type": "prompt", "text": text,
                                 "round": round_idx, "cells": cells})
    yield {"type": "start", "prompt": text, "cells": cells, "round": round_idx}

    n = len(cells)
    tasks = [asyncio.create_task(run_cell(store, comp_id, text,
                                          c["model"], c["mode"], i,
                                          round_idx * n + i + 1, round_idx))
             for i, c in enumerate(cells)]
    n_ok = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result.get("ok"):
            n_ok += 1
        yield result
    meta = store.get(comp_id) or {}
    # n_versions counts what actually rendered, so History stays honest.
    store.touch(comp_id, n_versions=meta.get("n_versions", 0) + n_ok,
                n_messages=round_idx + 1)
    yield {"type": "done", "n_cells": n, "n_ok": n_ok, "round": round_idx}


_TOOL_BY_MODE = {"codegen": ("render_music21", "code", "source.py"),
                 "abc": ("render_abc", "abc", "piece.abc")}


def _synth_history(store: SessionStore, comp_id: str, ok_evs: list[dict],
                   mode: str, fallback_prompt: str) -> list:
    """Comparisons from before per-cell histories were stored on disk:
    reconstruct a minimal conversation (prompt -> render -> result) per
    version from the piece dirs, so those cells stay forkable."""
    tool_name, arg_key, source_file = _TOOL_BY_MODE[mode]
    msgs: list = []
    for e in ok_evs:
        src = store.pieces_dir(comp_id) / f"v{e['version']}"
        m = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        source = ""
        if (src / source_file).exists():
            source = (src / source_file).read_text(encoding="utf-8")
        tid = "toolu_fork_" + secrets.token_hex(5)
        msgs.append({"role": "user", "content": m.get("prompt") or fallback_prompt})
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tid, "name": tool_name,
             "input": {arg_key: source, "title": m.get("title", "Untitled"),
                       "note": m.get("note", "")}}]})
        msgs.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": tid,
            "content": json.dumps({"ok": True, "version": e["version"],
                                   "mode": mode,
                                   "title": m.get("title", "Untitled"),
                                   "note": m.get("note", ""),
                                   "files": m.get("files", []),
                                   "analysis": m.get("analysis", {})},
                                  ensure_ascii=False),
            "is_error": False}]})
    return msgs


def fork_cell_to_chat(store: SessionStore, comp_id: str, index: int) -> dict:
    """Turn one comparison cell into a fresh chat on the cell's own model,
    carrying the cell's ENTIRE conversation — every round's prompt, the model's
    commentary, and every rendered version — so the model (and the composer)
    resume exactly where the cell left off. The cell's versions are renumbered
    v1..vK in the chat, including inside replayed tool results."""
    if store.get(comp_id) is None:
        raise KeyError(comp_id)
    ok_evs = [e for e in store.events(comp_id)
              if e["type"] == "cell" and e["index"] == index and e.get("ok")]
    if not ok_evs:
        raise ValueError(f"cell {index} has no rendered piece")
    last = ok_evs[-1]
    model, mode = last["model"], last["mode"]
    messages = store.cell_messages(comp_id, index)
    if not messages:
        first_prompt = next((e["text"] for e in store.events(comp_id)
                             if e["type"] == "prompt"), "")
        messages = _synth_history(store, comp_id, ok_evs, mode, first_prompt)

    chat = store.create(model=model,
                        title=f"↳ {last.get('title', 'Untitled')} · {model}",
                        kind="chat")
    cid = chat["id"]

    # Copy the cell's rendered versions in as v1..vK (independent of the
    # comparison's grid-wide numbering).
    vmap: dict[int, int] = {}
    for n, e in enumerate(ok_evs, start=1):
        src = store.pieces_dir(comp_id) / f"v{e['version']}"
        dst = store.pieces_dir(cid) / f"v{n}"
        shutil.copytree(src, dst)
        m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
        m["version"] = n
        (dst / "meta.json").write_text(json.dumps(m, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
        vmap[e["version"]] = n

    def renumber(block: dict) -> dict:
        if block.get("type") != "tool_result":
            return block
        try:
            obj = json.loads(block.get("content") or "")
        except (json.JSONDecodeError, TypeError):
            return block
        if isinstance(obj, dict) and obj.get("version") in vmap:
            obj["version"] = vmap[obj["version"]]
            return {**block, "content": json.dumps(obj, ensure_ascii=False)}
        return block

    new_msgs = []
    for m in messages:
        if isinstance(m.get("content"), list):
            m = {**m, "content": [renumber(b) for b in m["content"]]}
        new_msgs.append(m)
    store.save_messages(cid, new_msgs)

    # Rebuild the chat's visible transcript from the replayed conversation:
    # composer rounds (skipping synthetic nudges), assistant commentary, and a
    # piece card per successful render.
    ok_by_new_version = {vmap[e["version"]]: e for e in ok_evs}
    n_user = 0
    for m in new_msgs:
        content = m.get("content")
        if m["role"] == "user" and isinstance(content, str):
            if content == NUDGE:
                continue
            n_user += 1
            store.append_event(cid, {"type": "user", "text": content,
                                     "mode": mode, "model": model})
        elif m["role"] == "assistant" and isinstance(content, list):
            for b in content:
                if b.get("type") == "text" and b.get("text"):
                    store.append_event(cid, {"type": "assistant",
                                             "text": b["text"]})
        elif m["role"] == "user" and isinstance(content, list):
            for b in content:
                if b.get("type") != "tool_result" or b.get("is_error"):
                    continue
                try:
                    obj = json.loads(b.get("content") or "")
                except (json.JSONDecodeError, TypeError):
                    continue
                e = ok_by_new_version.get(obj.get("version")) if isinstance(obj, dict) else None
                if e is None:
                    continue
                store.append_event(cid, {
                    "type": "piece", "version": obj["version"], "mode": mode,
                    "model": model, "title": e.get("title", "Untitled"),
                    "note": e.get("note", ""), "files": e.get("files", []),
                    "analysis": e.get("analysis", {})})
    store.touch(cid, n_versions=len(ok_evs), n_messages=n_user)
    return chat
