"""FastAPI app: password gate, session CRUD, chat over SSE, piece-file serving,
and the static single-page UI.

Everything under /api requires the auth cookie except /api/login. The static
shell is public (it just shows the login form); all data is gated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent
from . import compare as compare_mod
from . import config as cfg
from . import review as review_mod
from .auth import LoginLimiter, check_password, make_token, verify_token
from .sessions import SessionStore

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
IDLE_NOTIFY_SECONDS = 2 * 3600  # "he's back" webhook after this much quiet

app = FastAPI(title="Composer Studio", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_limiter = LoginLimiter()
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_turn_tasks: set[asyncio.Task] = set()  # strong refs so running turns aren't GC'd


def _store() -> SessionStore:
    return SessionStore(cfg.data_dir())


# -- pace limit ---------------------------------------------------------------
# The API keys have no provider-side caps the composer can set, so the studio
# itself brakes spending: at most STUDIO_RATE_LIMIT generations per
# STUDIO_RATE_WINDOW seconds (defaults: 400/hour) across ALL sessions. One
# chat message = 1 unit; a
# comparison round = 1 unit per cell (each cell is its own model conversation).
# In-memory on purpose — resets on restart, which is fine for a cost brake.
_pace: deque = deque()  # timestamps of recently started generations


def _check_pace(cost: int) -> None:
    limit = int(os.environ.get("STUDIO_RATE_LIMIT", "400"))
    window = float(os.environ.get("STUDIO_RATE_WINDOW", "3600"))
    now = time.time()
    while _pace and _pace[0] <= now - window:
        _pace.popleft()
    if len(_pace) + cost > limit:
        wait = int(_pace[0] + window - now) + 1 if _pace else int(window)
        raise HTTPException(
            status_code=429,
            detail=f"taking a breather — the studio allows {limit} generations "
                   f"per {int(window // 60)} minutes; try again in about "
                   f"{max(1, -(-wait // 60))} min")
    _pace.extend([now] * cost)


def require_auth(request: Request) -> None:
    if not verify_token(cfg.secret(), request.cookies.get(cfg.COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="not logged in")


def _notify(payload: dict) -> None:
    """Fire-and-forget webhook (STUDIO_NOTIFY_URL) so you hear about activity."""
    url = cfg.notify_url()
    if not url:
        return

    def post():
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            log.warning("notify webhook failed: %s", e)

    threading.Thread(target=post, daemon=True).start()


# -- auth -------------------------------------------------------------------


class LoginBody(BaseModel):
    password: str


@app.post("/api/login")
def login(body: LoginBody, request: Request, response: Response):
    ip = request.client.host if request.client else "?"
    if _limiter.blocked(ip):
        raise HTTPException(status_code=429, detail="too many attempts — try later")
    if not cfg.password():
        raise HTTPException(status_code=500, detail="server has no STUDIO_PASSWORD set")
    if not check_password(body.password, cfg.password()):
        _limiter.record_failure(ip)
        raise HTTPException(status_code=401, detail="wrong password")
    _limiter.reset(ip)
    response.set_cookie(
        cfg.COOKIE_NAME,
        make_token(cfg.secret(), cfg.TOKEN_TTL_SECONDS),
        max_age=cfg.TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(cfg.COOKIE_NAME)
    return {"ok": True}


@app.get("/api/me", dependencies=[Depends(require_auth)])
def me():
    return {"models": cfg.available_models(), "default_model": cfg.default_model()}


# -- listeners ---------------------------------------------------------------
# The studio is shared by several people behind one password. A "user" here is
# just a self-declared display name that tags what they write (and scopes what
# they see back), so reviewers don't collide — it is labeling, not auth.


def _users_path() -> Path:
    return cfg.data_dir() / "users.json"


def _load_users() -> list[str]:
    path = _users_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("users", [])


class UserBody(BaseModel):
    name: str


@app.get("/api/users", dependencies=[Depends(require_auth)])
def list_users():
    return {"users": _load_users()}


@app.post("/api/users", dependencies=[Depends(require_auth)])
def register_user(body: UserBody):
    name = " ".join(body.name.split())[:40]
    if not name:
        raise HTTPException(status_code=400, detail="empty name")
    users = _load_users()
    match = next((u for u in users if u.lower() == name.lower()), None)
    if match is None:
        users.append(name)
        _users_path().parent.mkdir(parents=True, exist_ok=True)
        _users_path().write_text(json.dumps({"users": users}, indent=2),
                                 encoding="utf-8")
        _notify({"event": "user_registered", "name": name})
        match = name
    return {"users": users, "name": match}


# -- sessions ----------------------------------------------------------------


class NewSessionBody(BaseModel):
    model: str = ""
    title: str = ""


class MessageBody(BaseModel):
    text: str
    mode: str = "codegen"  # which render tool this turn offers: codegen | abc
    model: str = ""  # friendly model id; empty = the session's current model


@app.get("/api/sessions", dependencies=[Depends(require_auth)])
def list_sessions(user: str = ""):
    store = _store()
    user = user.strip()
    sessions = store.list()
    for meta in sessions:
        if meta.get("kind") == "review":
            # The list shows the requesting listener's own progress: notes are
            # per person, so someone new sees 0, not the previous reviewer's.
            meta["n_notes"] = sum(
                1 for e in store.events(meta["id"])
                if e["type"] == "review_note" and e.get("user", "") == user)
    return {"sessions": sessions}


@app.post("/api/sessions", dependencies=[Depends(require_auth)])
def create_session(body: NewSessionBody):
    model = body.model or cfg.default_model()
    if model not in cfg.available_models():
        raise HTTPException(status_code=400, detail=f"unknown model '{model}'")
    meta = _store().create(model=model, title=body.title.strip())
    _notify({"event": "session_created", "session": meta["id"], "title": meta["title"]})
    return meta


@app.get("/api/sessions/{session_id}", dependencies=[Depends(require_auth)])
def get_session(session_id: str):
    store = _store()
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such session")
    if meta is None:
        raise HTTPException(status_code=404, detail="no such session")
    # The UI rebuilds chat/grid from the event log; thinking and raw tool inputs
    # stay server-side (research data, not shown). "prompt"/"cell" reconstruct a
    # comparison grid; "user"/"assistant"/"piece" reconstruct a chat; "comment"
    # is the composer's own annotation on a version (never sent to the model).
    visible = [e for e in store.events(session_id)
               if e["type"] in ("user", "assistant", "piece", "error",
                                 "prompt", "cell", "comment")]
    return {"meta": meta, "events": visible}


class CommentBody(BaseModel):
    text: str
    version: int | None = None  # which piece version the thought is about
    user: str = ""  # optional display name (see /api/users)


@app.post("/api/sessions/{session_id}/comments",
          dependencies=[Depends(require_auth)])
def post_comment(session_id: str, body: CommentBody):
    """The composer's written reaction to a piece — research data, appended to
    the event log and shown in the UI, but never sent to any model."""
    store = _store()
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such session")
    if meta is None:
        raise HTTPException(status_code=404, detail="no such session")
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty comment")
    event = {"type": "comment", "text": text, "version": body.version}
    if body.user.strip():
        event["user"] = body.user.strip()
    store.append_event(session_id, event)
    store.touch(session_id)
    _notify({"event": "comment", "session": session_id, "text": text[:80]})
    return {"ok": True}


@app.post("/api/sessions/{session_id}/message", dependencies=[Depends(require_auth)])
async def post_message(session_id: str, body: MessageBody):
    store = _store()
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such session")
    if meta is None:
        raise HTTPException(status_code=404, detail="no such session")
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty message")
    if body.mode not in agent.TOOLS_BY_MODE:
        raise HTTPException(status_code=400, detail=f"unknown mode '{body.mode}'")
    if body.model and body.model not in cfg.available_models():
        raise HTTPException(status_code=400, detail=f"unknown model '{body.model}'")
    lock = _locks[session_id]
    if lock.locked():
        raise HTTPException(status_code=409, detail="a turn is already running")
    _check_pace(1)

    if time.time() - meta.get("last_active", 0) > IDLE_NOTIFY_SECONDS:
        _notify({"event": "session_resumed", "session": session_id,
                 "title": meta.get("title", "")})

    async def turn():
        async with lock:
            async for event in agent.stream_turn(store, session_id, text,
                                                 mode=body.mode,
                                                 model=body.model):
                yield event

    return _stream_events(turn(), store, session_id)


def _stream_events(gen, store: SessionStore, session_id: str,
                   lead: dict | None = None) -> StreamingResponse:
    """SSE-serialize an event generator, running it in its own task feeding a
    queue; the SSE loop only reads the queue. Two reasons: wait_for-cancelling
    an async generator's __anext__ (for keepalive pings) would kill the
    generator mid-turn, and this way a dropped connection doesn't abort the
    work — it finishes and is logged, and the session shows it on reload.
    ``lead`` is emitted first (e.g. the created session's meta for routing)."""
    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        try:
            async for event in gen:
                queue.put_nowait(event)
        except Exception as e:
            import traceback

            log.error("stream failed in session %s:\n%s", session_id,
                      cfg.redact(traceback.format_exc()))
            err = {"type": "error", "message": cfg.redact(str(e))}
            store.append_event(session_id, err)
            queue.put_nowait(err)
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(run())
    _turn_tasks.add(task)
    task.add_done_callback(_turn_tasks.discard)

    async def sse():
        if lead is not None:
            yield "data: " + json.dumps(lead, ensure_ascii=False) + "\n\n"
        while True:
            # Ping through long silences so proxies keep the stream open and
            # the browser knows we're alive.
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if event is None:
                break
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class ComparisonBody(BaseModel):
    prompt: str
    models: list[str]
    modes: list[str] = ["codegen"]  # any of codegen | abc
    title: str = ""


@app.post("/api/comparisons", dependencies=[Depends(require_auth)])
async def create_comparison(body: ComparisonBody):
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    models = [m for m in body.models if m]
    if not models:
        raise HTTPException(status_code=400, detail="pick at least one model")
    bad = [m for m in models if m not in cfg.available_models()]
    if bad:
        raise HTTPException(status_code=400, detail=f"unknown model(s): {bad}")
    bad_modes = [m for m in body.modes if m not in agent.TOOLS_BY_MODE]
    if bad_modes or not body.modes:
        raise HTTPException(status_code=400, detail=f"bad mode(s): {bad_modes or 'none'}")

    n_models = len(dict.fromkeys(models))
    if n_models > compare_mod.MAX_MODELS:
        # Reject rather than silently truncate — a grid missing models the
        # composer picked would just look like they were never selected.
        raise HTTPException(
            status_code=400,
            detail=f"{n_models} models selected but the limit is "
                   f"{compare_mod.MAX_MODELS} per comparison — more than "
                   f"that is hard to compare side by side")
    cells = compare_mod.build_cells(models, body.modes)
    _check_pace(len(cells))
    store = _store()
    meta = store.create(model=models[0], title=body.title.strip(), kind="comparison")
    comp_id = meta["id"]
    _notify({"event": "comparison_created", "session": comp_id,
             "prompt": prompt[:80], "n_cells": len(cells)})

    async def first_round():
        async with _locks[comp_id]:
            async for ev in compare_mod.run_round(store, comp_id, prompt, cells, 0):
                yield ev

    # Lead with the session meta so the browser can route to the grid view.
    return _stream_events(first_round(), store, comp_id,
                          lead={"type": "created", "meta": meta})


class CompareMessageBody(BaseModel):
    text: str


@app.post("/api/sessions/{session_id}/compare-message",
          dependencies=[Depends(require_auth)])
async def post_compare_message(session_id: str, body: CompareMessageBody):
    """A follow-up composer message to ALL of a comparison's cells; each cell
    continues its own independent conversation."""
    store = _store()
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such session")
    if meta is None:
        raise HTTPException(status_code=404, detail="no such session")
    if meta.get("kind") != "comparison":
        raise HTTPException(status_code=400, detail="not a comparison session")
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty message")
    prompts = [e for e in store.events(session_id) if e["type"] == "prompt"]
    if not prompts:
        raise HTTPException(status_code=409, detail="comparison has no first round")
    cells = prompts[0]["cells"]  # the grid is fixed at creation
    round_idx = len(prompts)
    lock = _locks[session_id]
    if lock.locked():
        raise HTTPException(status_code=409, detail="a round is already running")
    _check_pace(len(cells))

    async def next_round():
        async with lock:
            async for ev in compare_mod.run_round(store, session_id, text,
                                                  cells, round_idx):
                yield ev

    return _stream_events(next_round(), store, session_id)


@app.post("/api/sessions/{session_id}/cells/{index}/fork",
          dependencies=[Depends(require_auth)])
def fork_cell(session_id: str, index: int):
    try:
        chat = compare_mod.fork_cell_to_chat(_store(), session_id, index)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such comparison")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return chat


# -- listening sessions -------------------------------------------------------
# Qualitative review of already-generated batch pieces: no model is called,
# nothing is spent. Blind sessions never send model identities (or even the
# model-named file paths) to the browser until an explicit, logged reveal.


@app.get("/api/review/batches", dependencies=[Depends(require_auth)])
def review_batches():
    return {"batches": review_mod.list_batches(cfg.batches_dir())}


class NewReviewBody(BaseModel):
    batches: list[str]
    models: list[str]
    prompts: list[str] = []  # empty = every prompt in the chosen batches
    per_cell: int = 1  # pieces per (model x prompt x writing method) cell
    blind: bool = True
    seed: int = 0  # logged in the setup event, so any queue is reproducible
    title: str = ""


@app.post("/api/reviews", dependencies=[Depends(require_auth)])
def create_review(body: NewReviewBody):
    batches = [b for b in dict.fromkeys(body.batches) if b]
    models = [m for m in dict.fromkeys(body.models) if m]
    if not batches:
        raise HTTPException(status_code=400, detail="pick at least one batch")
    if not models:
        raise HTTPException(status_code=400, detail="pick at least one model")
    if body.per_cell < 1:
        raise HTTPException(status_code=400, detail="pieces per cell must be at least 1")
    try:
        setup = review_mod.build_setup(cfg.batches_dir(), batches, models,
                                       body.per_cell, body.seed, body.blind,
                                       prompts=[p for p in body.prompts if p] or None)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ts = batches[0].split("__")[0]
    store = _store()
    meta = store.create(
        model="",
        title=body.title.strip() or f"Listening session {ts[:4]}-{ts[4:6]}-{ts[6:8]}",
        kind="review")
    store.append_event(meta["id"], {"type": "review_setup", **setup})
    meta = store.touch(meta["id"], batches=batches, blind=body.blind,
                       revealed=not body.blind, n_pieces=len(setup["pieces"]))
    _notify({"event": "review_created", "session": meta["id"],
             "title": meta["title"], "n_pieces": len(setup["pieces"])})
    return {"meta": meta, **review_mod.client_view(setup, revealed=meta["revealed"])}


def _review(session_id: str) -> tuple[SessionStore, dict, dict]:
    store = _store()
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such session")
    if meta is None or meta.get("kind") != "review":
        raise HTTPException(status_code=404, detail="no such listening session")
    setups = [e for e in store.events(session_id) if e["type"] == "review_setup"]
    if not setups:
        raise HTTPException(status_code=409, detail="listening session has no pieces")
    return store, meta, setups[0]


def _user_revealed(meta: dict, user: str) -> bool:
    # Reveal state is per listener: one person unblinding must not unblind the
    # next. (meta["revealed"] survives as a legacy whole-session flag.)
    return bool(meta.get("revealed")) or (bool(user) and
                                          user in meta.get("revealed_by", []))


@app.get("/api/reviews/{session_id}", dependencies=[Depends(require_auth)])
def get_review(session_id: str, user: str = ""):
    store, meta, setup = _review(session_id)
    user = user.strip()
    # Each listener sees only their own notes — independent judgments, no
    # anchoring on what someone else already wrote.
    notes = [{"piece": e.get("piece"), "text": e["text"],
              "revealed": e.get("revealed", False)}
             for e in store.events(session_id)
             if e["type"] == "review_note" and e.get("user", "") == user]
    return {"meta": meta, "notes": notes, "user": user,
            **review_mod.client_view(setup, revealed=_user_revealed(meta, user))}


class ReviewNoteBody(BaseModel):
    text: str
    piece: int | None = None  # queue index; None = overall comparison notes
    user: str = ""


@app.post("/api/reviews/{session_id}/notes", dependencies=[Depends(require_auth)])
def post_review_note(session_id: str, body: ReviewNoteBody):
    store, meta, setup = _review(session_id)
    text = body.text.strip()
    user = body.user.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty note")
    if not user:
        raise HTTPException(status_code=400,
                            detail="add your name at the top of the studio first")
    if body.piece is not None and not 0 <= body.piece < len(setup["pieces"]):
        raise HTTPException(status_code=404, detail="no such piece")
    # Stamp the blind state at write time: blind notes and post-reveal notes
    # are different data in the qualitative analysis.
    store.append_event(session_id, {"type": "review_note", "piece": body.piece,
                                    "text": text, "user": user,
                                    "revealed": _user_revealed(meta, user)})
    store.touch(session_id, n_notes=meta.get("n_notes", 0) + 1)
    _notify({"event": "review_note", "session": session_id, "user": user,
             "text": text[:80]})
    return {"ok": True}


class RevealBody(BaseModel):
    user: str = ""


@app.post("/api/reviews/{session_id}/reveal", dependencies=[Depends(require_auth)])
def reveal_review(session_id: str, body: RevealBody):
    store, meta, setup = _review(session_id)
    user = body.user.strip()
    if not user:
        raise HTTPException(status_code=400,
                            detail="add your name at the top of the studio first")
    if not _user_revealed(meta, user):
        store.append_event(session_id, {"type": "review_reveal", "user": user})
        store.touch(session_id,
                    revealed_by=meta.get("revealed_by", []) + [user])
        _notify({"event": "review_reveal", "session": session_id, "user": user})
    return review_mod.client_view(setup, revealed=True)


@app.get("/api/reviews/{session_id}/pieces/{idx}/{kind}",
         dependencies=[Depends(require_auth)])
def review_piece_file(session_id: str, idx: int, kind: str):
    _, _, setup = _review(session_id)
    if not 0 <= idx < len(setup["pieces"]):
        raise HTTPException(status_code=404, detail="no such piece")
    p = setup["pieces"][idx]
    if kind == "abc":
        if not p.get("abc"):
            raise HTTPException(status_code=404, detail="no ABC for this piece")
        return PlainTextResponse(p["abc"])
    if kind not in ("audio", "score") or not p.get(kind):
        raise HTTPException(status_code=404, detail="no such file")
    try:
        path = review_mod.resolve_batch_file(cfg.batches_dir(), p["batch"], p[kind])
    except KeyError:
        raise HTTPException(status_code=404, detail="no such file")
    if not path.exists():
        raise HTTPException(status_code=404, detail="no such file")
    return FileResponse(path)


@app.get("/api/sessions/{session_id}/features/{version}",
         dependencies=[Depends(require_auth)])
async def piece_features(session_id: str, version: int):
    """Measured symbolic features for one version — the same panel the batch
    analysis computes, so the composer can eyeball (and comment on) them."""
    from starlette.concurrency import run_in_threadpool

    from . import features as features_mod

    store = _store()
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such session")
    if meta is None:
        raise HTTPException(status_code=404, detail="no such session")
    vdir = store.pieces_dir(session_id) / f"v{version}"
    if not (vdir / "meta.json").exists():
        raise HTTPException(status_code=404, detail="no such version")
    feats = await run_in_threadpool(features_mod.piece_features, vdir)
    if feats is None:
        return {"ok": False, "error": "this piece could not be analyzed"}
    return {"ok": True, "features": feats}


class PromptBody(BaseModel):
    text: str | None = None  # None/empty = reset to the default prompt


@app.get("/api/sessions/{session_id}/prompt", dependencies=[Depends(require_auth)])
def get_prompt(session_id: str, mode: str = "codegen"):
    """The session's effective system prompt: the custom override if set, plus
    the default for prefilling the editor."""
    store = _store()
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such session")
    if meta is None:
        raise HTTPException(status_code=404, detail="no such session")
    if mode not in agent.TOOLS_BY_MODE:
        raise HTTPException(status_code=400, detail=f"unknown mode '{mode}'")
    return {"custom": meta.get("custom_prompt"),
            "default": agent.system_prompt(mode)}


@app.put("/api/sessions/{session_id}/prompt", dependencies=[Depends(require_auth)])
def put_prompt(session_id: str, body: PromptBody):
    """Set (or clear) the session's system-prompt override. Logged to the event
    stream so every generation's prompt provenance is reconstructable."""
    store = _store()
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such session")
    if meta is None:
        raise HTTPException(status_code=404, detail="no such session")
    text = (body.text or "").strip() or None
    store.append_event(session_id, {"type": "system_prompt", "text": text})
    meta = store.touch(session_id, custom_prompt=text)
    return {"custom": meta.get("custom_prompt")}


@app.get("/api/sessions/{session_id}/pieces/{version}/{filename}",
         dependencies=[Depends(require_auth)])
def piece_file(session_id: str, version: str, filename: str):
    try:
        path = _store().piece_file(session_id, version, filename)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such file")
    if not path.exists():
        raise HTTPException(status_code=404, detail="no such file")
    return FileResponse(path)


# -- UI shell ----------------------------------------------------------------


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
