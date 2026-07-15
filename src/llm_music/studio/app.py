"""FastAPI app: password gate, session CRUD, chat over SSE, piece-file serving,
and the static single-page UI.

Everything under /api requires the auth cookie except /api/login. The static
shell is public (it just shows the login form); all data is gated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.request
from collections import defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent
from . import config as cfg
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


# -- sessions ----------------------------------------------------------------


class NewSessionBody(BaseModel):
    model: str = ""
    title: str = ""


class MessageBody(BaseModel):
    text: str
    mode: str = "codegen"  # which render tool this turn offers: codegen | abc
    model: str = ""  # friendly model id; empty = the session's current model


@app.get("/api/sessions", dependencies=[Depends(require_auth)])
def list_sessions():
    return {"sessions": _store().list()}


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
    # The UI rebuilds chat + version timeline from the event log; thinking and
    # raw tool inputs stay server-side (they are research data, not chat).
    visible = [e for e in store.events(session_id)
               if e["type"] in ("user", "assistant", "piece", "error")]
    return {"meta": meta, "events": visible}


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

    import time

    if time.time() - meta.get("last_active", 0) > IDLE_NOTIFY_SECONDS:
        _notify({"event": "session_resumed", "session": session_id,
                 "title": meta.get("title", "")})

    # The turn runs in its own task feeding a queue; the SSE loop only reads the
    # queue. Two reasons: wait_for-cancelling an async generator's __anext__
    # (for keepalive pings) would kill the generator mid-turn, and this way a
    # dropped connection doesn't abort a turn — it finishes and is logged, and
    # the chat shows it on reload.
    queue: asyncio.Queue = asyncio.Queue()

    async def run_turn():
        async with lock:
            try:
                async for event in agent.stream_turn(store, session_id, text,
                                                     mode=body.mode,
                                                     model=body.model):
                    queue.put_nowait(event)
            except Exception as e:
                log.exception("turn failed in session %s", session_id)
                err = {"type": "error", "message": f"turn failed: {e}"}
                store.append_event(session_id, err)
                queue.put_nowait(err)
            finally:
                queue.put_nowait(None)

    task = asyncio.create_task(run_turn())
    _turn_tasks.add(task)
    task.add_done_callback(_turn_tasks.discard)

    async def sse():
        while True:
            # Ping through long thinking silences so proxies keep the
            # stream open and the browser knows we're alive.
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
