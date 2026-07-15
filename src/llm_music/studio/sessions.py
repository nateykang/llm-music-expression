"""Disk-backed session store.

Layout under <data_dir>/sessions/<id>/:
  meta.json      id, title, model, created, last_active, counters
  events.jsonl   append-only log of everything that happened (the research data)
  messages.json  raw Anthropic message list, replayed verbatim to resume a chat
  pieces/vN/     rendered versions (see pieces.py)

events.jsonl is the artifact you will actually study: composer messages,
assistant text, full thinking traces, every generated source, tool errors,
token usage. The UI also rebuilds its chat history from it.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path

_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")


class SessionStore:
    def __init__(self, root: Path):
        self.sessions_dir = root / "sessions"

    def _dir(self, session_id: str) -> Path:
        # The id doubles as a path component in file-serving routes; the strict
        # format check is what makes traversal impossible.
        if not _ID_RE.match(session_id):
            raise KeyError(f"bad session id: {session_id!r}")
        return self.sessions_dir / session_id

    # -- lifecycle ---------------------------------------------------------

    def create(self, model: str, title: str = "") -> dict:
        session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        d = self.sessions_dir / session_id
        (d / "pieces").mkdir(parents=True)
        meta = {
            "id": session_id,
            "title": title or "Untitled session",
            "model": model,
            "created": time.time(),
            "last_active": time.time(),
            "n_messages": 0,
            "n_versions": 0,
        }
        self._write_meta(session_id, meta)
        self.save_messages(session_id, [])
        return meta

    def list(self) -> list[dict]:
        if not self.sessions_dir.is_dir():
            return []
        metas = []
        for d in self.sessions_dir.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                metas.append(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        return sorted(metas, key=lambda m: m["last_active"], reverse=True)

    def get(self, session_id: str) -> dict | None:
        path = self._dir(session_id) / "meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_meta(self, session_id: str, meta: dict) -> None:
        (self._dir(session_id) / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def touch(self, session_id: str, **updates) -> dict:
        meta = self.get(session_id)
        if meta is None:
            raise KeyError(session_id)
        meta.update(updates)
        meta["last_active"] = time.time()
        self._write_meta(session_id, meta)
        return meta

    # -- transcript --------------------------------------------------------

    def append_event(self, session_id: str, event: dict) -> None:
        event = {"ts": time.time(), **event}
        with (self._dir(session_id) / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def events(self, session_id: str) -> list[dict]:
        path = self._dir(session_id) / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def messages(self, session_id: str) -> list:
        path = self._dir(session_id) / "messages.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def save_messages(self, session_id: str, messages: list) -> None:
        (self._dir(session_id) / "messages.json").write_text(
            json.dumps(messages, ensure_ascii=False), encoding="utf-8"
        )

    # -- pieces ------------------------------------------------------------

    def pieces_dir(self, session_id: str) -> Path:
        return self._dir(session_id) / "pieces"

    def piece_file(self, session_id: str, version: str, filename: str) -> Path:
        if not re.match(r"^v[0-9]+$", version):
            raise KeyError(f"bad version: {version!r}")
        from .pieces import VERSION_FILES

        if filename not in VERSION_FILES:
            raise KeyError(f"bad piece filename: {filename!r}")
        return self.pieces_dir(session_id) / version / filename
