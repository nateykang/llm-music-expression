"""Studio tests: auth tokens, session store, render executors, HTTP surface.

No network: the agent loop itself (Anthropic calls) is exercised only up to its
guard clauses; tool executors run the real sandbox/renderers locally.
"""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from llm_music.studio import config as cfg
from llm_music.studio.auth import LoginLimiter, check_password, make_token, verify_token
from llm_music.studio.pieces import render_abc_version, render_codegen_version
from llm_music.studio.sessions import SessionStore


@pytest.fixture()
def studio_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_PASSWORD", "opensesame")
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "studio_data"))
    monkeypatch.setenv("STUDIO_SECRET", "test-secret")
    return tmp_path


# -- auth ---------------------------------------------------------------------


def test_token_roundtrip():
    secret = b"s1"
    token = make_token(secret, ttl_seconds=60)
    assert verify_token(secret, token)
    assert not verify_token(b"other", token)
    assert not verify_token(secret, token + "x")
    assert not verify_token(secret, None)
    assert not verify_token(secret, make_token(secret, ttl_seconds=-5))  # expired


def test_check_password():
    assert check_password("abc", "abc")
    assert not check_password("abc", "abcd")
    assert not check_password("abc", None)  # unset on server -> never authenticates


def test_login_limiter():
    lim = LoginLimiter(max_failures=2, window=60)
    assert not lim.blocked("ip")
    lim.record_failure("ip")
    assert not lim.blocked("ip")
    lim.record_failure("ip")
    assert lim.blocked("ip")
    lim.reset("ip")
    assert not lim.blocked("ip")


# -- session store --------------------------------------------------------------


def test_session_store_lifecycle(tmp_path):
    store = SessionStore(tmp_path)
    meta = store.create(model="fable-5", title="Nocturne sketches")
    sid = meta["id"]
    assert store.get(sid)["title"] == "Nocturne sketches"
    assert store.list()[0]["id"] == sid

    store.append_event(sid, {"type": "user", "text": "hello"})
    events = store.events(sid)
    assert events[0]["type"] == "user" and "ts" in events[0]

    store.save_messages(sid, [{"role": "user", "content": "hello"}])
    assert store.messages(sid)[0]["role"] == "user"

    store.touch(sid, n_versions=3)
    assert store.get(sid)["n_versions"] == 3

    with pytest.raises(KeyError):
        store.get("../../etc")  # traversal-shaped ids are rejected outright
    with pytest.raises(KeyError):
        store.piece_file(sid, "v1", "secrets.txt")  # non-allowlisted filename
    with pytest.raises(KeyError):
        store.piece_file(sid, "..", "piece.mp3")


# -- render executors -----------------------------------------------------------


CODE = """
from music21 import chord, note, stream

score = stream.Score()
part = stream.Part()
part.append(note.Note("C4", quarterLength=1.0))
part.append(chord.Chord(["E4", "G4"], quarterLength=1.0))
score.append(part)
"""

ABC = """X:1
T:Test tune
M:4/4
L:1/4
K:C
CDEF|GABc|
"""


def test_render_codegen_version(tmp_path):
    result = render_codegen_version(tmp_path, CODE, "Test", "initial version")
    assert result["ok"], result
    assert result["version"] == 1
    assert "piece.musicxml" in result["files"]
    assert "source.py" in result["files"]
    assert (tmp_path / "v1" / "meta.json").exists()
    # versions increment
    again = render_codegen_version(tmp_path, CODE, "Test 2", "again")
    assert again["version"] == 2


def test_render_codegen_error_leaves_no_version(tmp_path):
    result = render_codegen_version(tmp_path, "score = undefined_name", "Bad", "n")
    assert not result["ok"] and result["error"]
    assert not (tmp_path / "v1").exists()
    ok = render_codegen_version(tmp_path, CODE, "Good", "n")
    assert ok["version"] == 1


def test_render_abc_version(tmp_path):
    result = render_abc_version(tmp_path, ABC, "Tune", "initial version")
    assert result["ok"], result
    stored = (tmp_path / "v1" / "piece.abc").read_text(encoding="utf-8")
    assert stored.startswith("X:1")  # raw ABC, not the abc2midi-normalized copy
    bad = render_abc_version(tmp_path, "not abc at all", "Bad", "n")
    assert not bad["ok"]


# -- HTTP surface -----------------------------------------------------------------


@pytest.fixture()
def client(studio_env):
    from llm_music.studio.app import app

    return TestClient(app)


def login(client):
    res = client.post("/api/login", json={"password": "opensesame"})
    assert res.status_code == 200


def test_api_requires_auth(client):
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/sessions").status_code == 401
    assert client.post("/api/sessions", json={}).status_code == 401


def test_login_flow(client):
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
    login(client)
    me = client.get("/api/me").json()
    assert me["default_model"] in me["models"]


def test_session_endpoints(client):
    login(client)
    meta = client.post("/api/sessions", json={"title": "Trio"}).json()
    assert meta["title"] == "Trio"
    got = client.get(f"/api/sessions/{meta['id']}").json()
    assert got["meta"]["id"] == meta["id"] and got["events"] == []
    assert client.get("/api/sessions").json()["sessions"][0]["id"] == meta["id"]
    assert client.post("/api/sessions", json={"model": "nope"}).status_code == 400
    assert client.get("/api/sessions/20990101-000000-abcdef").status_code == 404
    assert (
        client.get(f"/api/sessions/{meta['id']}/pieces/v1/piece.mp3").status_code == 404
    )


def test_message_mode_validation(client):
    login(client)
    meta = client.post("/api/sessions", json={"title": "Modes"}).json()
    res = client.post(f"/api/sessions/{meta['id']}/message",
                      json={"text": "hi", "mode": "verovio"})
    assert res.status_code == 400


def test_tools_by_mode_offers_exactly_one_tool():
    from llm_music.studio.agent import TOOLS_BY_MODE

    assert [t["name"] for t in TOOLS_BY_MODE["codegen"]] == ["render_music21"]
    assert [t["name"] for t in TOOLS_BY_MODE["abc"]] == ["render_abc"]


def test_message_model_validation(client):
    login(client)
    meta = client.post("/api/sessions", json={"title": "Models"}).json()
    res = client.post(f"/api/sessions/{meta['id']}/message",
                      json={"text": "hi", "model": "not-a-model"})
    assert res.status_code == 400


def test_strip_thinking_enables_model_switch():
    from llm_music.studio.agent import _strip_thinking

    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "...", "signature": "sig-from-other-model"},
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "t1", "name": "render_abc", "input": {}},
        ]},
        {"role": "assistant", "content": [
            {"type": "redacted_thinking", "data": "..."},
        ]},
    ]
    stripped = _strip_thinking(history)
    assert stripped[0] == history[0]
    kinds = [b["type"] for b in stripped[1]["content"]]
    assert kinds == ["text", "tool_use"]
    assert len(stripped) == 2  # thinking-only message dropped entirely


def test_index_serves_shell(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Composer Studio" in res.text


def test_available_models_span_providers(studio_env):
    models = cfg.available_models()
    for m in ("fable-5", "gpt-5.5", "gemini-2.5-pro"):  # one per provider
        assert m in models
    assert cfg.default_model() == "opus-4.8"  # snappy default; thinking models opt-in


def test_openai_transcript_conversion():
    from llm_music.studio.backends import to_openai_messages, to_openai_tools
    from llm_music.studio.agent import TOOLS

    canonical = [
        {"role": "user", "content": "write me a jig"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Here it comes."},
            {"type": "tool_use", "id": "toolu_1", "name": "render_abc",
             "input": {"abc": "X:1", "title": "Jig", "note": "initial version"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": '{"ok": true, "version": 1}', "is_error": False},
        ]},
    ]
    msgs = to_openai_messages("be musical", canonical)
    assert msgs[0] == {"role": "system", "content": "be musical"}
    assert msgs[1] == {"role": "user", "content": "write me a jig"}
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"] == "Here it comes."
    call = msgs[2]["tool_calls"][0]
    assert call["id"] == "toolu_1" and call["function"]["name"] == "render_abc"
    assert '"title": "Jig"' in call["function"]["arguments"]
    assert msgs[3] == {"role": "tool", "tool_call_id": "toolu_1",
                       "content": '{"ok": true, "version": 1}'}

    fns = to_openai_tools(TOOLS)
    assert [f["function"]["name"] for f in fns] == ["render_music21", "render_abc"]
    assert fns[0]["function"]["parameters"]["required"] == ["code", "title", "note"]


def test_json_protocol_transcript_conversion():
    from llm_music.studio.agent import TOOLS_BY_MODE
    from llm_music.studio.backends import _to_json_messages

    tools = TOOLS_BY_MODE["abc"]
    canonical = [
        {"role": "user", "content": "write me a jig"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "render_abc",
             "input": {"abc": "X:1", "title": "Jig", "note": "initial version"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": '{"ok": true, "version": 1}', "is_error": False},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "Take a listen!"}]},
    ]
    msgs = _to_json_messages("be musical", tools, canonical)
    assert msgs[0]["role"] == "system" and "Response protocol" in msgs[0]["content"]
    assert "render_abc" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "write me a jig"}
    assert msgs[2]["role"] == "assistant" and '"tool": "render_abc"' in msgs[2]["content"]
    assert msgs[3]["role"] == "user" and msgs[3]["content"].startswith("RENDER RESULT:")
    assert msgs[4] == {"role": "assistant", "content": "Take a listen!"}
