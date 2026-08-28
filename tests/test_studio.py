"""Studio tests: auth tokens, session store, render executors, HTTP surface.

No network: the agent loop itself (Anthropic calls) is exercised only up to its
guard clauses; tool executors run the real sandbox/renderers locally.
"""

import json

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


def test_sanitize_env_strips_pasted_newlines(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test\n")
    monkeypatch.setenv("STUDIO_PASSWORD", " secret ")
    cfg.sanitize_env()
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert os.environ["STUDIO_PASSWORD"] == "secret"


def test_redact_scrubs_secret_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret123")
    msg = "Illegal header value b'sk-ant-supersecret123\\n'"
    out = cfg.redact(msg)
    assert "supersecret" not in out and "[ANTHROPIC_API_KEY]" in out


def test_sandbox_cannot_read_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-leakcheck")
    code = """
import os
raise RuntimeError("env says: " + os.environ.get("ANTHROPIC_API_KEY", "MISSING"))
"""
    result = render_codegen_version(tmp_path, code, "Leak", "n")
    assert not result["ok"]
    assert "leakcheck" not in result["error"]
    assert "MISSING" in result["error"]


# -- listening sessions -----------------------------------------------------------


BATCH_NAME = "20260819_000000__models_3_prompts_1"
CODEGEN_BATCH = "20260820_000000__models_3_prompts_1"
BATCH_MODELS = ("fable-5", "gpt-5.6", "gemini-3.7-flash")


@pytest.fixture()
def batch_root(studio_env, monkeypatch):
    root = studio_env / "batches"
    batch = root / BATCH_NAME
    (batch / "audio" / "express-yourself").mkdir(parents=True)
    pieces = []
    for model in BATCH_MODELS:
        for s in range(2):
            rel = f"audio/express-yourself/{model}{'_s' + str(s) if s else ''}.mp3"
            (batch / rel).write_bytes(b"ID3-fake-" + model.encode())
            pieces.append({
                "model": model, "prompt": "express-yourself",
                "prompt_label": "Express yourself", "mode": "abc", "sample": s,
                "ok": True, "title": f"Tune {s}",
                "abc": f"X:1\nT:Tune {s}\nM:4/4\nL:1/4\nK:C\nCDEF|GABc|\n",
                "audio": rel,
            })
    pieces.append({"model": "fable-5", "prompt": "express-yourself",
                   "prompt_label": "Express yourself", "mode": "abc",
                   "sample": 2, "ok": False, "error": "render failed"})
    (batch / "data.json").write_text(json.dumps({
        "timestamp": "20260819_000000", "models": list(BATCH_MODELS),
        "prompts": ["express-yourself"], "pieces": pieces}), encoding="utf-8")

    # A second, codegen-mode batch (two prompts) for cross-batch sessions.
    cg = root / CODEGEN_BATCH
    cg_pieces = []
    for prompt in ("express-yourself", "uniquely-you"):
        (cg / "audio" / prompt).mkdir(parents=True)
        (cg / "scores" / prompt).mkdir(parents=True)
        for model in BATCH_MODELS:
            audio_rel = f"audio/{prompt}/{model}.mp3"
            score_rel = f"scores/{prompt}/{model}.musicxml"
            (cg / audio_rel).write_bytes(b"ID3-code-" + model.encode())
            (cg / score_rel).write_text("<score-partwise/>", encoding="utf-8")
            cg_pieces.append({
                "model": model, "prompt": prompt,
                "prompt_label": prompt.replace("-", " ").capitalize(),
                "mode": "codegen", "sample": 0, "ok": True,
                "title": f"Code piece {model}", "code": "print('x')",
                "audio": audio_rel, "score": score_rel,
            })
    (cg / "data.json").write_text(json.dumps({
        "timestamp": "20260820_000000", "models": list(BATCH_MODELS),
        "prompts": ["express-yourself", "uniquely-you"], "pieces": cg_pieces}),
        encoding="utf-8")
    monkeypatch.setenv("STUDIO_BATCHES_DIR", str(root))
    return root


def test_review_batches_endpoint(client, batch_root):
    assert client.get("/api/review/batches").status_code == 401
    login(client)
    batches = client.get("/api/review/batches").json()["batches"]
    assert [b["name"] for b in batches] == [CODEGEN_BATCH, BATCH_NAME]  # newest first
    # the failed piece is not offered
    assert batches[1]["model_counts"] == {m: 2 for m in BATCH_MODELS}
    assert batches[1]["modes"] == ["abc"]
    assert batches[0]["modes"] == ["codegen"]


def test_review_blind_flow(client, batch_root):
    login(client)
    res = client.post("/api/reviews", json={
        "batches": [BATCH_NAME], "models": list(BATCH_MODELS),
        "per_cell": 2, "blind": True, "seed": 7})
    assert res.status_code == 200, res.text
    r = res.json()
    sid = r["meta"]["id"]
    assert r["meta"]["kind"] == "review" and r["revealed"] is False
    assert len(r["pieces"]) == 6
    assert {p["group"] for p in r["pieces"]} == {"Model A", "Model B", "Model C"}
    for p in r["pieces"]:
        assert "model" not in p  # blind: identity never crosses the wire
    # files serve by queue index, so URLs leak nothing either
    audio = client.get(f"/api/reviews/{sid}/pieces/0/audio")
    assert audio.status_code == 200 and audio.content.startswith(b"ID3")
    abc = client.get(f"/api/reviews/{sid}/pieces/0/abc")
    assert abc.status_code == 200 and abc.text.startswith("X:1")
    assert client.get(f"/api/reviews/{sid}/pieces/99/audio").status_code == 404
    assert client.get(f"/api/reviews/{sid}/pieces/0/score").status_code == 404
    assert client.get(f"/api/reviews/{sid}/pieces/0/source").status_code == 404
    # notes need a listener name, and are stamped with the blind state at write time
    assert client.post(f"/api/reviews/{sid}/notes",
                       json={"text": "anonymous", "piece": 0}).status_code == 400
    assert client.post(f"/api/reviews/{sid}/notes",
                       json={"text": "lovely voice-leading", "piece": 0,
                             "user": "Caio"}).status_code == 200
    assert client.post(f"/api/reviews/{sid}/notes",
                       json={"text": "Model B repeats itself", "user": "Caio"}).status_code == 200
    got = client.get(f"/api/reviews/{sid}", params={"user": "Caio"}).json()
    assert [n["piece"] for n in got["notes"]] == [0, None]
    assert all(n["revealed"] is False for n in got["notes"])
    # reveal is logged per listener; later notes are marked post-reveal
    assert client.post(f"/api/reviews/{sid}/reveal", json={}).status_code == 400
    rev = client.post(f"/api/reviews/{sid}/reveal", json={"user": "Caio"}).json()
    assert all("model" in p for p in rev["pieces"])
    client.post(f"/api/reviews/{sid}/notes",
                json={"text": "so that was fable", "piece": 0, "user": "Caio"})
    got = client.get(f"/api/reviews/{sid}", params={"user": "Caio"}).json()
    assert got["revealed"] is True
    assert set(got["groups"].values()) == set(BATCH_MODELS)
    assert got["notes"][-1]["revealed"] is True
    by_model = {}
    for p in got["pieces"]:
        by_model.setdefault(p["model"], set()).add(p["group"])
    assert all(len(groups) == 1 for groups in by_model.values())  # one letter per model


def test_review_multi_listener_isolation(client, batch_root):
    """Two listeners on the same window: notes stay private to their author,
    and one person's reveal must not unblind anyone else."""
    login(client)
    sid = client.post("/api/reviews", json={
        "batches": [BATCH_NAME], "models": list(BATCH_MODELS),
        "per_cell": 1, "blind": True, "seed": 5}).json()["meta"]["id"]
    client.post(f"/api/reviews/{sid}/notes",
                json={"text": "caio's blind note", "piece": 0, "user": "Caio"})
    client.post(f"/api/reviews/{sid}/reveal", json={"user": "Caio"})
    client.post(f"/api/reviews/{sid}/notes",
                json={"text": "sara's note", "piece": 0, "user": "Sara"})

    caio = client.get(f"/api/reviews/{sid}", params={"user": "Caio"}).json()
    sara = client.get(f"/api/reviews/{sid}", params={"user": "Sara"}).json()
    nobody = client.get(f"/api/reviews/{sid}").json()
    assert [n["text"] for n in caio["notes"]] == ["caio's blind note"]
    assert [n["text"] for n in sara["notes"]] == ["sara's note"]
    assert caio["revealed"] is True and "model" in caio["pieces"][0]
    assert sara["revealed"] is False and "model" not in sara["pieces"][0]
    assert sara["notes"][0]["revealed"] is False  # written after CAIO's reveal, still blind for Sara
    assert nobody["revealed"] is False and nobody["notes"] == []


def test_session_list_note_counts_are_per_user(client, batch_root):
    login(client)
    sid = client.post("/api/reviews", json={
        "batches": [BATCH_NAME], "models": ["fable-5"],
        "per_cell": 1, "blind": True, "seed": 2}).json()["meta"]["id"]
    client.post(f"/api/reviews/{sid}/notes",
                json={"text": "note", "piece": 0, "user": "Nathaniel"})

    def count(user=None):
        params = {"user": user} if user is not None else {}
        rows = client.get("/api/sessions", params=params).json()["sessions"]
        return next(s["n_notes"] for s in rows if s["id"] == sid)

    assert count("Nathaniel") == 1
    assert count("Caio") == 0
    assert count() == 0  # unregistered viewers see no one's progress


def test_review_list_order_is_fixed(client, batch_root):
    """Windows must not reshuffle by activity: the suite is a sequence."""
    login(client)
    ids = []
    for seed in (1, 2, 3):
        ids.append(client.post("/api/reviews", json={
            "batches": [BATCH_NAME], "models": ["fable-5"],
            "per_cell": 1, "blind": True, "seed": seed}).json()["meta"]["id"])
    order = lambda: [s["id"] for s in client.get("/api/sessions").json()["sessions"]
                     if s["kind"] == "review"]
    before = order()
    assert before == list(reversed(ids))  # newest-created first
    # activity on the oldest window must not move it
    client.post(f"/api/reviews/{ids[0]}/notes",
                json={"text": "note", "piece": 0, "user": "Caio"})
    assert order() == before


def test_user_registry(client, studio_env):
    assert client.get("/api/users").status_code == 401
    login(client)
    assert client.get("/api/users").json()["users"] == []
    r = client.post("/api/users", json={"name": "  Caio   Souza "})
    assert r.status_code == 200 and r.json()["name"] == "Caio Souza"
    # case-insensitive dedupe: the canonical (first-seen) casing comes back
    r2 = client.post("/api/users", json={"name": "caio souza"})
    assert r2.json()["name"] == "Caio Souza" and r2.json()["users"] == ["Caio Souza"]
    assert client.post("/api/users", json={"name": "   "}).status_code == 400


def test_review_cross_batch_stratified(client, batch_root):
    """One piece per (model x prompt x method) across an abc and a codegen
    batch — the design for comparing writing methods on the same brief."""
    login(client)
    res = client.post("/api/reviews", json={
        "batches": [BATCH_NAME, CODEGEN_BATCH], "models": list(BATCH_MODELS),
        "per_cell": 1, "blind": True, "seed": 3})
    assert res.status_code == 200, res.text
    r = res.json()
    sid = r["meta"]["id"]
    # cells: abc batch 3 models x 1 prompt, codegen batch 3 models x 2 prompts
    assert len(r["pieces"]) == 9
    combos = {(p["group"], p["prompt_label"], p["mode"]) for p in r["pieces"]}
    assert len(combos) == 9  # every cell exactly once
    # pieces arrive grouped by prompt: same brief back to back
    labels = [p["prompt_label"] for p in r["pieces"]]
    assert labels == sorted(labels, key=lambda x: labels.index(x))
    assert len([x for x in set(labels)]) == 2
    # files resolve to each piece's own batch
    for p in r["pieces"]:
        if p["mode"] == "codegen":
            got = client.get(f"/api/reviews/{sid}/pieces/{p['idx']}/score")
            assert got.status_code == 200 and "score-partwise" in got.text
            audio = client.get(f"/api/reviews/{sid}/pieces/{p['idx']}/audio")
            assert audio.content.startswith(b"ID3-code-")
        else:
            audio = client.get(f"/api/reviews/{sid}/pieces/{p['idx']}/audio")
            assert audio.content.startswith(b"ID3-fake-")


def test_review_prompt_filter(client, batch_root):
    login(client)
    r = client.post("/api/reviews", json={
        "batches": [CODEGEN_BATCH], "models": list(BATCH_MODELS),
        "prompts": ["uniquely-you"], "per_cell": 1, "blind": False, "seed": 1})
    assert r.status_code == 200, r.text
    pieces = r.json()["pieces"]
    assert len(pieces) == 3
    assert {p["prompt_label"] for p in pieces} == {"Uniquely you"}


def test_review_seed_reproducible_and_unblind(client, batch_root):
    login(client)
    body = {"batches": [BATCH_NAME], "models": ["fable-5", "gpt-5.6"],
            "per_cell": 1, "blind": False, "seed": 42}
    a = client.post("/api/reviews", json=body).json()
    b = client.post("/api/reviews", json=body).json()
    assert a["revealed"] and b["revealed"]  # not blind: names shown from the start
    assert [(p["model"], p["sample"]) for p in a["pieces"]] == \
           [(p["model"], p["sample"]) for p in b["pieces"]]


def test_review_validation(client, batch_root):
    login(client)
    post = lambda body: client.post("/api/reviews", json=body).status_code
    assert post({"batches": ["nope"], "models": ["fable-5"]}) == 404
    assert post({"batches": [], "models": ["fable-5"]}) == 400
    assert post({"batches": [BATCH_NAME], "models": []}) == 400
    assert post({"batches": [BATCH_NAME], "models": ["kimi-k3"]}) == 400  # no ok pieces
    assert post({"batches": [BATCH_NAME], "models": ["fable-5"], "per_cell": 0}) == 400
    assert client.get("/api/reviews/20990101-000000-abcdef").status_code == 404
    # a chat session is not a listening session
    chat = client.post("/api/sessions", json={"title": "Chat"}).json()
    assert client.get(f"/api/reviews/{chat['id']}").status_code == 404


def test_resolve_batch_file_stays_inside_batch(tmp_path):
    from llm_music.studio import review as review_mod

    batch = tmp_path / BATCH_NAME
    (batch / "audio").mkdir(parents=True)
    (batch / "data.json").write_text("{}", encoding="utf-8")
    (batch / "audio" / "x.mp3").write_bytes(b"a")
    assert review_mod.resolve_batch_file(tmp_path, BATCH_NAME, "audio/x.mp3").name == "x.mp3"
    with pytest.raises(KeyError):
        review_mod.resolve_batch_file(tmp_path, BATCH_NAME, "../outside.mp3")
    with pytest.raises(KeyError):
        review_mod.resolve_batch_file(tmp_path, BATCH_NAME, "data.json")  # not a piece asset
    with pytest.raises(KeyError):
        review_mod.resolve_batch_file(tmp_path, "evil/../..", "audio/x.mp3")


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
