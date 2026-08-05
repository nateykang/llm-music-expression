from concurrent.futures import ThreadPoolExecutor

from llm_music.models import get_client, list_models
from llm_music.models.registry import MODEL_REGISTRY


def test_claude_models_registered():
    assert "opus-4.8" in MODEL_REGISTRY
    assert "sonnet-4.6" in MODEL_REGISTRY
    assert MODEL_REGISTRY["opus-4.8"] == ("anthropic", "claude-opus-4-8")


def test_get_client_builds_named_client():
    client = get_client("sonnet-4.6")
    assert client.name == "sonnet-4.6"
    assert hasattr(client, "complete")


def test_openai_client_builds_without_key():
    # Construction is lazy: no key needed until .complete() is called.
    client = get_client("gpt-4.1")
    assert client.name == "gpt-4.1"
    assert client.model_id == "gpt-4.1"
    assert hasattr(client, "complete")


def test_sdk_is_loaded_at_construction():
    # The SDK import must not happen on a worker thread inside .complete() —
    # concurrent lazy imports deadlock CPython's import machinery.
    for name in ("opus-4.8", "gpt-4.1", "gemini-2.5-pro"):
        assert get_client(name)._sdk is not None


def test_concurrent_client_construction():
    names = ["opus-4.8", "gpt-4.1", "gemini-2.5-pro"] * 4
    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        clients = list(ex.map(get_client, names))
    assert [c.name for c in clients] == names


def test_unknown_model_raises():
    try:
        get_client("does-not-exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown model")


def test_list_models_nonempty():
    assert set(list_models()) >= {"opus-4.8", "sonnet-4.6"}
