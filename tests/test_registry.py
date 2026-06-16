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


def test_unknown_model_raises():
    try:
        get_client("does-not-exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown model")


def test_list_models_nonempty():
    assert set(list_models()) >= {"opus-4.8", "sonnet-4.6"}
