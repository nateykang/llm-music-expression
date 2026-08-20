from llm_music.retry import is_retryable


class _ApiError(Exception):
    def __init__(self, status, msg=None):
        super().__init__(msg or f"status {status}")
        self.status_code = status


def test_permanent_errors_not_retryable():
    for status in (400, 401, 403, 404):
        assert is_retryable(_ApiError(status)) is False


def test_transient_errors_retryable():
    for status in (408, 409, 429, 500, 502, 503, 529):
        assert is_retryable(_ApiError(status)) is True


def test_no_status_is_retryable():
    # bare network/transport error -> worth retrying
    assert is_retryable(RuntimeError("connection reset")) is True


def test_permanent_status_wins_over_transient_looking_message():
    # A 400 whose message happens to mention a transient marker must NOT be
    # retried — the status is authoritative.
    assert is_retryable(_ApiError(400, "connection closed: bad request")) is False
    assert is_retryable(_ApiError(401, "invalid key, please try again")) is False


class _StatusError(Exception):
    def __init__(self, status_code=None, msg="boom"):
        super().__init__(msg)
        self.status_code = status_code


def test_overloaded_by_status():
    from llm_music.retry import is_overloaded

    assert is_overloaded(_StatusError(status_code=529))
    assert not is_overloaded(_StatusError(status_code=429))
    assert not is_overloaded(_StatusError(status_code=500))


def test_overloaded_by_streamed_error_body():
    # Streaming delivers the overload as an in-stream error event: no status
    # attribute, only the error type in the message.
    from llm_music.retry import is_overloaded

    exc = Exception("{'type': 'error', 'error': {'type': 'overloaded_error', "
                    "'message': 'Overloaded'}}")
    assert is_overloaded(exc)
    assert not is_overloaded(Exception("connection reset by peer"))


def test_overload_does_not_consume_attempts(monkeypatch, tmp_path):
    # Two overloads then a permanent error: attempts must count only the
    # permanent error, not the capacity failures.
    from llm_music import generate as gen

    calls = {"n": 0}

    class FlakyClient:
        name = "fake"

        def complete(self, system, user, json_mode=False):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise _StatusError(status_code=529, msg="Overloaded")
            raise _StatusError(status_code=400, msg="bad request")

    monkeypatch.setattr(gen, "backoff_sleep", lambda *a, **k: None)
    result = gen.generate_piece(FlakyClient(), "express-yourself", "abc", tmp_path,
                                max_attempts=5, bake_audio=False)
    assert not result.ok
    assert result.attempts == 1              # only the 400 charged an attempt
    assert len(result.failed_attempts) == 1  # overloads are not model drafts
    assert sum("not charged" in e for e in result.errors) == 2


def test_429_not_charged_in_generation(monkeypatch, tmp_path):
    from llm_music import generate as gen

    calls = {"n": 0}

    class Flaky:
        name = "fake"

        def complete(self, system, user, json_mode=False):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise _StatusError(status_code=429, msg="Error code: 429")
            raise _StatusError(status_code=400, msg="bad request")

    monkeypatch.setattr(gen, "backoff_sleep", lambda *a, **k: None)
    result = gen.generate_piece(Flaky(), "express-yourself", "abc", tmp_path,
                                max_attempts=5, bake_audio=False)
    assert result.attempts == 1
    assert len(result.failed_attempts) == 1


def test_rate_gate_trips_and_releases():
    import time
    from llm_music.retry import RateGate

    g = RateGate()
    g.trip("kimi", seconds=0.25)
    t0 = time.monotonic()
    g.wait("kimi")                      # must block ~0.25s
    assert time.monotonic() - t0 >= 0.2
    g.wait("gemini")                    # other keys unaffected: returns instantly
    assert time.monotonic() - t0 < 1.0


def test_429_cools_down_and_does_not_burn_attempts(monkeypatch, tmp_path):
    from llm_music import judge as jmod

    calls = {"n": 0}

    class Flaky:
        name = "fake-judge"

        def complete(self, system, user, json_mode=False):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise _StatusError(status_code=429, msg="Error code: 429 throttled")
            return ('{"coherence": {"reason": "ok", "score": 4}, '
                    '"emotion_label": "serene"}')

    monkeypatch.setattr(jmod, "RATE_COOLDOWN_S", 0.05)
    monkeypatch.setattr(jmod, "backoff_sleep", lambda *a, **k: None)
    monkeypatch.setattr(jmod, "representation", lambda pc, bd: ("ABC", "X:1\nK:C\nCDEF|"))
    v = jmod.judge_piece(Flaky(), {"model": "m", "prompt": "p"}, tmp_path, attempts=1)
    assert v is not None and v["coherence"]["score"] == 4.0
    assert calls["n"] == 3   # two throttled tries + one real, within attempts=1


def test_connection_error_classification():
    from llm_music.retry import is_connection_error

    class APIConnectionError(Exception):
        pass

    assert is_connection_error(APIConnectionError("Connection error."))
    assert is_connection_error(Exception("connection reset by peer"))
    assert is_connection_error(Exception("Request timed out."))
    # A real HTTP status is a provider verdict, never a transport failure —
    # even when the message mentions timeouts.
    assert not is_connection_error(_StatusError(status_code=400, msg="request timed out"))
    assert not is_connection_error(_StatusError(status_code=429, msg="Error code: 429"))
    assert not is_connection_error(Exception("could not parse JSON response"))


def test_connection_errors_not_charged_in_generation(monkeypatch, tmp_path):
    from llm_music import generate as gen

    calls = {"n": 0}

    class Flaky:
        name = "fake"

        def complete(self, system, user, json_mode=False):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise ConnectionError("Connection error.")
            raise _StatusError(status_code=400, msg="bad request")

    monkeypatch.setattr(gen, "backoff_sleep", lambda *a, **k: None)
    result = gen.generate_piece(Flaky(), "express-yourself", "abc", tmp_path,
                                max_attempts=5, bake_audio=False)
    assert result.attempts == 1              # only the 400 charged an attempt
    assert len(result.failed_attempts) == 1
    assert sum("not charged" in e for e in result.errors) == 3


def test_mid_stream_transport_failures_are_connection_errors():
    from llm_music.retry import is_connection_error

    class RemoteProtocolError(Exception):
        pass

    assert is_connection_error(RemoteProtocolError(
        "peer closed connection without sending complete message body "
        "(incomplete chunked read)"))
