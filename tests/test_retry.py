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
