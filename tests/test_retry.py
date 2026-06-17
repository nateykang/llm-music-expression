from llm_music.generate import _is_retryable


class _ApiError(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


def test_permanent_errors_not_retryable():
    for status in (400, 401, 403, 404):
        assert _is_retryable(_ApiError(status)) is False


def test_transient_errors_retryable():
    for status in (408, 409, 429, 500, 502, 503):
        assert _is_retryable(_ApiError(status)) is True


def test_no_status_is_retryable():
    # bare network/transport error -> worth retrying
    assert _is_retryable(RuntimeError("connection reset")) is True
