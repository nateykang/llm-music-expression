"""Sandbox behaviour: the security boundary must fail loudly and informatively."""

from llm_music.sandbox import run_music21_code


def test_timeout_error_includes_captured_output(tmp_path):
    code = (
        "import time\n"
        "print('building score', flush=True)\n"
        "time.sleep(60)\n"
    )
    r = run_music21_code(code, tmp_path, timeout=3)
    assert not r.ok
    assert "timed out after 3s" in r.error
    assert "building score" in r.error


def test_missing_score_reports_clear_error(tmp_path):
    r = run_music21_code("x = 1\n", tmp_path, timeout=30)
    assert not r.ok
    assert "score" in r.error


def test_exception_in_code_surfaces_traceback(tmp_path):
    r = run_music21_code("raise ValueError('boom')\n", tmp_path, timeout=30)
    assert not r.ok
    assert "boom" in r.error
