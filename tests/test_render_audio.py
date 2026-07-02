"""Audio-baking failure paths must be visible, not silent."""

import logging
import subprocess

import llm_music.render as render


def _reset_warn_once():
    render._warn_once.cache_clear()


def test_abc_to_midi_missing_tool_logs_once(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    _reset_warn_once()
    with caplog.at_level(logging.WARNING, logger="llm_music.render"):
        assert render.abc_to_midi("X:1\nK:C\nCDEF|\n", tmp_path) is None
        assert render.abc_to_midi("X:1\nK:C\nGABc|\n", tmp_path) is None
    hits = [r for r in caplog.records if "abc2midi" in r.message]
    assert len(hits) == 1  # warned, but only once per message


def test_midi_to_audio_missing_tools_logs_reason(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    _reset_warn_once()
    with caplog.at_level(logging.WARNING, logger="llm_music.render"):
        assert render.midi_to_audio(tmp_path / "x.mid", tmp_path / "x.mp3") is False
    assert any("fluidsynth" in r.message for r in caplog.records)


def test_midi_to_audio_surfaces_fluidsynth_stderr(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(render.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(render, "find_soundfont", lambda: tmp_path / "sf.sf2")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="synth exploded")

    monkeypatch.setattr(render.subprocess, "run", fake_run)
    with caplog.at_level(logging.WARNING, logger="llm_music.render"):
        assert render.midi_to_audio(tmp_path / "x.mid", tmp_path / "x.mp3") is False
    assert any("synth exploded" in r.message for r in caplog.records)
