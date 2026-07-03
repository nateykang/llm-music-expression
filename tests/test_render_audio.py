"""Audio-baking failure paths must be visible, not silent."""

import logging
import subprocess

import llm_music.render as render


def _reset_warn_once():
    render._warn_once.cache_clear()


def test_prepare_abc_gchords_switch():
    abc = 'X:1\nM:4/4\nK:C\n"Em" C D E F |\n'
    on = render._prepare_abc_for_audio(abc, gchords=True)
    off = render._prepare_abc_for_audio(abc, gchords=False)
    assert "%%MIDI gchordoff" not in on
    # the off switch lands right after the K: header line
    lines = off.splitlines()
    assert lines[lines.index("K:C") + 1] == "%%MIDI gchordoff"


def test_prepare_abc_strips_blank_lines_and_fixes_voice_markers():
    abc = "X:1\nK:C\nV:RH\n[RH] C D |\n\nE F |\n"
    out = render._prepare_abc_for_audio(abc)
    assert "\n\n" not in out
    assert "[V:RH]" in out and "[RH]" not in out


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
