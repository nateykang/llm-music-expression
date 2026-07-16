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


def test_part_level_tempo_survives_musicxml_export(tmp_path):
    # The toolkit tells models to insert tempo into the Part at offset 0, but
    # music21's MusicXML export drops a MetronomeMark that isn't inside a
    # Measure — and MIDI is derived from the MusicXML, so the tempo was lost
    # end to end (then measured as the 120 fallback). The runner must relocate
    # such marks so the documented placement actually works.
    code = (
        "from music21 import stream, note, tempo\n"
        "score = stream.Score()\n"
        "part = stream.Part()\n"
        "m = stream.Measure(number=1)\n"
        "m.append(note.Note('C4', quarterLength=4))\n"
        "part.append(m)\n"
        "part.insert(0, tempo.MetronomeMark(number=72))\n"
        "score.append(part)\n"
    )
    r = run_music21_code(code, tmp_path, timeout=60)
    assert r.ok, r.error
    from music21 import converter, tempo

    reparsed = converter.parse(str(r.musicxml_path))
    marks = [m.number for m in reparsed.recurse().getElementsByClass(tempo.MetronomeMark)]
    assert marks == [72]


def test_score_level_mid_piece_tempo_lands_in_right_measure(tmp_path):
    code = (
        "from music21 import stream, note, tempo\n"
        "score = stream.Score()\n"
        "part = stream.Part()\n"
        "for i in range(2):\n"
        "    m = stream.Measure(number=i + 1)\n"
        "    m.append(note.Note('C4', quarterLength=4))\n"
        "    part.append(m)\n"
        "score.append(part)\n"
        "score.insert(0, tempo.MetronomeMark(number=60))\n"
        "score.insert(4, tempo.MetronomeMark(number=100))\n"
    )
    r = run_music21_code(code, tmp_path, timeout=60)
    assert r.ok, r.error
    from music21 import converter, tempo

    reparsed = converter.parse(str(r.musicxml_path))
    marks = [(m.getOffsetInHierarchy(reparsed), m.number)
             for m in reparsed.recurse().getElementsByClass(tempo.MetronomeMark)]
    assert sorted(marks) == [(0.0, 60), (4.0, 100)]
