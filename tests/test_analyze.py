"""Feature extraction: the tempo-defaulted flag must reflect what the model
actually wrote, not what the pipeline's fallbacks produced."""

from llm_music.analyze import _abc_declares_tempo

HEADER = "X:1\nT:t\nM:4/4\nL:1/4\n"
BODY = "CDEF|GABc|\n"


def test_q_before_k_counts_as_declared():
    assert _abc_declares_tempo(HEADER + "Q:1/4=60\nK:C\n" + BODY)


def test_no_q_is_defaulted():
    assert not _abc_declares_tempo(HEADER + "K:C\n" + BODY)


def test_q_after_k_is_defaulted():
    # K: terminates the ABC header; abc2midi ignores a later Q: and bakes its
    # 120 default into the MIDI, so this must be caught on the raw text.
    assert not _abc_declares_tempo(HEADER + "K:C\nQ:1/4=60\n" + BODY)


def test_indented_and_lowercase_headers_recognized():
    assert _abc_declares_tempo(HEADER + "  q:80\nK:C\n" + BODY)
