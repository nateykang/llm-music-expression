"""Shared report framework: table/pane/page building blocks."""

import pytest

from llm_music.report_common import (MODE_TOGGLE, cell, dfn, fnote, fnum,
                                     mode_filter, page, paned, table, toggle)


def test_fnum_parses_and_rejects():
    assert fnum("1.5") == 1.5
    assert fnum(2) == 2.0
    assert fnum("") is None
    assert fnum(None) is None
    assert fnum("abc") is None
    assert fnum(float("nan")) is None


def test_cell_formats():
    assert cell(None, "f2") == "—"
    assert cell("a<b", "text") == "a&lt;b"
    assert cell(0.5, "pct") == "50%"
    assert cell(3.14159, "f1") == "3.1"
    assert cell(7, "int") == "7"


def test_table_structure_and_tooltips():
    h = table([("model", None), ("score", "how good")], [["m1", "1.0"], (["ref-row", "2.0"], "ref")])
    assert "<th>model</th>" in h                     # tooltip-less header is plain
    assert 'data-tip="how good"' in h
    assert "<thead>" in h and "<tbody>" in h
    assert "class='sortable'" in h
    assert "<tr class='ref'>" in h
    assert "<td class='m'>m1</td>" in h              # first cell left-aligned
    assert "<td class=''>1.0</td>" in h


def test_mode_filter_groups_abc_variants():
    items = [{"mode": "abc"}, {"mode": "smt-abc"}, {"mode": "codegen"}]
    assert len(mode_filter(items, "abc")) == 2
    assert len(mode_filter(items, "code")) == 1
    assert len(mode_filter(items, "all")) == 3


def test_paned_marks_default_visible():
    h = paned(lambda m: f"[{m}]", MODE_TOGGLE, default="all")
    assert "data-mode='all'>[all]" in h
    assert "data-mode='abc' hidden>[abc]" in h


def test_footnotes_numbered_by_first_appearance():
    body = f"<p>A{fnote('russell')} B{fnote('mert')} A again{fnote('russell')}</p>"
    doc = page("T", "judge.html", body)
    assert "href='#fn-russell'>1<" in doc and doc.count(">1</a>") == 2  # reuse, same number
    assert "href='#fn-mert'>2<" in doc
    assert "Footnotes &amp; references" in doc
    assert "<li id='fn-russell'>" in doc and "<li id='fn-mert'>" in doc
    assert "fn-muspy" not in doc                     # uncited entries stay off the page
    assert "Footnotes" not in page("T", "judge.html", "<p>no markers</p>")


def test_fnote_rejects_unknown_key():
    with pytest.raises(KeyError):
        fnote("nope")


def test_dfn_renders_tooltip_term():
    h = dfn("nPVI", 'rhythmic "contrast" measure')
    assert h.startswith("<span class='tip'")
    assert "data-tip=\"rhythmic &quot;contrast&quot; measure\"" in h
    assert ">nPVI</span>" in h


def test_toggle_and_page_skeleton():
    body = f"{toggle(MODE_TOGGLE, default='all')}<p>hello</p>"
    doc = page("My Title", "judge.html", body, extra_css=".x { color: red; }")
    assert doc.count("<title>My Title</title>") == 1
    assert 'href="judge.html" class="active"' in doc
    assert 'href="results.html">' in doc             # other tabs present, not active
    assert 'link rel="stylesheet" href="style.css' in doc
    assert doc.count("function makeSortable") == 1   # shared JS embedded exactly once
    assert ".x { color: red; }" in doc
    assert "data-default='all'" in doc
    assert "var(--bg)" in doc                        # palette via CSS variables
