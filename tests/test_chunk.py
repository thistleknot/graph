"""Pins chunkgraph._chunk (R16): recursive, never inside a word.

Pure unit tests -- no database, no model. Three varied shapes plus the
boundary cases, because a single passing case says nothing about a splitter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunkgraph import _chunk


def _lines(n, words=8):
    return "\n".join(" ".join(f"s{i}w{k}" for k in range(words)) + " ." for i in range(n))


def test_short_paragraphs_pass_through_untouched():
    doc = "One short paragraph .\n\nAnother one .\n\n\n\nA third ."
    assert _chunk(doc) == ["One short paragraph .", "Another one .", "A third ."]


def test_long_paragraph_splits_on_its_lines_not_on_words():
    """The shape the Brown loader writes: sentences on lines, paragraphs on
    blank lines. A long paragraph must break BETWEEN lines."""
    doc = _lines(60)                       # 60 lines x 9 tokens = 540 words
    out = _chunk(doc, target=120, max_len=200)
    assert len(out) > 1
    for c in out:
        assert c.split()[0].startswith("s"), f"chunk starts mid-line: {c[:40]!r}"
        for line in c.split("\n"):
            assert line.endswith(" ."), f"line cut short: {line!r}"
        assert len(c.split()) <= 120 + 9   # target plus at most one line


def test_chunks_cover_every_line_exactly_once():
    doc = _lines(45)
    out = _chunk(doc, target=100, max_len=200)
    seen = [l for c in out for l in c.split("\n")]
    assert seen == doc.split("\n")


def test_single_overlong_line_falls_back_to_word_windows():
    line = " ".join(f"w{i}" for i in range(500))
    out = _chunk(line, target=120, max_len=200)
    words = set(line.split())
    assert len(out) >= 4
    for c in out:
        assert set(c.split()) <= words, "window split inside a word"
        assert len(c.split()) <= 120
    assert out[0].split()[0] == "w0" and out[-1].split()[-1] == "w499"


def test_overlong_line_inside_a_paragraph_flushes_the_buffer_first():
    doc = _lines(5) + "\n" + " ".join(f"x{i}" for i in range(300)) + "\n" + _lines(3)
    out = _chunk(doc, target=120, max_len=200)
    assert out[0].startswith("s0w0"), "buffered lines were not flushed before the window"
    assert any(c.startswith("x0 ") for c in out)
    assert out[-1].startswith("s0w0") or out[-1].split()[0].startswith("s")


def test_no_chunk_is_ever_empty_or_whitespace():
    doc = "\n\n\n" + _lines(10) + "\n\n   \n\n" + _lines(2) + "\n\n"
    for c in _chunk(doc):
        assert c.strip() == c and c


@pytest.mark.parametrize("target,max_len", [(40, 80), (120, 200), (300, 500)])
def test_no_chunk_exceeds_target_by_more_than_one_line(target, max_len):
    doc = _lines(80, words=6)
    for c in _chunk(doc, target=target, max_len=max_len):
        assert len(c.split()) <= target + 7
