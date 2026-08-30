"""Pins chunkgraph R17: DOCUMENT-level atoms sized by the corpus.

derive_chunk_params counts lines per document, Box-Cox, m = median,
hi = m + 2*MAD. A document at or under hi is one chunk; over hi it splits at
paragraph boundaries into windows of at most hi lines, a short tail merges
back, no overlap, conservation exact. Pure unit tests, a battery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunkgraph import _chunk, _paras, derive_chunk_params


def _para(n_lines, tag="s", words=6):
    return "\n".join(" ".join(f"{tag}{i}w{k}" for k in range(words)) + " ." for i in range(n_lines))


def _doc(sizes, tag="p"):
    return "\n\n".join(_para(n, f"{tag}{j}_") for j, n in enumerate(sizes))


def _lines(text):
    return [l for p in _paras(text) for l in p]


# ---------------------------------------------------------- derive params


def test_params_count_lines_per_document():
    docs = [_doc([3, 4, 3]) for _ in range(7)] + [_doc([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])]
    pr = derive_chunk_params(docs)
    assert pr["unit"] == "lines"
    assert pr["n"] == 8
    assert 9 <= pr["m"] <= 12, pr                  # the bulk is 10-line docs
    assert pr["hi"] >= pr["m"]
    assert 0 <= pr["hi_frac"] <= 1 / 8 + 1e-9


def test_params_fall_back_to_chars_for_single_line_documents():
    docs = ["x " * (50 + 13 * i) for i in range(10)]     # no newlines anywhere
    pr = derive_chunk_params(docs)
    assert pr["unit"] == "chars" and pr["m"] >= 1 and pr["hi"] >= pr["m"]


def test_params_survive_tiny_and_empty_corpora():
    assert derive_chunk_params([_doc([2, 2])])["m"] >= 1
    assert derive_chunk_params(["", "   "])["m"] >= 1


# ---------------------------------------------------------- one document


def test_document_at_or_under_hi_is_one_chunk_verbatim_structure():
    doc = _doc([3, 4, 3])                                    # 10 lines
    out = _chunk(doc, m=8, hi=10)
    assert len(out) == 1
    assert out[0].count("\n\n") == 2, "paragraph breaks preserved"
    assert _lines(out[0]) == _lines(doc)


def test_document_over_hi_splits_at_paragraph_boundaries():
    doc = _doc([5, 5, 5, 5, 5, 5])                           # 30 lines
    out = _chunk(doc, m=8, hi=12)
    sizes = [len(_lines(c)) for c in out]
    assert sizes == [10, 10, 10], sizes                      # never cuts a paragraph
    assert all(c.count("\n\n") == 1 for c in out)


def test_short_tail_merges_back_into_previous_window():
    doc = _doc([6, 6, 6, 2])                                 # 20 lines
    out = _chunk(doc, m=5, hi=12)
    sizes = [len(_lines(c)) for c in out]
    assert sizes == [12, 8], sizes                           # 2-line tail (< m) merged


def test_tail_at_or_above_m_stays_separate():
    doc = _doc([6, 6, 6, 6])                                 # 24 lines
    out = _chunk(doc, m=5, hi=12)
    assert [len(_lines(c)) for c in out] == [12, 12]


def test_oversize_paragraph_is_cut_on_lines_never_words():
    doc = _doc([2, 30, 2])
    out = _chunk(doc, m=5, hi=12)
    words = set(doc.split())
    for c in out:
        assert len(_lines(c)) <= 12
        assert set(c.split()) <= words
    assert _lines("\n\n".join(out)) == _lines(doc)


def test_no_overlap_between_windows():
    doc = _doc([7] * 8)
    seen = set()
    for c in _chunk(doc, m=10, hi=20):
        ls = set(_lines(c))
        assert not (ls & seen)
        seen |= ls


# ---------------------------------------------------------- conservation


@pytest.mark.parametrize("sizes", [[3, 3], [40], [1] * 30, [9, 1, 9, 1, 9], [12, 12, 12, 3], [2, 50, 2]])
@pytest.mark.parametrize("m,hi", [(5, 12), (10, 20), (100, 170)])
def test_every_source_line_lands_in_exactly_one_chunk(sizes, m, hi):
    doc = _doc(sizes)
    out = _chunk(doc, m=m, hi=hi)
    assert [l for c in out for l in _lines(c)] == _lines(doc)
    assert all(c.strip() == c and c for c in out)


def test_chars_unit_cuts_on_lines_and_never_cuts_an_overlong_line():
    doc = "\n".join(['a' * 40, 'b' * 40, 'y' * 500, 'c' * 40])
    out = _chunk(doc, m=60, hi=200, unit='chars')
    got = [l for c in out for l in _lines(c)]
    assert got == doc.split("\n")                     # conservation
    assert 'y' * 500 in got, 'overlong line must survive uncut'
    assert len(out) >= 2, 'a 624-char doc over hi=200 must split'


def test_empty_and_whitespace_docs_yield_nothing():
    assert _chunk("", m=5, hi=12) == []
    assert _chunk("\n\n  \n", m=5, hi=12) == []
