"""Pins domain_corpora.py's D2-D5 guards, with conservation (D4) as the load-bearing one.

The packer is the only stage that can silently lose or duplicate text, and a
duplicated paragraph inflates every term's df in it -- which is the exact
quantity the term-analysis sidecar measures. So conservation is asserted as an
identity over the joined text, not as a length check.

No PDF, no database and no network: the PDF path is exercised in the module's own
diagnostic against the live corpus, because a fixture PDF would prove nothing
about real producer variation (D2).

Run:  pytest tests/test_domain_corpora.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain_corpora import (derive_pack_target, md_sections, md_units, pack,
                            units_for)


def test_pack_conserves_every_unit_exactly_once():
    """D4: concatenation round-trips. This is the guard that protects df."""
    units = ["a" * 100, "b" * 100, "c" * 100, "d" * 50]
    chunks = pack(units, target=250)
    assert "\n\n".join(chunks) == "\n\n".join(units)


def test_pack_never_splits_a_unit():
    """D4: a unit larger than the target still emerges whole."""
    big = "x" * 9000
    chunks = pack(["short", big, "tail"], target=1000)
    assert big in chunks[1]
    assert chunks[1] == big
    assert "\n\n".join(chunks) == "\n\n".join(["short", big, "tail"])


def test_pack_fills_to_target_rather_than_one_unit_per_chunk():
    """Three 100-char units under a 250 target pack 2 then 1, not 1/1/1."""
    chunks = pack(["a" * 100, "b" * 100, "c" * 100], target=250)
    assert len(chunks) == 2
    assert len(chunks[0]) == 202          # 100 + 2 (separator) + 100
    assert len(chunks[1]) == 100


def test_pack_empty_input_gives_no_chunks():
    assert pack([], target=500) == []


def test_md_units_drops_headings_and_image_placeholders():
    text = "## A Heading\n\nbody one\n\n<image 3>\n\n## Another\n\nbody two"
    assert md_units(text) == ["body one", "body two"]


def test_md_units_joins_wrapped_lines_into_one_unit():
    assert md_units("first line\nsecond line\n\nnext") == [
        "first line second line", "next"]


def test_md_sections_excludes_zero_paragraph_sections():
    """23 of Lewy's 53 headings are back-to-back; the skill excludes them."""
    text = "## One\n\n## Two\n\npara a\n\npara b\n\n## Three\n\npara c"
    secs = md_sections(text)
    assert secs == [["para a", "para b"], ["para c"]]


def test_derive_pack_target_is_the_median_chars_per_section():
    """D3: the target is fitted, and on a symmetric input it is the median."""
    doc = "".join("## S%d\n\n%s\n\n" % (i, "w" * n)
                  for i, n in enumerate([100, 200, 300, 400, 500,
                                         600, 700, 800, 900]))
    got = derive_pack_target([doc])
    assert got["n_sections"] == 9
    assert got["p50"] == 500.0
    assert got["target"] == pytest.approx(500.0, rel=0.05)


def test_derive_pack_target_raises_without_sections():
    """A constant fallback here would silently license a made-up ruler (D3)."""
    with pytest.raises(ValueError, match="no ATX sections"):
        derive_pack_target(["just prose\n\nmore prose"])


def test_units_for_pdf_text_splits_on_blank_lines_not_markdown():
    """load_neop pre-joins PDF blocks with a blank line; units_for must recover them."""
    joined = "block one\n\nblock two\n\nblock three"
    assert units_for(joined, "neop_article") == [
        "block one", "block two", "block three"]


def test_units_for_arxiv_uses_the_markdown_path():
    text = "## Title\n\nabstract text\n\n## Intro\n\nintro text"
    assert units_for(text, "arxiv") == ["abstract text", "intro text"]
