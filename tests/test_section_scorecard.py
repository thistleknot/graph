"""Pins tools/section_scorecard.py: every number on a planted map whose answer is known, the heading normaliser, and the frozen rubric.

Spec: approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md, task T158. Synthetic labels, no disk beyond tmp_path.

Run:  pytest tests/test_section_scorecard.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import section_scorecard as sc


def test_codes_gives_equal_values_one_id_and_survives_one_enormous_value():
    c = sc.codes(["b", "a", "b", "c", "a"])
    assert c.tolist() == [0, 1, 0, 2, 1] and c.dtype == np.int64                      # ids in order of first appearance
    big = ["x" * 5_000] + ["short%d" % (i % 3) for i in range(30_000)]
    out = sc.codes(big)                                                                # np.unique would build a 30,001 x 5,000-character array here
    assert out[0] == 0 and out.max() == 3 and len(out) == 30_001 and out[1] == out[4]
    assert sc.codes([]).tolist() == []


def test_heading_kind_strips_numbering_and_punctuation():
    assert sc.heading_kind("3.2 Related Work") == "related work"
    assert sc.heading_kind("B.1. Training Dataset and Setup") == "training dataset and setup"
    assert sc.heading_kind("IV. Results") == "results"
    assert sc.heading_kind("Abstract") == "abstract"
    assert sc.heading_kind("4)") == "" and sc.heading_kind("   ") == ""


def _map():
    """Community 0: 12 sections of ONE paper, all headed 'Introduction'. Community 1: 12 sections from 12 papers with 12 different headings.
    Community 2: 1 singleton. Community 3: 4 sections (below size 10)."""
    lab = np.array([0] * 12 + [1] * 12 + [2] + [3] * 4)
    docs = ["p0"] * 12 + ["p%d" % i for i in range(1, 13)] + ["z"] + ["q"] * 4
    titles = ["1 Introduction"] * 12 + ["Heading %s" % "abcdefghijkl"[i] for i in range(12)] + ["x"] + ["y"] * 4
    return lab, docs, titles


def test_scorecard_numbers_on_a_planted_map():
    lab, docs, titles = _map()
    s = sc.scorecard(lab, docs, titles, seed_ari=0.9, m_edges=100)                   # bound = sqrt(100) = 10
    assert (s["sections"], s["communities"], s["size_ge_10"], s["singletons"], s["largest"]) == (29, 4, 2, 1, 12)
    assert s["oversize"] == 2 and s["oversize_share"] == pytest.approx(24 / 29)       # both 12-section communities exceed 10
    assert s["bound"] == pytest.approx(10.0)
    assert s["paper_ge_50"] == 1 and s["top_paper_median"] == pytest.approx((1.0 + 1 / 12) / 2)    # community 0 is one paper; community 1 is 12 papers
    assert s["heading_ge_50"] == 1 and s["top_heading_median"] == pytest.approx((1.0 + 1 / 12) / 2)
    assert s["seed_ari"] == 0.9


def test_a_map_with_no_community_of_size_ten_reports_zero_medians_and_does_not_crash():
    s = sc.scorecard(np.array([0, 0, 1, 2]), ["a", "b", "c", "d"], ["x", "y", "z", "w"], m_edges=10_000)
    assert s["size_ge_10"] == 0 and s["top_paper_median"] == 0.0 and s["top_heading_median"] == 0.0 and s["oversize"] == 0


def test_the_scorecard_separates_a_paper_bound_map_from_a_cross_paper_map():
    docs = ["p%d" % (i // 10) for i in range(100)]                                    # 10 papers of 10 sections
    titles = ["t%d" % i for i in range(100)]
    bound_to_paper = np.arange(100) // 10                                              # each community IS a paper
    cross_paper = np.arange(100) % 10                                                  # each community takes exactly one section from every paper
    bad = sc.scorecard(bound_to_paper, docs, titles, m_edges=100)
    good = sc.scorecard(cross_paper, docs, titles, m_edges=100)
    assert bad["paper_ge_50"] == 10 and good["paper_ge_50"] == 0
    assert bad["top_paper_median"] == 1.0 and good["top_paper_median"] == pytest.approx(0.1)


def test_one_doc_id_and_title_per_section_is_required():
    with pytest.raises(AssertionError):
        sc.scorecard(np.array([0, 1]), ["a"], ["x", "y"])


def test_the_rubric_is_written_once_and_never_overwritten(tmp_path):
    p = str(tmp_path / "rubric.md")
    assert sc.write_rubric(p) is True
    first = Path(p).read_text(encoding="utf-8")
    Path(p).write_text(first + "\nEDITED AFTER READING", encoding="utf-8")
    assert sc.write_rubric(p) is False
    assert Path(p).read_text(encoding="utf-8").endswith("EDITED AFTER READING")     # the second call left it alone
    assert "topic" in first and "genre" in first and "mixed" in first


def test_format_rows_has_one_line_per_map_and_prints_na_for_a_missing_seed_ari():
    lab, docs, titles = _map()
    text = sc.format_rows([("a_map", sc.scorecard(lab, docs, titles)), ("b_map", sc.scorecard(lab, docs, titles, 0.846))])
    lines = text.splitlines()
    assert len(lines) == 4 and lines[2].startswith("a_map") and "n/a" in lines[2] and "0.846" in lines[3]
