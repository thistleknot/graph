"""Pins tools/section_corpus.index_text and its use in section_sparse.build (arm C: the heading is left out of what is indexed, kept for display).

Spec: approved plan C:\\Users\\user\\.claude\\plans\\jolly-soaring-moler.md, task T166. Synthetic records, no disk.

Run:  pytest tests/test_section_corpus.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import section_corpus as sc
import section_sparse as ss


def test_index_text_drops_the_heading_line_and_leaves_the_record_alone():
    rec = {"text": "## 3.2 Related Work\n\nPrior systems used retrieval.\n\nSecond paragraph."}
    assert sc.index_text(rec) == "Prior systems used retrieval.\n\nSecond paragraph."
    assert rec["text"].startswith("## 3.2 Related Work")                              # the record still carries its heading for display


def test_index_text_returns_a_text_without_a_heading_whole_and_survives_a_heading_only_text():
    assert sc.index_text({"text": "plain text with no heading"}) == "plain text with no heading"
    assert sc.index_text({"text": "no newline at all"}) == "no newline at all"
    assert sc.index_text({"text": "# Only a heading"}) == "# Only a heading"          # nothing after the heading: returned whole, not emptied
    assert sc.index_text({"text": "# H\n\n  body"}).strip() == "body"


def _topic_world():
    """24 sections: topic = i % 4 (six sections share each topic's words), heading = i % 3 (eight sections share each heading word). No word is in more
    than half the sections, so the corpus's whale rule leaves every column in."""
    topics = [["quantum", "lattice", "fermion"], ["protein", "enzyme", "ribosome"], ["market", "equity", "bond"], ["glacier", "tundra", "permafrost"]]
    heads = ["zebrafish", "aardvark", "platypus"]
    return [{"text": "## %s\n\n%s" % (heads[i % 3], " ".join(topics[i % 4] * 8 + ["filler%s" % "abcdefghijklmnopqrstuvwx"[i]]))} for i in range(24)]


def test_a_heading_only_word_enters_the_vocabulary_unless_it_is_stripped():
    recs = _topic_world()
    whole, info_whole = ss.build(recs, merges=None, lam=0.0, strip_heading=False)
    cut, info_cut = ss.build(recs, merges=None, lam=0.0, strip_heading=True)
    assert info_whole["U"] == info_cut["U"] + 3                                        # 'zebrafish', 'aardvark', 'platypus' were columns only when the heading was kept
    sim_whole = (whole @ whole.T).toarray()
    sim_cut = (cut @ cut.T).toarray()
    # sections 0 and 4 share a topic and have different headings; sections 0 and 3 share a heading and have different topics
    assert sim_cut[0, 4] > sim_whole[0, 4] > 0                                         # same topic, different heading: more alike once the heading is left out
    assert sim_cut[0, 3] < sim_whole[0, 3]                                             # same heading, different topic: less alike once it is left out


def test_strip_heading_is_off_by_default_so_the_baseline_build_is_unchanged():
    recs = [{"text": "## Heading%d\n\n%s" % (i % 2, " ".join(["token%d" % (j % 5) for j in range(40)]))} for i in range(6)]
    a, _ = ss.build(recs, merges=None, lam=1.0)
    b, _ = ss.build(recs, merges=None, lam=1.0, strip_heading=False)
    assert (a != b).nnz == 0 and a.shape == b.shape
