"""Pins src/section_query_panel.py guards P1-P2 on hand-built results. No database, no model, no network; one small matplotlib draw.

Run:  pytest tests/test_section_query_panel.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import section_query_panel as qp


def world():
    res = {"question": "how does it work", "seeds": [0, 1], "subgraph": [{"ord": 2, "weight": 2.84, "via": [0]}, {"ord": 3, "weight": 1.9, "via": [1]}],
           "candidates": [5, 7, 9], "partials": [(5, "First sentence of the answer. " * 8 + "\n\nSecond paragraph.")], "global_answer": "First sentence of the answer. " * 8 + "\n\nSecond paragraph.",
           "final": "The final LLM answer cites [arxiv/2401_1#3].\n\nA second paragraph of the answer.",
           "evidence": [{"key": "arxiv/2404_8#1", "ord": 4, "role": "typical section of a community that holds 3 of the 4 sections retrieved"},
                        {"key": "arxiv/2401_1#3", "ord": 0, "role": "retrieved, similarity 0.53"}]}
    rows = {0: {"doc_id": "arxiv/2401_1", "section_idx": 3, "section_title": "Draft then verify", "community": 5},
            1: {"doc_id": "arxiv/2401_2", "section_idx": 1, "section_title": "A" * 80, "community": 5},
            2: {"doc_id": "arxiv/2402_9", "section_idx": 7, "section_title": "Neighbour", "community": 6},
            3: {"doc_id": "arxiv/2403_4", "section_idx": 2, "section_title": "Other", "community": 7},
            4: {"doc_id": "arxiv/2404_8", "section_idx": 1, "section_title": "Typical of five", "community": 5}}
    comm = {5: {"title": "Speculative decoding", "size": 309}}
    return res, rows, comm


def test_shorten_keeps_a_fitting_title_whole_and_ends_a_long_one_with_an_ellipsis():
    assert qp.shorten("Draft then verify") == "Draft then verify"
    s = qp.shorten("A" * 80)
    assert len(s) == qp.TITLE_CHARS and s.endswith("…")


def test_column_has_the_global_view_whole_then_the_subgraph_with_keys_and_weights():
    res, rows, comm = world()
    lines = qp.column_lines(res, rows, comm)
    text = "\n".join(l for _, l in lines)
    heads = [l for s, l in lines if s == "h"]
    assert heads[0] == "GLOBAL  - LLM summary of the 1 of 3 candidate communities that answered"
    assert heads[1] == "ANSWER  - LLM over question + global summary + ▸ ◆ sections"
    assert all(len(h) <= 80 for h in heads)                                                            # a heading must fit its column (it overran into the next one at 95 characters)
    assert heads[2].startswith("SUBGRAPH  - 2 retrieved, 2 neighbours in 3 communities")
    assert heads[3].startswith("TYPICAL SECTION of each top community")
    assert "c5  Speculative decoding  (309 sections)" in text
    assert "Second paragraph." in text and text.count("First sentence of the answer.") == 8          # nothing in the global summary is cut
    assert "The final LLM answer cites [arxiv/2401_1#3]." in text and "A second paragraph of the answer." in text      # the ANSWER is shown whole, labelled
    assert "▸★ 2401_1#3  c5  Draft then verify" in text                                        # retrieved AND given to the LLM
    assert " ★ 2401_2#1  c5  " in text and "▸·" not in text                               # retrieved but not given; no neighbour was given
    assert "· 2402_9#7  c6  w2.8  Neighbour" in text
    assert "◆ 2404_8#1  c5  Typical of five" in text
    pos = [i for i, (s, _) in enumerate(lines) if s == "h"]
    assert len(pos) == 4 and pos == sorted(pos)                                                         # GLOBAL, then ANSWER, then SUBGRAPH, then TYPICAL
    assert "The final LLM answer" in lines[pos[1] + 1][1] and lines[pos[2] - 1][0] == "b"


def test_a_traversal_adds_the_hops_the_size_the_entities_and_one_group_per_later_hop_P3():
    res, rows, comm = world()
    res["traversal"] = {"hops": 3, "size": 5, "per_hop": [2, 2, 0, 1], "read": 2, "stop": "answered", "queries": 3, "entities": [("harness", 4), ("verifier", 2)],
                        "by_community": [{"cid": 5, "n": 3, "entities": [("harness", 3), ("verifier", 2)]}, {"cid": 7, "n": 2, "entities": []}]}
    res["nodes"] = [{"ord": 0, "key": "arxiv/2401_1#3", "hop": 0}, {"ord": 2, "key": "arxiv/2402_9#7", "hop": 1}, {"ord": 4, "key": "arxiv/2404_8#1", "hop": 3}]
    res["shown_communities"] = 24
    lines = qp.column_lines(res, rows, comm)
    heads = [l for s, l in lines if s == "h"]
    text = "\n".join(l for _, l in lines)
    assert heads == ["GLOBAL  - LLM summary of the 1 of 3 candidate communities that answered", "ANSWER  - LLM over question + global summary + ▸ ◆ sections",
                     "TRAVERSAL  - ReAct agent: 3 hops, 5 sections seen, 2 read", "ENTITIES BY COMMUNITY  (sections of the subgraph mentioning each)",
                     "SUBGRAPH  - 2 retrieved, 2 neighbours in 3 communities  (▸ = given to the LLM)",
                     "HOP 3  - 1 sections", "TYPICAL SECTION of each top community (◆ = given to the LLM)"] and all(len(h) <= 80 for h in heads)
    assert "sections per hop  h0 2 · h1 2 · h2 0 · h3 1" in text and "stopped: the agent chose ANSWER; 3 queries run; 24 communities shown to it" in text
    assert "entities of the subgraph (sections mentioning each): harness 4 · verifier 2" in text
    assert "▸· 2404_8#1  c5  Typical of five" in text                                                     # a hop-3 section the LLM was given
    assert "c5  Speculative decoding  3 of 5 sections" in text and "    harness 3 · verifier 2" in text      # the subgraph broken out by community, with the community's own title
    assert "c7  (no summary)  2 of 5 sections" in text and "    no entity" in text                       # a community without a title or an entity says so
    res["traversal"]["stop"], res["traversal"]["entities"] = "done", []
    assert "stopped: every planned hop and round was used" in "\n".join(l for _, l in qp.column_lines(res, rows, comm)) and "none" in "\n".join(l for _, l in qp.column_lines(res, rows, comm))


def test_a_known_paper_title_follows_the_section_heading_on_every_section_line_P4():
    res, rows, comm = world()
    res["traversal"] = {"hops": 3, "size": 5, "per_hop": [2, 2, 0, 1], "read": 0, "stop": "done", "entities": [], "by_community": []}
    res["nodes"] = [{"ord": 4, "key": "arxiv/2404_8#1", "hop": 3}]
    titles = {"arxiv/2401_1": "Speculative Decoding: A Survey of Draft-and-Verify Methods for LLM Inference", "arxiv/2404_8": "Memory for Agents", "arxiv/2402_9": "Neighbour Paper"}
    text = "\n".join(l for _, l in qp.column_lines(res, rows, comm, titles))
    assert "▸★ 2401_1#3  c5  Draft then verify | " + qp.shorten(titles["arxiv/2401_1"], qp.PAPER_CHARS) in text and len(qp.shorten(titles["arxiv/2401_1"], qp.PAPER_CHARS)) == 40      # heading whole (it fits 24), the long title cut to 40 with an ellipsis
    line = next(l for l in text.splitlines() if " ★ 2401_2#1" in l)
    assert line == " ★ 2401_2#1  c5  " + qp.shorten("A" * 80)                                                                  # no title for this paper: its heading keeps the 46-character cut and gets no ' | '
    assert "· 2402_9#7  c6  w2.8  Neighbour | Neighbour Paper" in text and "· 2404_8#1  c5  Typical of five | Memory for Agents" in text and "◆ 2404_8#1  c5  Typical of five | Memory for Agents" in text
    plain = "\n".join(l for _, l in qp.column_lines(res, rows, comm))
    assert "Draft then verify" in plain and "|" not in plain.split("SUBGRAPH")[1].split("TYPICAL")[0]               # no titles given: the lines are as before


def test_column_says_so_when_no_community_answered():
    res, rows, comm = world()
    res["partials"], res["global_answer"] = [], ""
    assert "(no community summary bore on the question)" in "\n".join(l for _, l in qp.column_lines(res, rows, comm))
    res["final"], res["evidence"] = "", []
    text = "\n".join(l for _, l in qp.column_lines(res, rows, comm))
    assert "(no answer was generated for this question)" in text and "TYPICAL SECTION" not in text


def test_strip_height_follows_the_longest_column_and_the_figure_is_that_tall():
    short, long_ = [("b", "x")] * 3, [("b", "x")] * 40
    assert qp.strip_height_in([short, long_]) == qp.PAD_IN + 4.2 + qp.LINE_IN * 40
    res, rows, comm = world()
    XY = np.random.default_rng(0).normal(size=(10, 2))
    fig = qp.draw_strip([res], [rows], [comm], XY, np.arange(10) % 8, width_in=8.0, dpi=50)
    assert abs(fig.get_figheight() - qp.strip_height_in([qp.column_lines(res, rows, comm)])) < 1e-9
    assert fig.get_figwidth() == 8.0
