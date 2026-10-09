"""Pins tools/section_map.split_oversize: only communities above factor x the serving-size mean are re-partitioned, inside their own subgraph, under the
same seed-stability gate; a failed gate leaves the community whole; ids stay contiguous.

Spec: operator 2026-10-05/06 (communities should be topics; "recommend one" -> split the oversize ones), playbook.md T154. Synthetic planted graphs, no disk.

Run:  pytest tests/test_section_map_split.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import arxiv_community_map as m
import section_map as smap


def _planted(blocks, p_in=0.5, p_out=0.002, seed=0):
    """Symmetric weighted graph with the given block sizes: dense inside a block, near-empty between."""
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(len(blocks)), blocks)
    n = len(lab)
    same = lab[:, None] == lab[None, :]
    A = np.triu((rng.random((n, n)) < np.where(same, p_in, p_out)), 1)
    return sp.csr_matrix((A | A.T).astype(np.float32)), lab


def test_a_community_of_three_planted_topics_is_split_in_three_and_the_small_one_is_left_alone(monkeypatch):
    monkeypatch.setattr(m, "N_RES", 6)
    G, truth = _planted([60, 60, 60, 12])
    lab = np.where(truth == 3, 1, 0)                                     # the three big blocks sit in ONE community 0; the small block is community 1
    out, genre, rep = smap.split_oversize(G, lab, target_n=4, bound=72)  # mean = 192/4 = 48: community 0 (180) > 72, community 1 (12) is not
    assert [r["community"] for r in rep if r["depth"] == 0] == [0] and rep[0]["applied"] and rep[0]["parts"] == 3
    assert rep[0]["seed_ari"] >= m.GATE
    from sklearn.metrics import adjusted_rand_score as ari
    assert ari(truth, out) > 0.95                                        # the planted blocks are recovered
    assert sorted(np.unique(out).tolist()) == list(range(4))             # ids stay contiguous
    assert (out[lab == 1] == out[lab == 1][0]).all()                     # the small community is one community still
    assert np.array_equal(genre, lab)                                    # the genre level is the label before any split


def test_nothing_is_split_when_no_community_exceeds_the_bound(monkeypatch):
    monkeypatch.setattr(m, "N_RES", 6)
    G, truth = _planted([40, 40, 40])
    out, genre, rep = smap.split_oversize(G, truth, target_n=3, bound=80)    # every community is 40
    assert rep == [] and np.array_equal(out, truth) and np.array_equal(genre, truth)


def test_a_split_that_fails_the_seed_stability_gate_is_not_applied(monkeypatch):
    monkeypatch.setattr(m, "N_RES", 6)
    monkeypatch.setattr(m, "GATE", 1.01)                                 # no partition can reach it
    G, truth = _planted([60, 60, 60])
    out, genre, rep = smap.split_oversize(G, np.zeros(len(truth), int), target_n=6, bound=30)
    assert len(rep) == 1 and rep[0]["applied"] is False
    assert np.array_equal(out, np.zeros(len(truth), int))                # the community stays whole


def test_the_default_bound_is_the_square_root_of_the_edge_count(monkeypatch):
    monkeypatch.setattr(m, "N_RES", 6)
    G, truth = _planted([60, 60, 60])
    bound = float(np.sqrt(G.nnz / 2))                                    # about 52 for these dense blocks
    lab = np.where(np.arange(len(truth)) < 30, 0, 1)                     # community 0 has 30 sections (< bound), community 1 has 150 (> bound)
    assert 30 < bound < 150
    _, _, rep = smap.split_oversize(G, lab, target_n=3)                  # no `bound` given: the default is sqrt(edge count)
    assert len(rep) >= 1 and rep[0]["community"] == 1 and rep[0]["size"] == 150
    assert all(r["community"] != 0 for r in rep)                         # the small community was never a candidate


def test_a_part_still_above_the_bound_is_split_again_until_max_depth(monkeypatch):
    monkeypatch.setattr(m, "N_RES", 6)
    G, truth = _planted([60, 60, 60])
    lab = np.zeros(len(truth), int)
    _, _, deep = smap.split_oversize(G, lab, target_n=4, bound=50, max_depth=3)       # three parts of 60 each exceed 50 and are queued again
    _, _, flat = smap.split_oversize(G, lab, target_n=4, bound=50, max_depth=1)
    assert {r["depth"] for r in deep} == {0, 1} and {r["depth"] for r in flat} == {0}
    assert all(r["depth"] == 0 for r in deep if r["applied"])             # a homogeneous planted block has no structure left: the depth-1 attempts do not apply
