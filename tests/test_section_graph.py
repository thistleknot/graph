"""Pins tools/section_graph.py: the streamed edges equal the brute-force ones, each node's neighbours are its exact top-k, the negative tail is
counted, the union carries provenance. Synthetic clusters, dense and sparse inputs, no disk.

Spec: approved plan step B3 (operator 2026-10-05), playbook.md T153.

Run:  pytest tests/test_section_graph.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import section_graph as sg


def _world(n_per=60, k=4, dim=24, seed=0):
    """k clusters of n_per rows with a shared component, centred and L2-normalised."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim)) * 2.0
    E = np.concatenate([centres[c] + rng.normal(size=(n_per, dim)) for c in range(k)]) + 3.0
    E = E - E.mean(0)
    return (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32), np.repeat(np.arange(k), n_per)


def _brute(X):
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    return S


def _pairs(a, b):
    return {(int(i), int(j)) for i, j in zip(a, b)}


def test_fit_null_is_seeded_and_the_threshold_rises_with_k():
    X, _ = _world()
    a, b = sg.fit_null(X, rows=100, seed=3), sg.fit_null(X, rows=100, seed=3)
    assert (a.lam, a.mu, a.sd) == (b.lam, b.mu, b.sd)
    assert a.threshold(1.0) < a.threshold(2.0) < a.threshold(3.0)
    assert float(a.z(np.array([a.threshold(2.0)]))[0]) == pytest.approx(2.0, abs=1e-3)      # threshold inverts z


def test_with_every_neighbour_kept_the_edges_are_exactly_the_brute_force_pairs_above_the_threshold():
    X, _ = _world()
    cut = sg.fit_null(X, rows=100)
    e = sg.stream_edges(X, cut, k_sigma=1.0, topk=len(X) - 1, block=37)                  # a block size that does not divide N: seams are exercised
    S = _brute(X)
    thr = e["stats"]["threshold"]
    want = _pairs(*np.nonzero(np.triu(S >= thr, 1)))
    assert _pairs(e["i"], e["j"]) == want and len(want) > 100
    assert np.allclose(e["s"], S[e["i"], e["j"]])


def test_each_nodes_neighbours_are_its_exact_top_k_and_an_edge_is_a_significant_member_of_one():
    X, _ = _world()
    cut = sg.fit_null(X, rows=100)
    e = sg.stream_edges(X, cut, k_sigma=1.0, topk=5, block=41)
    S = _brute(X)
    top = np.argsort(-S, axis=1)[:, :5]
    assert np.allclose(e["knn_sim"], np.take_along_axis(S, top, 1), atol=1e-6)            # same similarities (ties could swap ids; random data has none)
    thr = e["stats"]["threshold"]
    want = set()
    for i in range(len(X)):
        for j in top[i]:
            if S[i, j] >= thr:
                want.add((min(i, int(j)), max(i, int(j))))
    assert _pairs(e["i"], e["j"]) == want and e["stats"]["significant_edges"] == len(want)
    assert np.allclose(e["s"], S[e["i"], e["j"]])
    deg = np.bincount(np.concatenate([e["i"], e["j"]]), minlength=len(X))
    assert e["stats"]["nodes_without_edge"] == int((deg == 0).sum())


def test_a_global_budget_is_gone_so_one_tight_clique_cannot_take_every_edge():
    rng = np.random.default_rng(2)
    clique = np.tile(rng.normal(size=(1, 16)), (30, 1)) + rng.normal(size=(30, 16)) * 0.01      # 30 near-duplicates
    rest = rng.normal(size=(200, 16))
    E = np.concatenate([clique, rest]) - np.concatenate([clique, rest]).mean(0)
    X = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    cut = sg.fit_null(X, rows=100)
    e = sg.stream_edges(X, cut, k_sigma=0.0, topk=4, block=64)
    inside = ((e["i"] < 30) & (e["j"] < 30)).sum()
    assert inside < len(e["i"]) and (np.bincount(np.concatenate([e["i"], e["j"]]), minlength=len(X))[30:] > 0).sum() > 100   # the other rows keep their own edges


def test_with_a_group_no_neighbour_edge_or_tail_pair_joins_two_rows_of_the_same_group():
    X, lab = _world()
    group = np.arange(len(X)) // 6                                                       # contiguous blocks of 6 rows = "papers"
    cut = sg.fit_null(X, rows=100)
    e = sg.stream_edges(X, cut, k_sigma=1.0, topk=5, block=41, group=group)
    assert (group[e["i"]] != group[e["j"]]).all() and len(e["i"]) > 100
    assert (group[:, None] != group[e["knn_idx"]]).all()                                 # every neighbour is from another group
    S = _brute(X)
    S[group[:, None] == group[None, :]] = -np.inf
    top = np.argsort(-S, axis=1)[:, :5]
    assert np.allclose(e["knn_sim"], np.take_along_axis(S, top, 1), atol=1e-6)           # the exact top-k among the OTHER groups
    thr = e["stats"]["threshold"]
    assert e["stats"]["positive_tail"] == int(np.triu(S >= thr, 1).sum())
    assert e["stats"]["negative_tail"] == int(np.triu(S <= -thr, 1).sum())


def test_the_negative_tail_is_counted_against_the_mirror_of_the_positive_cut():
    X, _ = _world()
    cut = sg.fit_null(X, rows=100)
    e = sg.stream_edges(X, cut, k_sigma=1.0, topk=3, block=50)
    S = _brute(X)
    t = -e["stats"]["neg_threshold"]
    assert e["stats"]["negative_tail"] == int(np.triu(S <= -t, 1).sum()) and e["stats"]["negative_tail"] > 0
    assert e["stats"]["strongest_negative"][0][2] == pytest.approx(float(np.triu(S, 1)[np.triu_indices(len(X), 1)].min()), abs=1e-6)
    assert e["stats"]["positive_tail"] == int(np.triu(S >= t, 1).sum())


def test_sparse_input_gives_the_same_edges_as_dense_input():
    X, _ = _world()
    Xs = sp.csr_matrix(X)
    cut = sg.fit_null(X, rows=100)
    d = sg.stream_edges(X, cut, k_sigma=1.5, topk=4, block=40)
    s = sg.stream_edges(Xs, cut, k_sigma=1.5, topk=4, block=40)
    assert _pairs(d["i"], d["j"]) == _pairs(s["i"], s["j"])
    assert np.allclose(d["knn_sim"], s["knn_sim"], atol=1e-5)


def test_significant_edges_join_rows_of_the_same_cluster_far_more_than_chance():
    X, lab = _world()
    cut = sg.fit_null(X, rows=100)
    e = sg.stream_edges(X, cut, k_sigma=2.0, topk=5, block=64)
    same = float((lab[e["i"]] == lab[e["j"]]).mean())
    base = float(sum((lab == c).sum() * ((lab == c).sum() - 1) for c in np.unique(lab)) / (len(lab) * (len(lab) - 1)))
    assert base == pytest.approx(0.25, abs=0.01) and same > 0.9


def test_fuse_is_the_union_with_provenance_exact_similarities_and_no_isolates():
    X, lab = _world()
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(len(X), 10)).astype(np.float32)                                  # a second, unrelated representation
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    reps = {}
    for name, M in (("dense", X), ("other", Y)):
        cut = sg.fit_null(M, rows=100)
        reps[name] = {"X": M, "cut": cut, "edges": sg.stream_edges(M, cut, k_sigma=2.0, topk=4, block=64)}
    f = sg.fuse(len(X), reps, backbone=2)
    G = f["G"]
    assert (G != G.T).nnz == 0 and G.data.min() > 0
    assert (np.asarray(G.sum(1)).ravel() > 0).all()                                       # the backbone leaves no isolated node
    only_dense = (f["prov"] & 1 > 0) & ~(f["prov"] & 2 > 0)
    assert only_dense.sum() > 0
    assert np.allclose(f["s"]["dense"], (X[f["i"]] * X[f["j"]]).sum(1), atol=1e-6)         # exact in BOTH spaces, not only where significant
    assert np.allclose(f["s"]["other"], (Y[f["i"]] * Y[f["j"]]).sum(1), atol=1e-6)
    zbar = (f["z"]["dense"] + f["z"]["other"]) / 2
    assert np.allclose(f["w"], np.maximum(zbar, 0) + sg.EDGE_FLOOR, atol=1e-5)
