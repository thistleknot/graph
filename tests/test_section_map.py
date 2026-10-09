"""Pins tools/section_map.py's pieces: the self-first neighbour layout, the cached-or-computed edges, and communities recovered from a fused graph.

Spec: approved plan step B4 (operator 2026-10-05), playbook.md T154. Synthetic clusters, small sizes, no disk beyond tmp_path.

Run:  pytest tests/test_section_map.py -v
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
import section_map as smap


def _world(n_per=70, k=3, dim=20, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim)) * 2.5
    E = np.concatenate([centres[c] + rng.normal(size=(n_per, dim)) for c in range(k)]) + 3.0
    E = E - E.mean(0)
    return (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32), np.repeat(np.arange(k), n_per)


def test_self_first_puts_the_node_itself_first_at_distance_zero_and_keeps_k_columns():
    idx = np.array([[5, 6, 7], [0, 2, 3]], np.int32)
    sim = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], np.float32)
    i, d = smap.self_first(idx, sim)
    assert i.shape == d.shape == (2, 3)
    assert i[:, 0].tolist() == [0, 1] and (d[:, 0] == 0).all()
    assert i[0].tolist() == [0, 5, 6] and np.allclose(d[0], [0, 0.1, 0.2], atol=1e-6)        # the last neighbour is dropped to keep k columns


def test_edges_for_computes_once_then_reads_the_cache_and_refits_the_same_null(tmp_path):
    X, _ = _world()
    path = str(tmp_path / "e.npz")
    cut1, e1 = smap.edges_for("dense", X, path, topk=5)
    assert Path(path).exists()
    cut2, e2 = smap.edges_for("dense", X, path, topk=5)
    assert (cut1.lam, cut1.mu, cut1.sd) == (cut2.lam, cut2.mu, cut2.sd)
    assert set(zip(e1["i"].tolist(), e1["j"].tolist())) == set(zip(e2["i"].tolist(), e2["j"].tolist()))
    assert np.array_equal(e1["knn_idx"], e2["knn_idx"])


def test_communities_from_graph_recovers_planted_clusters_from_the_fused_two_space_graph(monkeypatch):
    from sklearn.metrics import adjusted_rand_score as ari
    import arxiv_community_map as m
    X, lab = _world()
    Y = sp.csr_matrix(np.abs(np.random.default_rng(3).normal(size=(len(X), 40))) * (lab[:, None] == np.arange(40) % 3) + 1e-3)   # sparse view with the same clusters
    Y = sp.csr_matrix(Y.multiply(1 / np.sqrt(Y.multiply(Y).sum(1))))
    reps = {}
    for name, M in (("dense", X), ("sparse", Y)):
        cut = sg.fit_null(M, rows=100)
        reps[name] = {"X": M, "cut": cut, "edges": sg.stream_edges(M, cut, k_sigma=1.0, topk=8, block=64)}
    f = sg.fuse(len(X), reps)
    monkeypatch.setattr(m, "N_RES", 6)
    out = smap.communities_from_graph(f["G"], target_n=3)
    assert ari(lab, out["lab"]) > 0.9 and out["lab"].max() + 1 == 3
    assert out["cons_ari"] >= m.GATE


def test_a_fixed_resolution_skips_the_sweep_and_the_pick_and_clusters_at_exactly_that_resolution(monkeypatch):
    import arxiv_community_map as m
    import hnsw_communities as hc
    X, lab = _world()
    cut = sg.fit_null(X, rows=100)
    e = sg.stream_edges(X, cut, k_sigma=1.0, topk=8, block=64)
    G = sg.fuse(len(X), {"dense": {"X": X, "cut": cut, "edges": e}})["G"]
    called = []
    monkeypatch.setattr(m, "sweep", lambda G_: called.append(1) or (_ for _ in ()).throw(AssertionError("sweep ran")))
    out = smap.communities_from_graph(G, target_n=3, fixed_res=1.25)
    assert called == [] and out["chosen"] == 1.25 and hc.RES == 1.25                    # no sweep; the resolution is the one given
    assert out["res"] is None and out["plateau"] is None and out["lab"].shape == (len(X),)
    assert 0.0 <= out["cons_ari"] <= 1.0


def test_a_genre_run_gets_its_own_files_and_shares_only_the_sparse_edge_cache():
    base_edges, base_out, base_state = smap.run_names(0)
    edges, out, state = smap.run_names(8)
    assert (base_edges, base_out, base_state) == (smap.EDGES, smap.OUT, smap.STATE)             # no genre axes: the arm A names, unchanged
    assert edges["sparse"] == base_edges["sparse"] and edges["dense"] != base_edges["dense"]    # the sparse view is untouched by genre removal; the dense edges differ
    assert state != base_state and "xpg8" in state
    assert set(out) == set(base_out) and not set(out.values()) & set(base_out.values())         # no output file is shared with the arm A map
    assert all("xpg8" in v for v in out.values())
    assert smap.run_names(2)[2] != state                                                        # another k, another state file


def test_run_args_read_a_genre_k_and_refuse_anything_else():
    assert smap.parse_run_args([]) == 0 and smap.parse_run_args(["genre=8"]) == 8
    with pytest.raises(AssertionError):
        smap.parse_run_args(["genre8"])
    with pytest.raises(AssertionError):
        smap.parse_run_args(["genre=2", "genre=8"])


def test_main_rebinds_every_output_path_so_the_chunk_map_files_are_not_overwritten():
    import arxiv_community_map as m
    chunk = {k: getattr(m, k) for k in smap.OUT}
    assert all("sections_" in v for v in smap.OUT.values())
    assert not set(smap.OUT.values()) & set(chunk.values())
