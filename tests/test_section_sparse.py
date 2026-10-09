"""Pins tools/section_sparse.py: the unigram and adjacent-pair views, the unit-row inner product, and that adjacency breaks the bag-of-tokens tie.

Spec: approved plan step B2 (operator 2026-10-05), playbook.md T152. Tiny synthetic corpora, no disk, no model.

Run:  pytest tests/test_section_sparse.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import section_sparse as ss2

FILL = [["f%d" % i, "g%d" % i, "h%d" % i, "k%d" % i] for i in range(6)]            # six documents sharing nothing with the rest (N = 12)
DOCS = [["alpha", "beta", "gamma", "delta"],        # 0
        ["delta", "gamma", "beta", "alpha"],        # 1: the same bag as 0, reversed
        ["alpha", "beta", "x", "y"],                # 2
        ["p", "q", "r", "s"],                       # 3
        ["q", "r", "s", "t"],                       # 4
        ["alpha", "beta", "gamma", "delta"]] + FILL  # 5: identical to 0


def _views(lam=1.0, cap=ss2.MAX_NNZ):
    U, col, idf, whales, avgdl = ss2.unigram_view(DOCS, cap)
    P, pcol, pidf, pavg, tok = ss2.pair_view(DOCS, min_df=2, cap=cap)
    return U, col, idf, avgdl, P, pcol, pidf, pavg, tok


def test_rows_are_unit_length_and_the_pair_view_keeps_order():
    U, col, idf, avgdl, P, pcol, pidf, pavg, tok = _views()
    for M in (U, P):
        n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
        assert np.allclose(n[n > 0], 1.0)
    a, b = tok["alpha"], tok["beta"]
    assert (a, b) in pcol and (b, a) not in pcol                          # (alpha, beta) is in 3 documents; (beta, alpha) in 1, so it is dropped
    assert P[0, pcol[(a, b)]] > 0 and P[1].nnz < P[0].nnz                  # document 1 (reversed) holds fewer of the kept pairs


def test_pairs_in_fewer_than_min_df_or_in_over_half_the_documents_are_dropped():
    docs = [["u", "v"]] * 7 + [["a%d" % i, "b%d" % i] for i in range(5)]    # (u, v) in 7 of 12 documents: a whale; the a-b pairs have df 1
    P, pcol, pidf, avg, tok = ss2.pair_view(docs, min_df=2)
    assert pcol == {} and P.nnz == 0


def test_rows_are_capped_at_the_heaviest_entries():
    docs = [["w%d" % j for j in range(40)] for _ in range(3)] + [["z%d" % i, "y%d" % i, "x%d" % i] for i in range(9)]
    P, pcol, pidf, avg, tok = ss2.pair_view(docs, min_df=2, cap=10)
    assert np.diff(P.indptr)[:3].max() == 10 and np.diff(P.indptr)[:3].min() == 10


def test_the_inner_product_of_the_joined_rows_is_cos_unigram_plus_lambda_cos_pairs():
    U, col, idf, avgdl, P, pcol, pidf, pavg, tok = _views()
    lam = 0.5
    Z = sp.hstack([U, np.sqrt(lam) * P]).tocsr()
    assert np.allclose((Z @ Z.T).toarray(), (U @ U.T).toarray() + lam * (P @ P.T).toarray(), atol=1e-6)


def test_joined_rows_are_unit_length_and_their_inner_product_is_the_weighted_average_of_the_two_cosines():
    U, col, idf, avgdl, P, pcol, pidf, pavg, tok = _views()
    hasP = P.getnnz(1) > 0
    assert hasP.sum() > 3 and (~hasP & (U.getnnz(1) > 0)).sum() > 0                  # the corpus has rows with pairs and rows with tokens only
    for lam in (0.5, 1.0, 2.0):
        Z = ss2.joined(U, P, lam)
        n = np.sqrt(np.asarray(Z.multiply(Z).sum(1)).ravel())
        nz = np.asarray((U.getnnz(1) + P.getnnz(1)) > 0)
        assert np.allclose(n[nz], 1.0, atol=1e-5) and np.allclose(n[~nz], 0.0)       # every non-empty row is unit, including the rows with no pair
        idx = np.where(hasP)[0]
        want = ((U[idx] @ U[idx].T).toarray() + lam * (P[idx] @ P[idx].T).toarray()) / (1 + lam)
        assert np.allclose((Z[idx] @ Z[idx].T).toarray(), want, atol=1e-5) and (Z @ Z.T).max() <= 1.0 + 1e-5


def test_build_runs_end_to_end_on_records_with_words_and_with_bpe_pieces():
    recs = [{"text": " ".join(d)} for d in DOCS]
    for merges in (None, 30):
        Z, info = ss2.build(recs, merges=merges, lam=1.0)
        assert Z.shape[0] == len(recs) and info["U"] + info["P"] == Z.shape[1] and info["lam"] == 1.0
        sim = (Z @ Z.T).toarray()
        assert sim[0, 5] == pytest.approx(1.0, abs=1e-5)                          # documents 0 and 5 are identical
        assert sim[0, 5] > sim[0, 1]                                              # the reversed bag is less similar: adjacency separates it


def test_adjacency_demotes_the_reversed_bag_which_the_unigram_view_cannot_tell_apart():
    U, col, idf, avgdl, P, pcol, pidf, pavg, tok = _views()
    q = ["alpha", "beta", "gamma", "delta"]                               # the query is document 0's order
    u, p = ss2.query_rows(q, col, idf, avgdl, pcol, pidf, pavg, tok)
    QU, QP = ss2.to_matrix([u], U.shape[1]), ss2.to_matrix([p], P.shape[1])
    assert ss2.rank_of_truth(U, P, QU, QP, [1], lam=0.0)[0] == 0           # reversed document 1 ties the others: a bag has no order
    assert ss2.rank_of_truth(U, P, QU, QP, [1], lam=1.0)[0] == 2           # with pairs, documents 0 and 5 (same order) are strictly above it
    assert ss2.rank_of_truth(U, P, QU, QP, [0], lam=1.0)[0] == 0


def test_query_rows_drop_what_the_corpus_never_kept_and_are_unit_length():
    U, col, idf, avgdl, P, pcol, pidf, pavg, tok = _views()
    u, p = ss2.query_rows(["alpha", "beta", "zzz", "gamma"], col, idf, avgdl, pcol, pidf, pavg, tok)
    assert set(u) <= set(col.values()) and abs(sum(v * v for v in u.values()) - 1.0) < 1e-9
    assert len(p) == 1                                                    # only (alpha, beta) is adjacent AND kept; (beta, zzz), (zzz, gamma) are not
    assert ss2.query_rows([], col, idf, avgdl, pcol, pidf, pavg, tok) == ({}, {})
