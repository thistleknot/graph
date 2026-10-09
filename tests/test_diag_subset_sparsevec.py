"""Pins the two pure decisions in the subset lane: the feasibility cap and the
heaviest-kept truncation.

The allocation cap is the correction to the operator's log-ratio rule -- a corpus
never contributes more records than it has -- and if it silently regressed to the
raw quota, the sample would draw with replacement and every df in the small corpus
would be inflated. The truncation is what makes the numpy parity check honest:
psql holds the 1000 heaviest pieces per row, so numpy must too before the two are
compared.

No database, no corpus.  Run:  pytest tests/test_diag_subset_sparsevec.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from diag_subset_sparsevec import allocation, truncate_rows  # noqa: E402


def test_allocation_caps_the_small_corpus_at_what_it_has():
    """Measured shape: 588 neop vs 168,794 arxiv, total 100k."""
    al = allocation(588, 168_794, 100_000)
    assert al["infeasible"] is True
    assert al["quota_neop"] == 34_631            # the log-ratio ask
    assert al["take_neop"] == 588                # capped at what exists
    assert al["take_arxiv"] == 100_000 - 588
    assert al["take_neop"] + al["take_arxiv"] == 100_000


def test_allocation_uses_the_log_ratio_when_both_corpora_are_large():
    al = allocation(50_000, 200_000, 100_000)
    assert al["infeasible"] is False
    # log(50000)/(log(50000)+log(200000)) = 10.82/(10.82+12.21) = 0.4698
    assert al["take_neop"] == round(100_000 * al["log_ratio_neop"])
    assert 46_000 < al["take_neop"] < 48_000
    assert al["take_arxiv"] == 100_000 - al["take_neop"]


def test_truncate_rows_keeps_the_heaviest_and_leaves_short_rows_alone():
    M = sparse.csr_matrix(np.array([
        [0.1, 9.0, 0.2, 5.0, 7.0],      # 5 nnz -> keep top-3 by weight: cols 1,3,4
        [0.0, 1.0, 0.0, 2.0, 0.0],      # 2 nnz -> untouched
    ]))
    T, n = truncate_rows(M, cap=3)
    assert n == 1
    r0 = T.getrow(0).toarray().ravel()
    assert r0.tolist() == [0.0, 9.0, 0.0, 5.0, 7.0]
    assert T.getrow(1).toarray().ravel().tolist() == [0.0, 1.0, 0.0, 2.0, 0.0]
    assert T.nnz == 5
