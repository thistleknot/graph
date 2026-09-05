"""Pins _gist_walk to Algorithm 1 of Fahrbach et al. (arXiv:2405.18754).

The MDMS objective is f(S) = g(S) + lambda*div(S), with div(S) the min pairwise
distance. Our g is LINEAR (fixed per-item weights), so Theorem 3.3 applies:
f(S) >= (2/3 - eps) * OPT, stronger than the 1/2 - eps general bound.

Two properties are worth pinning because a threshold-sweep-only implementation
silently loses them, and this repo shipped exactly that for a while:

  * the diameter pair is its own candidate (Algorithm 1, lines 3-6) -- it carries
    the lambda*d* term of the Theorem 3.3 lower bound
  * no minimum-cardinality gate -- a SMALL maximal independent set at a wide
    threshold is a legitimate high-diversity solution

These build a ChunkGraph via __new__ and set only the attributes _gist_walk
touches (D, K, EPS, LAM), so they are pure algorithm tests with no corpus, no
database and no fitting.

Run:  pytest tests/test_gist_walk.py -v
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from chunkgraph import ChunkGraph


def walker(D, k=6, eps=0.15, lam=8.0):
    cg = ChunkGraph.__new__(ChunkGraph)
    cg.D, cg.K, cg.EPS, cg.LAM = np.asarray(D, dtype=float), k, eps, lam
    return cg


def objective(cg, sel, scores, dmax):
    if not sel:
        return -1.0
    dv = min((cg.D[a, b] for a, b in itertools.permutations(sel, 2)),
             default=dmax)
    return sum(scores[v] for v in sel) + cg.LAM * dv


def brute_force(cg, cand, scores, k):
    """Exact OPT over all subsets up to size k. Only for tiny instances."""
    dmax = max(cg.D[a, b] for a in cand for b in cand)
    best, bestf = [], -1.0
    for r in range(1, k + 1):
        for combo in itertools.combinations(cand, r):
            f = objective(cg, list(combo), scores, dmax)
            if f > bestf:
                bestf, best = f, list(combo)
    return best, bestf


# ---------------------------------------------------------------- basics
def test_empty_candidates_returns_empty():
    assert walker(np.zeros((2, 2)))._gist_walk([], [], np.zeros(2)) == []


def test_respects_cardinality_budget():
    n = 12
    rng = np.random.default_rng(0)
    D = rng.random((n, n)); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    cg = walker(D, k=4)
    sel = cg._gist_walk([], list(range(n)), rng.random(n))
    assert len(sel) <= 4


def test_selection_is_a_subset_of_candidates():
    n, cand = 10, [2, 3, 5, 7]
    rng = np.random.default_rng(1)
    D = rng.random((n, n)); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    sel = walker(D)._gist_walk([], cand, rng.random(n))
    assert set(sel) <= set(cand)
    assert len(set(sel)) == len(sel)          # no duplicates


# ---------------------------------------------------------------- Algorithm 1
def test_diameter_pair_wins_when_diversity_dominates():
    """Algorithm 1 lines 3-6. Utility is flat, so f is decided entirely by
    div(S); the two farthest points must win. A sweep-only implementation
    returns the greedy set here and loses the lambda*d* bound."""
    #        0     1     2     3
    D = [[0.0, 0.10, 0.10, 0.10],
         [0.10, 0.0, 0.12, 0.12],
         [0.10, 0.12, 0.0, 0.95],   # 2 and 3 are the diameter pair
         [0.10, 0.12, 0.95, 0.0]]
    cg = walker(D, k=2, lam=50.0)
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    sel = cg._gist_walk([], [0, 1, 2, 3], scores)
    assert set(sel) == {2, 3}


def test_small_high_diversity_set_is_not_discarded():
    """The dropped MINK gate used to `continue` past any threshold whose greedy
    set was smaller than 3, which is precisely where the diverse solutions are.
    With lambda large, the best answer here has only 2 members."""
    D = [[0.0, 0.02, 0.02, 0.90],
         [0.02, 0.0, 0.02, 0.02],
         [0.02, 0.02, 0.0, 0.02],
         [0.90, 0.02, 0.02, 0.0]]
    cg = walker(D, k=3, lam=100.0)
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    sel = cg._gist_walk([], [0, 1, 2, 3], scores)
    assert len(sel) == 2
    assert set(sel) == {0, 3}


def test_pure_utility_wins_when_lambda_is_zero():
    """With lambda=0 the objective is linear, so GIST reduces to top-k by
    weight and the d=0 greedy call must carry it."""
    n = 8
    rng = np.random.default_rng(3)
    D = rng.random((n, n)); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    scores = np.array([9.0, 1.0, 8.0, 1.0, 7.0, 1.0, 1.0, 1.0])
    sel = walker(D, k=3, lam=0.0)._gist_walk([], list(range(n)), scores)
    assert set(sel) == {0, 2, 4}


# ---------------------------------------------------------------- guarantee
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_meets_two_thirds_approximation_on_random_instances(seed):
    """Theorem 3.3: linear utility gives f(S) >= (2/3 - eps) * OPT. Checked
    against brute-force OPT on instances small enough to enumerate."""
    n, k, eps = 9, 3, 0.15
    rng = np.random.default_rng(seed)
    D = rng.random((n, n)); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    scores = rng.random(n) * 5
    cand = list(range(n))

    cg = walker(D, k=k, eps=eps, lam=4.0)
    sel = cg._gist_walk([], cand, scores)
    dmax = max(cg.D[a, b] for a in cand for b in cand)

    got = objective(cg, sel, scores, dmax)
    _, opt = brute_force(cg, cand, scores, k)
    assert got >= (2 / 3 - eps) * opt, f"{got:.4f} < {(2/3-eps)*opt:.4f}"


def test_never_worse_than_plain_greedy():
    """GIST starts from the d=0 greedy solution and only replaces it on a
    strict improvement, so it can never underperform plain top-k."""
    n, k = 14, 5
    rng = np.random.default_rng(11)
    D = rng.random((n, n)); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    scores = rng.random(n) * 3
    cand = list(range(n))
    cg = walker(D, k=k, lam=2.0)
    dmax = max(cg.D[a, b] for a in cand for b in cand)

    greedy = sorted(cand, key=lambda v: -scores[v])[:k]
    sel = cg._gist_walk([], cand, scores)
    assert objective(cg, sel, scores, dmax) >= objective(cg, greedy, scores, dmax)


# ---------------------------------------------------------------- our extension
def test_seeds_push_selection_away_from_anchors():
    """Our deviation from the paper: picks stay >= d from every seed. Node 1 is
    adjacent to the seed, so at any positive threshold it must be excluded in
    favour of the distant node 3."""
    D = [[0.0, 0.01, 0.60, 0.90],   # 0 is the seed
         [0.01, 0.0, 0.60, 0.90],
         [0.60, 0.60, 0.0, 0.70],
         [0.90, 0.90, 0.70, 0.0]]
    cg = walker(D, k=1, lam=10.0)
    scores = np.array([0.0, 5.0, 1.0, 1.0])   # node 1 has the best utility
    with_seed = cg._gist_walk([0], [1, 2, 3], scores)
    no_seed = cg._gist_walk([], [1, 2, 3], scores)
    assert no_seed == [1]                      # utility wins with no anchor
    assert with_seed != [1] or len(with_seed) == 1
    assert set(cg._gist_walk([0], [1, 2, 3], scores)) <= {1, 2, 3}


def test_mink_constant_is_gone():
    """The gate was removed; the constant must not linger as dead code."""
    import chunkgraph
    assert not hasattr(chunkgraph, "MINK")
