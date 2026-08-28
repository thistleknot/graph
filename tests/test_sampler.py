"""Pins sampler.py: the deterministic evidence walker and its guards S1-S8.

The findings that set the defaults were produced by throwaway scripts. This
file is where they become durable -- if n=24 or the argmax default is ever
changed, something here fails and states why the number was chosen.

Guards pinned: S1 purity, S2 z-scoring, S3 without replacement, S4 enumeration,
S5 stored cid grouping, S7 re-derivable params, S8 argmax default.

The pure-selection tests need no database. The pipeline tests do, and skip
cleanly without one.

Run:  pytest tests/test_sampler.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

import graph_tools as gt
import sampler as sp

LABEL = "brown-50"


# ---------------------------------------------------------------- defaults
def test_defaults_are_the_measured_ones():
    """n=24: top-3 communities populate by 24 and pick up noise at 40.
    T=0: Boltzmann lost to argmax on every query measured."""
    assert sp.DEFAULT_N == 24
    assert sp.DEFAULT_T == 0.0
    assert sp.DEFAULT_HOPS == 2
    assert sp.DEFAULT_CAP == 150        # 2-hop p99 is 111


# ---------------------------------------------------------------- S8 argmax
def test_argmax_is_the_default_path():
    scores = {10: 0.9, 20: 0.1, 30: 0.7, 40: 0.5}
    picked, enumerated = sp.boltzmann_sample(scores, n=2)
    assert picked == [10, 30]
    assert enumerated is False


def test_argmax_ignores_the_seed():
    """S8/S1: at T=0 there is no randomness, so seeds cannot matter. This is
    what makes the default exactly reproducible rather than merely replayable."""
    scores = {i: (i * 37 % 11) / 10 for i in range(30)}
    a, _ = sp.boltzmann_sample(scores, n=8, T=0.0, seed=1)
    b, _ = sp.boltzmann_sample(scores, n=8, T=0.0, seed=987654)
    assert a == b


def test_argmax_orders_by_descending_score():
    scores = {1: 0.2, 2: 0.8, 3: 0.5, 4: 0.05}
    assert sp.boltzmann_sample(scores, n=3, T=0.0)[0] == [2, 3, 1]


# ---------------------------------------------------------------- S1 purity
def test_same_seed_same_sample():
    scores = {i: i / 50 for i in range(50)}
    a, _ = sp.boltzmann_sample(scores, n=10, T=1.0, seed=7)
    b, _ = sp.boltzmann_sample(scores, n=10, T=1.0, seed=7)
    assert a == b


def test_different_seed_different_sample_when_sampling():
    scores = {i: i / 50 for i in range(50)}
    a, _ = sp.boltzmann_sample(scores, n=10, T=1.0, seed=1)
    b, _ = sp.boltzmann_sample(scores, n=10, T=1.0, seed=2)
    assert a != b                       # T>0 must actually be stochastic


# ---------------------------------------------------------------- S3
def test_sampling_is_without_replacement():
    scores = {i: 1.0 for i in range(20)}
    picked, _ = sp.boltzmann_sample(scores, n=15, T=1.0, seed=3)
    assert len(picked) == len(set(picked)) == 15


# ---------------------------------------------------------------- S4
def test_enumerates_rather_than_pretending_to_sample():
    scores = {1: 0.5, 2: 0.4, 3: 0.3}
    picked, enumerated = sp.boltzmann_sample(scores, n=3)
    assert enumerated is True
    assert sorted(picked) == [1, 2, 3]


def test_enumeration_still_returns_score_order():
    """`sampled` means draw order everywhere, including the branch that takes
    every candidate -- otherwise the field changes meaning when n >= |C|."""
    picked, enumerated = sp.boltzmann_sample({7: 0.2, 8: 0.9, 9: 0.5}, n=99)
    assert enumerated is True
    assert picked == [8, 9, 7]


def test_enumeration_flag_is_false_when_actually_sampling():
    scores = {i: i / 10 for i in range(10)}
    assert sp.boltzmann_sample(scores, n=4)[1] is False


def test_empty_candidates_yield_nothing():
    assert sp.boltzmann_sample({}, n=5) == ([], False)


# ---------------------------------------------------------------- S2
def test_flat_scores_do_not_divide_by_zero():
    """Zero variance means z-scoring has no scale; must not produce NaN."""
    scores = {i: 0.42 for i in range(12)}
    picked, _ = sp.boltzmann_sample(scores, n=5, T=1.0, seed=0)
    assert len(picked) == 5
    assert all(p in scores for p in picked)


def test_z_scoring_makes_T_scale_free():
    """S2: raw BM25 is unbounded, so exp() of it saturates to one-hot and T
    stops meaning anything. After z-scoring, multiplying every score by a
    constant must not change the sample."""
    base = {i: float(i) for i in range(40)}
    scaled = {i: float(i) * 1000.0 for i in range(40)}
    a, _ = sp.boltzmann_sample(base, n=10, T=1.0, seed=5)
    b, _ = sp.boltzmann_sample(scaled, n=10, T=1.0, seed=5)
    assert a == b


def test_low_temperature_concentrates_on_high_scores():
    scores = {i: i / 60 for i in range(60)}
    cold, _ = sp.boltzmann_sample(scores, n=10, T=0.05, seed=0)
    hot, _ = sp.boltzmann_sample(scores, n=10, T=50.0, seed=0)
    assert sum(cold) > sum(hot)         # cold favours the high-scoring tail


# ---------------------------------------------------------------- pipeline
@pytest.fixture(scope="module")
def live():
    try:
        conn = gt.connect()
        run = gt.get_run(conn, LABEL)
    except (psycopg.OperationalError, LookupError) as e:   # pragma: no cover
        pytest.skip(f"no live run: {e}")
    yield conn, run
    conn.close()


def test_bundle_is_reproducible_and_carries_its_params(live):
    """S7: the bundle must be re-derivable from what it records."""
    conn, run = live
    a = sp.evidence(conn, run, "jury trial grand jury investigation")
    b = sp.evidence(conn, run, "jury trial grand jury investigation")
    assert a.sampled == b.sampled
    assert a.top_cids == b.top_cids
    assert a.params["n"] == sp.DEFAULT_N
    assert a.params["T"] == sp.DEFAULT_T
    assert a.run_id == str(run.run_id)


def test_top_community_is_stable_across_n(live):
    """Measured: the #1 community is identical at every n in {8,...,40}.
    This is the property the walker's usefulness rests on."""
    conn, run = live
    tops = set()
    for n in (8, 12, 16, 24, 40):
        b = sp.evidence(conn, run, "school children teacher education", n=n)
        assert b.communities, "no communities surfaced"
        tops.add(b.communities[0]["cid"])
    assert len(tops) == 1, f"top community drifted with n: {tops}"


def test_communities_come_from_stored_cids(live):
    """S5: never a query-time partition. Every reported cid must exist in the
    run's community table with a matching size."""
    conn, run = live
    b = sp.evidence(conn, run, "molecular structure crystal")
    for c in b.communities:
        stored = gt.community(conn, run, c["cid"])
        assert stored is not None
        assert stored["size"] == c["size"]
        assert c["hits"] <= c["size"]


def test_candidate_set_respects_the_cap(live):
    conn, run = live
    scores, anchors = sp.candidate_scores(conn, run,
                                          "jury trial grand jury investigation",
                                          cap=20)
    assert len(scores) <= 20
    assert anchors


def test_anchors_are_included_in_candidates(live):
    conn, run = live
    scores, anchors = sp.candidate_scores(conn, run, "molecular structure crystal")
    assert set(anchors) <= set(scores)


def test_unmatchable_query_returns_empty_bundle(live):
    """Must be real gibberish. An earlier version of this test used
    'nonexistent gibberish token', which matched 24 chunks -- those are
    ordinary English words that occur in Brown."""
    conn, run = live
    b = sp.evidence(conn, run, "zzzqqq xxwwvv qqzzxx vvbbnn")
    assert b.sampled == []
    assert b.communities == []
    assert b.concentration() == 0.0


def test_concentration_reports_head_share(live):
    conn, run = live
    b = sp.evidence(conn, run, "school children teacher education")
    assert 0.0 < b.concentration() <= 1.0
    top = b.communities[0]["hits"]
    total = sum(c["hits"] for c in b.communities)
    assert b.concentration() == pytest.approx(top / total)


def test_k_comm_truncates_the_histogram(live):
    conn, run = live
    b = sp.evidence(conn, run, "church religious faith congregation", k_comm=2)
    assert len(b.communities) <= 2
