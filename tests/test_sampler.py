"""Pins sampler.py: the deterministic evidence walker and its guards S1-S8.

The findings that set the defaults were produced by throwaway scripts. This
file is where they become durable -- if n=24 or the argmax default is ever
changed, something here fails and states why the number was chosen.

Guards pinned: S1 purity, S2 z-scoring, S3 without replacement, S4 enumeration,
S5 stored cid grouping, S7 re-derivable params, S8 argmax default, S15/S16
(amended 2026-09-03) additive-only competitive anchor extras: base is the
untouched global top-k, a minority source may add at most one competitive
extra, extras never displace a base anchor.

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
    assert sp.DEFAULT_K_ANCHOR == 3     # allocation changes WHERE, never HOW MANY
    assert sp.COMPETITIVE_FRAC == 0.5


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
    """Measured on the 1,789-chunk graph: the #1 community is identical at
    every n up to DEFAULT_N=24 ("at n=40 noise reaches position 3"). Pinned
    on the 500-document run -- comparable node count -- not on brown-50, whose
    51 nodes in 5 communities flip the top-1 between n=8 and n=24 outright."""
    conn, _ = live
    run = gt.get_run(conn, "brown-500-dual")
    tops, sampled_ns = set(), []
    for n in (8, 12, 16, 24):
        b = sp.evidence(conn, run, "school children teacher education", n=n)
        assert b.communities, "no communities surfaced"
        if b.enumerated:            # S4: n >= |C| is an enumeration, not a sample;
            continue                # the module docstring records this exact artifact
        tops.add(b.communities[0]["cid"]); sampled_ns.append(n)
    assert len(sampled_ns) >= 2, "fixture too small: every n enumerated"
    assert len(tops) == 1, f"top community drifted with n {sampled_ns}: {tops}"


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


# ------------------------------------------------ ef_search (S9-S12)


def _dual(live):
    conn, _ = live
    try:
        return conn, gt.get_run(conn, "brown-500-dual")
    except Exception as e:                                # pragma: no cover
        pytest.skip(f"no dual run: {e}")


Q = "jury trial grand jury investigation"


def test_ef_search_depth_is_set_by_ef_not_by_the_hop_cap(live):
    """S9: depth is emergent. Raising max_hops must not change the answer."""
    conn, run = _dual(live)
    W8, t8 = sp.ef_search(conn, run, Q, ef=24, T=0.0, m=3, max_hops=8)
    W40, t40 = sp.ef_search(conn, run, Q, ef=24, T=0.0, m=3, max_hops=40)
    assert W8 == W40
    assert t8["depth"] == t40["depth"]
    assert t8["hop_capped"] == 0 and t40["hop_capped"] == 0
    assert t8["stop"] == "converged"


def test_ef_search_result_width_is_ef(live):
    conn, run = _dual(live)
    for ef in (8, 24, 64):
        W, t = sp.ef_search(conn, run, Q, ef=ef, T=0.0, m=3)
        assert len(W) == ef, f"ef={ef} returned {len(W)}"
        assert t["ef"] == ef


def test_ef_search_larger_ef_reaches_deeper(live):
    """S10: ef is the dial. More width -> more exploration -> more depth."""
    conn, run = _dual(live)
    _, t_small = sp.ef_search(conn, run, Q, ef=8, T=0.0, m=3)
    _, t_big = sp.ef_search(conn, run, Q, ef=64, T=0.0, m=3)
    assert t_big["depth"] > t_small["depth"]
    assert t_big["expanded"] > t_small["expanded"]


def test_ef_search_results_are_monotone_in_score_and_carry_anchor(live):
    conn, run = _dual(live)
    W, _ = sp.ef_search(conn, run, Q, ef=24, T=0.0, m=3)
    anchors = {h["ord"] for h in gt.search(conn, run, Q, k=3)}
    assert anchors & set(W), "top anchor must survive into W"
    assert all(0.0 <= v <= 1.0 for v in W.values()), "bounded path scores (R15)"


def test_ef_search_temperature_changes_frontier_but_not_width(live):
    """S11: T samples WHICH neighbours expand. Output width is still ef."""
    conn, run = _dual(live)
    W0, t0 = sp.ef_search(conn, run, Q, ef=24, T=0.0, m=3)
    W1, t1 = sp.ef_search(conn, run, Q, ef=24, T=0.7, m=3, seed=1)
    assert len(W0) == len(W1) == 24
    assert t1["pools_over_m"] > 0, "T never applied -- m is above the pool size"
    assert set(W0) != set(W1), "T=0.7 explored an identical frontier to greedy"


def test_ef_search_is_a_pure_function_of_its_seed(live):
    """S1 carried over: same seed, same answer."""
    conn, run = _dual(live)
    a = sp.ef_search(conn, run, Q, ef=24, T=0.7, m=3, seed=3)
    b = sp.ef_search(conn, run, Q, ef=24, T=0.7, m=3, seed=3)
    assert a == b


def test_ef_search_reports_no_anchor_rather_than_walking_nothing(live):
    conn, run = _dual(live)
    W, t = sp.ef_search(conn, run, "quokka wombat", ef=24)
    assert W == {} and t["stop"] == "no_anchor" and t["depth"] == 0


def test_ef_search_rejects_a_zero_hop_guard(live):
    conn, run = _dual(live)
    with pytest.raises(ValueError):
        sp.ef_search(conn, run, Q, ef=24, max_hops=0)


def test_ef_evidence_bundle_carries_telemetry_and_no_second_sampling(live):
    """S7 + S12: the Bundle re-derives exactly, and W IS the evidence."""
    conn, run = _dual(live)
    b, tele = sp.ef_evidence(conn, run, Q, ef=24, T=0.0)
    assert len(b.sampled) == 24 + tele["ring"] and not b.enumerated
    assert b.params["ef"] == 24 and b.params["stop"] == tele["stop"]
    assert b.params["depth"] == tele["depth"]
    assert b.communities, "histogram over W should be non-empty"


def test_s13_ring_is_one_edge_out_from_the_top_of_w_and_never_outranks_its_parent(live):
    """Design 6.10: every ring member is a direct edge_sym neighbour of a
    top-`ring_top` member of W, was not in W, and scores <= its parent."""
    conn, run = _dual(live)
    W, _ = sp.ef_search(conn, run, Q, ef=24, T=0.0)
    extra = sp.ring(conn, run, W, top=3, per=5)
    assert extra, "the top chunks of a 24-wide walk must have unseen neighbours"
    assert not set(extra) & set(W)
    parents = sorted(W, key=lambda o: (-W[o], o))[:3]
    nb = {p: {r["ord"]: r["strength"] for r in gt.neighbors(conn, run, p, limit=200)}
          for p in parents}
    for o, sc in extra.items():
        owners = [p for p in parents if o in nb[p]]
        assert owners, f"#{o} is not a direct neighbour of any top-3 chunk"
        assert sc <= max(W[p] for p in owners) + 1e-12
    assert len(extra) <= 3 * 5


def test_s13_ef_evidence_grows_by_the_ring_and_records_it(live):
    conn, run = _dual(live)
    b0, t0 = sp.ef_evidence(conn, run, Q, ef=24, T=0.0, ring_top=0)
    b1, t1 = sp.ef_evidence(conn, run, Q, ef=24, T=0.0)
    assert t0["ring"] == 0 and len(b0.sampled) == 24
    assert t1["ring"] > 0 and len(b1.sampled) == 24 + t1["ring"]
    assert b1.params["ring"] == t1["ring"] and set(b0.sampled) <= set(b1.sampled)
    assert b1.sampled == sorted(b1.sampled, key=lambda o: (-b1.scores[o], o))


def test_w15_pathways_shape_and_dwpc_ordering(live):
    """Design 6.11: DWPC pairs are ordered; every best path stays inside the
    subgraph, joins its two anchors, repeats no node; shape numbers in range."""
    conn, run = _dual(live)
    b, _ = sp.ef_evidence(conn, run, Q, ef=24, T=0.0)
    anchors = b.sampled[:4]
    pw = gt.pathways(conn, run, b.sampled, anchors)
    assert pw["n"] == len(b.sampled) and pw["components"] >= 1
    assert sum(pw["wcc_sizes"]) == pw["n"] and pw["wcc_sizes"] == sorted(pw["wcc_sizes"], reverse=True)
    assert 0.0 < pw["largest_component_frac"] <= 1.0
    assert 0.0 <= pw["density"] <= 1.0 and 0.0 <= pw["conductance"] <= 1.0
    assert pw["pairs"], "top-4 walk chunks should be connected"
    scores = [p["dwpc"] for p in pw["pairs"]]
    assert scores == sorted(scores, reverse=True)
    S = set(b.sampled)
    for p in pw["pairs"]:
        assert p["path"][0] == p["a"] and p["path"][-1] == p["b"]
        assert set(p["path"]) <= S and len(p["path"]) == len(set(p["path"]))
        assert 2 <= len(p["path"]) <= 4                      # <= 3 edges


def test_w15_hub_damping_costs_every_path(live):
    """The point of damping: with damp=0 every pair scores strictly higher
    than with damp=0.4, and the induced degrees actually vary."""
    conn, run = _dual(live)
    b, _ = sp.ef_evidence(conn, run, Q, ef=24, T=0.0)
    deg = gt.degrees(conn, run, b.sampled)
    assert set(deg) == set(b.sampled) and max(deg.values()) > min(deg.values())
    d1 = {(p["a"], p["b"]): p["dwpc"]
          for p in gt.pathways(conn, run, b.sampled, b.sampled[:6], damp=0.4, top_pairs=99)["pairs"]}
    d0 = {(p["a"], p["b"]): p["dwpc"]
          for p in gt.pathways(conn, run, b.sampled, b.sampled[:6], damp=0.0, top_pairs=99)["pairs"]}
    common = set(d1) & set(d0)
    assert common and all(d1[k] < d0[k] for k in common)


def test_w16_best_path_is_valid_and_damping_prices_hubs(live):
    """Every hop of the returned path is a real live edge; the weight is in
    (0,1]; and raising damp never RAISES a path's weight."""
    conn, run = _dual(live)
    b, _ = sp.ef_evidence(conn, run, Q, ef=24, T=0.0, bridge_pairs=0)
    a_, b_ = b.sampled[0], b.sampled[-1]
    r = gt.best_path(conn, run, a_, b_, damp=0.4)
    assert r is not None
    path, w = r
    assert path[0] == a_ and path[-1] == b_ and 0.0 < w <= 1.0
    adj = gt.full_adjacency(conn, run)
    for u, v in zip(path, path[1:]):
        assert v in adj[u], f"{u}->{v} is not a live edge"
    r0 = gt.best_path(conn, run, a_, b_, damp=0.0)
    assert r0 is not None and r0[1] >= w


def test_s14_bridges_are_discovered_off_walk_and_never_outrank_endpoints(live):
    """S14: every bridge chunk sits on a reported whole-graph path between two
    top walk chunks, was NOT in the walk+ring, and scores <= both endpoints.
    bridge_pairs=0 disables the stage entirely."""
    conn, run = _dual(live)
    b0, t0 = sp.ef_evidence(conn, run, Q, ef=24, T=0.0, bridge_pairs=0)
    assert t0["bridge"] == 0 and set(b0.origin.values()) <= {"walk", "ring"}
    b1, t1 = sp.ef_evidence(conn, run, Q, ef=24, T=0.0)
    assert set(b1.origin) == set(b1.sampled)
    br = [o for o in b1.sampled if b1.origin[o] == "bridge"]
    assert len(br) == t1["bridge"] <= sp.MAX_BRIDGE
    base = set(b0.sampled)
    top = sorted({o: b1.scores[o] for o in b1.sampled if b1.origin[o] == "walk"},
                 key=lambda o: -b1.scores[o])[:sp.DEFAULT_BRIDGE_PAIRS]
    for o in br:
        assert o not in base, "a bridge must be NEW evidence"
        homes = [p for p in t1["bridge_paths"] if o in p]
        assert homes, f"#{o} is on no reported bridge path"
        for p in homes:
            assert p[0] in top and p[-1] in top
        assert b1.scores[o] <= min(b1.scores[p[0]], b1.scores[p[-1]]) + 1e-12


# --------------------------- select_anchors: additive-only extras (S15/S16)
# no DB

def _row(ord_, score, source):
    return {"ord": ord_, "score": score, "source": source}


def test_a1_shape_one_source_dominates_extras_denied():
    """Regression for the A1/E3 flip: 99% one source, minority hits well under
    the competitive threshold. The old sqrt scheme would have capped wiki;
    the amendment must leave base untouched."""
    base = [_row(28410, 9.0, "wiki"), _row(16278, 8.0, "wiki"), _row(9001, 7.0, "wiki")]
    pool = base + [_row(50, 1.0, "quotes"), _row(51, 0.9, "brown")]
    picked = sp.select_anchors(base, pool, {"wiki": 3, "quotes": 1, "brown": 1}, ef=24)
    assert picked is base
    assert {28410, 16278, 9001} <= {h["ord"] for h in picked}
    assert picked[0]["ord"] == 28410


def test_granted_exactly_at_the_boundary():
    base = [_row(1, 9.0, "wiki"), _row(2, 8.0, "wiki"), _row(3, 7.0, "wiki")]
    pool = base + [_row(50, 3.5, "quotes")]           # == 0.5 * 7.0, inclusive
    picked = sp.select_anchors(base, pool, {"wiki": 3, "quotes": 1}, ef=24)
    assert len(picked) == 4
    assert [h["ord"] for h in picked[:3]] == [1, 2, 3]


def test_denied_just_under_the_boundary():
    base = [_row(1, 9.0, "wiki"), _row(2, 8.0, "wiki"), _row(3, 7.0, "wiki")]
    pool = base + [_row(50, 3.5 - 1e-9, "quotes")]
    picked = sp.select_anchors(base, pool, {"wiki": 3, "quotes": 1}, ef=24)
    assert picked is base


def test_byte_identity_multi_source_nobody_competitive():
    base = [_row(1, 9.0, "wiki"), _row(2, 8.0, "wiki"), _row(3, 7.0, "wiki")]
    pool = base + [_row(50, 1.0, "quotes"), _row(51, 0.5, "brown")]
    picked = sp.select_anchors(base, pool, {"wiki": 3, "quotes": 1, "brown": 1}, ef=24)
    assert picked == base
    assert picked is base


def test_one_extra_per_source_maximum():
    base = [_row(1, 9.0, "wiki")]
    pool = base + [_row(50, 8.0, "quotes"), _row(51, 7.0, "quotes"), _row(52, 6.0, "quotes"),
                   _row(60, 5.0, "brown"), _row(61, 4.0, "brown")]
    picked = sp.select_anchors(base, pool, {"wiki": 1, "quotes": 1, "brown": 1}, ef=24)
    assert len(picked) == 3
    assert sum(1 for h in picked if h["source"] == "quotes") == 1
    assert sum(1 for h in picked if h["source"] == "brown") == 1
    assert 50 in [h["ord"] for h in picked]           # quotes' best, not runners-up


def test_extras_never_displace_a_base_anchor_under_the_ef_clamp():
    base = [_row(1, 9.0, "wiki"), _row(2, 8.0, "wiki"), _row(3, 7.0, "wiki")]
    pool = base + [_row(50, 8.0, "quotes")]
    picked = sp.select_anchors(base, pool, {"wiki": 3, "quotes": 1}, ef=3)
    assert picked is base
    assert len(picked) == 3

    picked2 = sp.select_anchors(base, pool, {"wiki": 3, "quotes": 1}, ef=4)
    assert len(picked2) == 4
    assert {1, 2, 3} <= {h["ord"] for h in picked2}


def test_tie_at_the_kth_score_sorts_ahead_no_base_row_dropped():
    base = [_row(10, 9.0, "wiki"), _row(11, 8.0, "wiki"), _row(12, 7.0, "wiki")]
    pool = base + [_row(1, 7.0, "quotes")]             # tie on score, lower ord
    picked = sp.select_anchors(base, pool, {"wiki": 3, "quotes": 1}, ef=24)
    assert len(picked) == 4
    ords = [h["ord"] for h in picked]
    assert ords == [10, 11, 1, 12]                     # tie: ord 1 sorts ahead of ord 12
    assert {10, 11, 12} <= set(ords)                   # no base row dropped


def test_a_source_already_in_base_gets_no_extra():
    base = [_row(1, 9.0, "wiki"), _row(2, 8.0, "quotes")]
    pool = base + [_row(50, 7.5, "quotes")]            # quotes already present in base
    picked = sp.select_anchors(base, pool, {"wiki": 1, "quotes": 1}, ef=24)
    assert picked is base


def test_determinism_independent_of_sources_dict_order():
    base = [_row(1, 9.0, "wiki")]
    pool = base + [_row(50, 8.0, "quotes"), _row(60, 7.0, "brown")]
    sources = {"wiki": 3, "quotes": 1, "brown": 1}
    a = sp.select_anchors(base, pool, sources, ef=24)
    b = sp.select_anchors(base, pool, dict(reversed(list(sources.items()))), ef=24)
    assert a == b


def test_empty_base_returns_base():
    assert sp.select_anchors([], [], {"wiki": 1}, ef=24) == []


def test_merged_list_is_score_ordered_and_deduped():
    base = [_row(1, 9.0, "a"), _row(3, 7.0, "a"), _row(5, 5.0, "a")]
    pool = base + [_row(2, 8.0, "b")]
    picked = sp.select_anchors(base, pool, {"a": 1, "b": 1}, ef=24)
    assert picked == sorted(picked, key=lambda r: (-r["score"], r["ord"]))
    assert len(picked) == len({h["ord"] for h in picked})


def test_starved_pool_grants_what_it_can_rather_than_raising():
    """Fewer competitive candidates than missing sources: no raise, just fewer
    extras than sources eligible."""
    base = [_row(1, 9.0, "a")]
    pool = base + [_row(2, 8.0, "b")]                  # "c" never appears in pool
    picked = sp.select_anchors(base, pool, {"a": 1, "b": 1, "c": 1}, ef=24)
    assert len(picked) == 2
    assert {1, 2} == {h["ord"] for h in picked}


def test_unlabelled_row_never_wins_an_extra_but_is_not_barred_from_base():
    base = [_row(1, 9.0, None), _row(2, 8.0, "a")]     # unlabelled row IS a base anchor
    pool = base + [_row(50, 8.5, None)]                # unlabelled pool row cannot be an extra
    picked = sp.select_anchors(base, pool, {"a": 1, "b": 1}, ef=24)
    assert picked is base                              # no pool row is labelled "b"
    assert 1 in [h["ord"] for h in picked]


# ---------------------------------------------------- anchor_hits pipeline
# uses `live`, skips without a DB

def _multi(live):
    conn, _ = live
    for label in ("mixed-full-dual",):
        try:
            run = gt.get_run(conn, label)
        except Exception:
            continue
        if len(gt.run_sources(conn, run)) >= 2:
            return conn, run
    pytest.skip("no multi-source run")


def test_single_source_anchors_are_identical_to_the_global_search(live):
    conn, run = _dual(live)
    if len(gt.run_sources(conn, run)) >= 2:
        pytest.skip("brown-500-dual reports >= 2 source labels")
    got = [h["ord"] for h in sp.anchor_hits(conn, run, Q, 3, 24)]
    want = [h["ord"] for h in gt.search(conn, run, Q, k=3)]
    assert got == want
    _, tele = sp.ef_search(conn, run, Q, ef=24, T=0.0)
    assert tele["anchors"] == want


def test_multi_source_anchors_start_with_the_global_top_k(live):
    conn, run = _multi(live)
    sources = gt.run_sources(conn, run)
    elig = {s: n for s, n in sources.items() if n > 0}
    base = gt.search(conn, run, Q, k=3)
    hits = sp.anchor_hits(conn, run, Q, 3, 24)
    assert hits[:len(base)] == base
    assert len(hits) <= 3 + (len(elig) - 1)


def test_ef_search_anchors_never_exceed_ef(live):
    conn, run = _multi(live)
    _, tele = sp.ef_search(conn, run, Q, ef=2, T=0.0, k_anchor=3)
    assert len(tele["anchors"]) <= 2


# ---------------------------------------------- select_ring / ring (S17)
# no DB for select_ring; ring's live tests reuse `live` and `_multi`

def _rrow(ord_, score, source):
    return {"ord": ord_, "score": score, "source": source}


def test_s17_proportional_split():
    """Design acceptance (a): anchors {wiki 2, quotes 1}, budget 24 -> 16/8."""
    pool = [_rrow(i, 100.0 - i, "wiki") for i in range(20)] + \
           [_rrow(1000 + i, 90.0 - i, "quotes") for i in range(20)]
    chosen = sp.select_ring(pool, {"wiki": 2, "quotes": 1}, budget=24)
    assert len(chosen) == 24
    assert sum(1 for r in chosen if r["source"] == "wiki") == 16
    assert sum(1 for r in chosen if r["source"] == "quotes") == 8


def test_s17_largest_remainder_tie_breaks_by_name():
    """mix={wiki:2, quotes:1, brown:1}, B=10 -> floors 5/2/2=9, one leftover,
    quotes/brown tie on remainder and anchor count -> name-ascending -> brown."""
    pool = [_rrow(i, 100.0 - i, "wiki") for i in range(10)] + \
           [_rrow(200 + i, 50.0 - i, "quotes") for i in range(10)] + \
           [_rrow(300 + i, 40.0 - i, "brown") for i in range(10)]
    chosen = sp.select_ring(pool, {"wiki": 2, "quotes": 1, "brown": 1}, budget=10)
    counts = {}
    for r in chosen:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    assert counts == {"wiki": 5, "brown": 3, "quotes": 2}


def test_s17_steering_beats_global_strength_fill():
    """Every quotes candidate scores below the global top-24: a plain
    global-strength fill returns 0 quotes, select_ring still returns >= 8."""
    pool = [_rrow(i, 1000.0 - i, "wiki") for i in range(30)] + \
           [_rrow(2000 + i, 10.0 - i * 0.1, "quotes") for i in range(10)]
    global_fill = sorted(pool, key=lambda r: (-r["score"], r["ord"]))[:24]
    assert sum(1 for r in global_fill if r["source"] == "quotes") == 0
    chosen = sp.select_ring(pool, {"wiki": 2, "quotes": 1}, budget=24)
    assert sum(1 for r in chosen if r["source"] == "quotes") >= 8


def test_s17_shortfall_forfeits_to_global_strength_order():
    """quotes quota 8 but only 3 quotes candidates exist -- all 3 taken, the
    other 5 slots go to the strongest remaining rows globally; the walk does
    not shrink."""
    pool = [_rrow(i, 100.0 - i, "wiki") for i in range(30)] + \
           [_rrow(500 + i, 50.0 - i, "quotes") for i in range(3)]
    chosen = sp.select_ring(pool, {"wiki": 1, "quotes": 1}, budget=24)
    assert len(chosen) == min(24, len(pool))
    assert sum(1 for r in chosen if r["source"] == "quotes") == 3
    assert {500, 501, 502} <= {r["ord"] for r in chosen}


def test_s17_unlabelled_and_out_of_mix_rows_only_eligible_in_forfeit():
    """wiki's own quota (budget=3, mix={wiki:1} -> quota 3) outruns its 2
    candidates; the shortfall forfeits to the strongest remaining row
    regardless of label -- the unlabelled #2 (score 8.0) beats brown's #3
    (score 7.0), so it is picked before brown even though neither is 'wiki'."""
    pool = [_rrow(1, 9.0, "wiki"), _rrow(2, 8.0, None), _rrow(3, 7.0, "brown"),
            _rrow(4, 6.0, "wiki")]
    chosen = sp.select_ring(pool, {"wiki": 1}, budget=3)
    assert {r["ord"] for r in chosen} == {1, 4, 2}   # wiki's 2, then forfeit's best
    chosen_full = sp.select_ring(pool, {"wiki": 1}, budget=4)
    assert {r["ord"] for r in chosen_full} == {1, 2, 3, 4}


def test_s17_order_independence():
    import random
    pool = [_rrow(i, float(i % 13), ("wiki" if i % 2 else "quotes")) for i in range(40)]
    shuffled = pool[:]
    random.Random(11).shuffle(shuffled)
    a = sp.select_ring(pool, {"wiki": 2, "quotes": 1}, budget=17)
    b = sp.select_ring(shuffled, {"wiki": 2, "quotes": 1}, budget=17)
    assert a == b


def test_s17_degenerate_inputs():
    assert sp.select_ring([], {"wiki": 1}, budget=10) == []
    pool = [_rrow(1, 9.0, "wiki")]
    assert sp.select_ring(pool, {"wiki": 1}, budget=0) == []
    assert sp.select_ring(pool, {"wiki": 1}, budget=5) == pool


def test_s17_ring_byte_identity_when_mix_is_single_source(live):
    """T8: the dominant path. mix with one source must reproduce today's loop
    exactly -- same dict, same key order."""
    conn, run = _dual(live)
    W, _ = sp.ef_search(conn, run, Q, ef=24, T=0.0)
    a = sp.ring(conn, run, W, top=3, per=8)
    b = sp.ring(conn, run, W, top=3, per=8, mix={"wiki": 3})
    assert a == b
    assert list(a) == list(b)


def test_s17_ef_evidence_identity_when_anchor_mix_below_two(live):
    """T9: when tele['anchor_mix'] has fewer than 2 entries, the ring-origin
    members must equal ring() recomputed with no mix, and anchor_mix must be
    present in both tele and the bundle params."""
    conn, run = _dual(live)
    b, tele = sp.ef_evidence(conn, run, Q, ef=24, T=0.0)
    assert "anchor_mix" in tele and "anchor_mix" in b.params
    if len(tele.get("anchor_mix") or {}) >= 2:
        pytest.skip("brown-500-dual reports >= 2 anchor sources")
    ring_ords = {o for o, origin in b.origin.items() if origin == "ring"}
    W, _ = sp.ef_search(conn, run, Q, ef=24, T=0.0)
    want = sp.ring(conn, run, W, top=sp.DEFAULT_RING_TOP, per=sp.DEFAULT_RING_PER)
    assert ring_ords == set(want)


def test_s17_live_steering_invariant(live):
    """T10: for each source in the anchor mix, chosen ring count >= its quota
    OR the pool held fewer of that source than its quota."""
    conn, run = _multi(live)
    W, tele = sp.ef_search(conn, run, Q, ef=24, T=0.0)
    mix = tele.get("anchor_mix") or {}
    if len(mix) < 2:
        pytest.skip("no probe query yielded an anchor mix >= 2")

    budget = sp.DEFAULT_RING_TOP * sp.DEFAULT_RING_PER
    A = sum(mix.values())
    parents = sorted(W, key=lambda o: (-W[o], o))[:sp.DEFAULT_RING_TOP]
    pool_counts: dict = {}
    for p in parents:
        for nb in gt.neighbors(conn, run, p, limit=sp.DEFAULT_RING_PER * 3):
            if nb["ord"] in W:
                continue
            src = gt.source_of(nb)
            pool_counts[src] = pool_counts.get(src, 0) + 1

    extra = sp.ring(conn, run, W, top=sp.DEFAULT_RING_TOP, per=sp.DEFAULT_RING_PER, mix=mix)
    chosen_counts: dict = {}
    # Re-derive each chosen ord's source by re-labelling via its parents.
    labelled: dict = {}
    for p in parents:
        for nb in gt.neighbors(conn, run, p, limit=sp.DEFAULT_RING_PER * 3):
            if nb["ord"] in extra and nb["ord"] not in labelled:
                labelled[nb["ord"]] = gt.source_of(nb)
    for o in extra:
        src = labelled.get(o)
        chosen_counts[src] = chosen_counts.get(src, 0) + 1

    for s, n_s in mix.items():
        quota = int(budget * n_s / A)  # floor; largest-remainder can only add
        held = pool_counts.get(s, 0)
        assert chosen_counts.get(s, 0) >= min(quota, held) or held < quota
