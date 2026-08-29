"""Exercises graph_tools' traversal primitives against a live Postgres run.

These are integration tests by design: the whole point of the tool surface is
that it is index-safe and read-only *at the server*, and neither property can be
tested against a mock. Assertions are pinned to the `brown-50` run's measured
shape, so a drift in ingest shows up here rather than in the UI.

Run:  pytest tests/test_graph_tools.py -v      (needs `docker compose up -d`)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

import graph_tools as gt

LABEL = "brown-50"

# Measured off the live run. Regenerate deliberately if brown-50 is re-ingested.
N_CHUNKS = 1789
N_EDGES = 4568
N_COMMUNITIES = 40
HUB = 1476          # joint-maximum degree node
HUB_DEGREE = 28


@pytest.fixture(scope="module")
def conn():
    try:
        c = gt.connect()
    except psycopg.OperationalError as e:                # pragma: no cover
        pytest.skip(f"no database: {e}")
    yield c
    c.close()


@pytest.fixture(scope="module")
def run(conn):
    try:
        return gt.get_run(conn, LABEL)
    except LookupError:                                   # pragma: no cover
        pytest.skip(f"no live run {LABEL!r}")


# ---------------------------------------------------------------- W3
def test_connection_is_read_only_at_the_server(conn):
    """W3: read-only is enforced by Postgres, not by convention. This is the
    guard that makes the tool surface safe to hand an autonomous caller."""
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE should_never_exist (x int)")


# ---------------------------------------------------------------- runs
def test_get_run_reports_measured_shape(run):
    assert run.label == LABEL
    assert run.n_chunks == N_CHUNKS
    assert run.n_edges == N_EDGES
    assert run.n_communities == N_COMMUNITIES


def test_get_run_reports_sparse_only_capability(run):
    """W5: brown-50 has embed_dim NULL and zero embedding rows, so the run must
    describe itself as sparse-only rather than pretending to be fused."""
    assert run.embed_dim is None
    assert run.dense is False
    assert run.mode == "sparse-only"


def test_single_provenance_is_stated(run):
    """R5.2: every edge carries one value, so say so."""
    assert run.provenance == {"sparse": N_EDGES}
    assert run.single_provenance == "sparse"


def test_get_run_raises_on_unknown_label(conn):
    with pytest.raises(LookupError):
        gt.get_run(conn, "no-such-label-exists")


# ---------------------------------------------------------------- tokenize
def test_tokenize_drops_stopwords_and_short_tokens():
    assert gt.tokenize("The Quick fox of a To be") == ["quick", "fox"]


def test_tokenize_empty_query_yields_nothing():
    assert gt.tokenize("the of and a") == []


def test_search_with_no_usable_terms_returns_empty(conn, run):
    assert gt.search(conn, run, "the of and", k=5) == []


# ---------------------------------------------------------------- search
def test_search_returns_ranked_hits(conn, run):
    hits = gt.search(conn, run, "jury election county", k=5)
    assert 0 < len(hits) <= 5
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    for h in hits:
        assert h["terms_hit"] >= 1
        assert h["doc_id"]


# ---------------------------------------------------------------- neighbors
def test_neighbors_returns_hub_degree(conn, run):
    """W4: the bound is explicit; asked for more than the degree, we get the
    degree. 1476 is one of three joint-maximum-degree nodes."""
    nbrs = gt.neighbors(conn, run, HUB, limit=100)
    assert len(nbrs) == HUB_DEGREE


def test_neighbors_ordered_by_strength_desc(conn, run):
    nbrs = gt.neighbors(conn, run, HUB, limit=100)
    s = [n["strength"] for n in nbrs]
    assert s == sorted(s, reverse=True)


def test_neighbors_respects_limit(conn, run):
    assert len(gt.neighbors(conn, run, HUB, limit=5)) == 5


def test_neighbors_respects_strength_floor(conn, run):
    """The floor the cookbook hard-codes at 0.05 is a parameter here."""
    floor = 0.3
    nbrs = gt.neighbors(conn, run, HUB, limit=100, min_strength=floor)
    assert all(n["strength"] >= floor for n in nbrs)
    assert len(nbrs) < HUB_DEGREE


def test_neighbors_never_returns_the_seed(conn, run):
    assert all(n["ord"] != HUB for n in gt.neighbors(conn, run, HUB, limit=100))


# ---------------------------------------------------------------- communities
def test_communities_touched_groups_by_stored_cid(conn, run):
    nbrs = gt.neighbors(conn, run, HUB, limit=100)
    ords = [HUB] + [n["ord"] for n in nbrs]
    touched = gt.communities_touched(conn, run, ords)
    assert touched
    hits = [t["hits"] for t in touched]
    assert hits == sorted(hits, reverse=True)
    assert sum(hits) <= len(ords)          # nodes with no community are dropped
    for t in touched:
        assert t["hits"] <= t["size"]


def test_communities_touched_empty_input(conn, run):
    assert gt.communities_touched(conn, run, []) == []


def test_community_lookup_matches_node_membership(conn, run):
    n = gt.node(conn, run, HUB)
    c = gt.community(conn, run, n["cid"])
    assert c["cid"] == n["cid"]
    assert c["size"] >= 5                  # min_size at ingest
    assert list(c["keywords"]) == list(n["keywords"])


# ---------------------------------------------------------------- quotient
def test_quotient_pairs_are_canonical_and_cross_community(conn, run):
    q = gt.quotient(conn, run, limit=25)
    assert q
    for r in q:
        assert r["cid_a"] < r["cid_b"]     # least/greatest normalisation
        assert r["edges"] >= 1
    counts = [r["edges"] for r in q]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------- subgraph
def test_subgraph_edges_needs_two_nodes(conn, run):
    assert gt.subgraph_edges(conn, run, [HUB]) == []
    assert gt.subgraph_edges(conn, run, []) == []


def test_subgraph_edges_are_induced(conn, run):
    nbrs = gt.neighbors(conn, run, HUB, limit=10)
    ords = [HUB] + [n["ord"] for n in nbrs]
    edges = gt.subgraph_edges(conn, run, ords)
    assert edges
    for e in edges:
        assert e["src"] in ords and e["dst"] in ords
        assert e["src"] < e["dst"]         # edge_canonical CHECK


def test_node_returns_body_and_attribution(conn, run):
    n = gt.node(conn, run, HUB)
    assert n["ord"] == HUB
    assert n["doc_id"]                     # R8: every chunk names its source
    assert len(n["body"]) > 0
    assert n["n_tok"] > 0


def test_node_missing_ordinal_returns_none(conn, run):
    assert gt.node(conn, run, 10**7) is None


# ------------------------------------------------------- walk() evidence (W7)


def test_walk_carries_a_provenance_edge_per_hop(conn, run):
    """W7: a route must name its own justification, not just its node ids."""
    rows = gt.walk(conn, run, HUB, hops=2, cap=25)
    assert rows, "hub should reach something"
    for r in rows:
        assert len(r["node_path"]) == r["hop"] + 1
        assert len(r["doc_path"]) == r["hop"] + 1
        assert len(r["prov_path"]) == r["hop"], (
            "one provenance value per traversed EDGE")


def test_walk_prov_path_is_a_list_not_an_enum_string(conn, run):
    """psycopg has no loader for edge_provenance; uncast it returns '{sparse}'.

    That string iterates as characters, so a caller zipping it against
    node_path corrupts silently rather than raising. Pinned deliberately.
    """
    r = gt.walk(conn, run, HUB, hops=1, cap=1)[0]
    assert isinstance(r["prov_path"], list), f"got {type(r['prov_path'])}"
    assert r["prov_path"] == ["sparse"], "brown-50 is sparse-only"


def test_walk_path_endpoints_are_seed_and_result(conn, run):
    rows = gt.walk(conn, run, HUB, hops=2, cap=25)
    for r in rows:
        assert r["node_path"][0] == HUB
        assert r["node_path"][-1] == r["ord"]
        assert r["doc_path"][-1] == r["doc_id"]


def test_walk_never_revisits_a_node_on_one_path(conn, run):
    rows = gt.walk(conn, run, HUB, hops=2, cap=25)
    for r in rows:
        assert len(set(r["node_path"])) == len(r["node_path"]), "cycle in path"


def test_walk_never_returns_the_seed(conn, run):
    rows = gt.walk(conn, run, HUB, hops=2, cap=50)
    assert HUB not in [r["ord"] for r in rows]


def test_walk_keeps_the_best_scoring_path_per_node(conn, run):
    rows = gt.walk(conn, run, HUB, hops=2, cap=50)
    ords = [r["ord"] for r in rows]
    assert len(ords) == len(set(ords)), "one row per reached node"
    assert rows == sorted(rows, key=lambda r: -r["score"])


def test_walk_hop_1_matches_neighbors_above_the_floor(conn, run):
    """The walk reports the graph's own traversal, it does not define a new one."""
    floor = 0.05
    nb = {n["ord"] for n in gt.neighbors(conn, run, HUB, limit=100)
          if n["strength"] > floor}
    w1 = {r["ord"] for r in gt.walk(conn, run, HUB, hops=1,
                                    min_score=floor, cap=200)}
    assert w1 == nb


def test_walk_cross_doc_flag_agrees_with_the_doc_path(conn, run):
    rows = gt.walk(conn, run, HUB, hops=2, cap=25)
    for r in rows:
        assert r["cross_doc"] == (r["doc_path"][0] != r["doc_path"][-1])


def test_walk_respects_cap_and_score_floor(conn, run):
    assert len(gt.walk(conn, run, HUB, hops=2, cap=3)) <= 3
    assert gt.walk(conn, run, HUB, hops=2, min_score=0.99, cap=50) == []


def test_walk_rejects_zero_hops(conn, run):
    with pytest.raises(ValueError):
        gt.walk(conn, run, HUB, hops=0)


def test_why_renders_every_hop_and_the_doc_lineage(conn, run):
    r = gt.walk(conn, run, HUB, hops=2, cap=25)[0]
    line = gt.why(r)
    assert line.count("-->") == r["hop"], "one arrow per traversed edge"
    for p in r["prov_path"]:
        assert f"--{p}-->" in line
    assert " > ".join(r["doc_path"]) in line


# ------------------------------------------------------- term_stats (W8)


def test_term_stats_reports_absent_terms_rather_than_dropping_them(conn, run):
    """W8: a df-0 term is the usual reason a query looks broken. Keep it."""
    rows = gt.term_stats(conn, run, "jury quokka")
    terms = {r["term"]: r for r in rows}
    assert "quokka" in terms, "absent term was silently dropped"
    assert terms["quokka"]["df"] == 0
    assert terms["jury"]["df"] > 0


def test_term_stats_orders_rarest_first(conn, run):
    rows = gt.term_stats(conn, run, "jury trial investigation grand")
    dfs = [r["df"] for r in rows]
    assert dfs == sorted(dfs), "most discriminating term should read first"


def test_term_stats_drops_stopwords_like_search_does(conn, run):
    assert gt.term_stats(conn, run, "the of and to") == []


def test_term_stats_hits_are_confined_to_the_supplied_ords(conn, run):
    scope = [n["ord"] for n in gt.neighbors(conn, run, HUB, limit=10)]
    for r in gt.term_stats(conn, run, "jury trial investigation", scope):
        assert set(r["hits"]) <= set(scope)


def test_term_stats_hits_agree_with_the_stored_tf_map(conn, run):
    scope = [n["ord"] for n in gt.neighbors(conn, run, HUB, limit=25)]
    rows = gt.term_stats(conn, run, "jury trial investigation report", scope)
    for r in rows:
        for o in r["hits"]:
            assert r["term"] in gt.node(conn, run, o)["tf"], (
                f"{r['term']} claimed in #{o} but absent from its tf map")


def test_term_stats_separates_lexical_from_graph_reached(conn, run):
    """The whole point: chunks with NO query term came from the graph.

    If every sampled chunk carried a query term, the graph added nothing over
    plain lexical search and the walk would be decoration.
    """
    import sampler
    b = sampler.evidence(conn, run, "jury trial grand jury investigation")
    rows = gt.term_stats(conn, run, "jury trial grand jury investigation",
                         b.sampled)
    lex = {o for r in rows for o in r["hits"]}
    graph_only = set(b.sampled) - lex
    assert lex, "some evidence should be lexically anchored"
    assert graph_only, "graph contributed nothing beyond lexical hits"


# ------------------------------------------- persisted strength is bounded (R15)


@pytest.mark.parametrize("label", ["brown-50", "brown-50-dual"])
def test_persisted_strength_is_a_bounded_decay_weight(conn, label):
    """R15: every consumer multiplies path score by strength, so it must sit in
    [0, 1] or path scores amplify with depth and the walk never converges.

    Measured before the fix on the fused run: strength in [-6.53, 3.55], 58% of
    edges over 1.0, ef_search depth == max_hops at every cap.
    """
    try:
        run = gt.get_run(conn, label)
    except Exception as e:                                # pragma: no cover
        pytest.skip(f"no run {label}: {e}")
    with conn.cursor() as cur:
        cur.execute("""SELECT min(strength), max(strength),
                              count(*) FILTER (WHERE strength > 1.0 OR strength < 0.0)
                         FROM edge WHERE run_id = %s AND valid_to IS NULL""",
                    (run.run_id,))
        lo, hi, out_of_range = cur.fetchone().values()
    assert out_of_range == 0, f"{label}: {out_of_range} edges outside [0,1] ({lo:.3f}..{hi:.3f})"
    assert 0.0 <= lo and hi <= 1.0
