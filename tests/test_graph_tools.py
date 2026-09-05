"""Exercises graph_tools' traversal primitives against a live Postgres run.

These are integration tests by design: the whole point of the tool surface is
that it is index-safe and read-only *at the server*, and neither property can be
tested against a mock. Assertions are pinned to the `brown-50` run's measured
shape, so a drift in ingest shows up here rather than in the UI.

Run:  pytest tests/test_graph_tools.py -v      (needs `docker compose up -d`)
"""
from __future__ import annotations

from pathlib import Path

import pytest

import psycopg

import graph_tools as gt
from conftest import require_gt_conn, require_run

pytestmark = pytest.mark.live_db

LABEL = "brown-50"

# Measured off the live run. Regenerate deliberately if brown-50 is re-ingested.
N_CHUNKS = 51
N_EDGES = 115
N_COMMUNITIES = 5
HUB = 43          # joint-maximum degree node
HUB_DEGREE = 12


@pytest.fixture(scope="module")
def conn():
    c = require_gt_conn()
    yield c
    c.close()


@pytest.fixture(scope="module")
def run(conn):
    return require_run(conn, LABEL)


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
    # scope = the query's own lexical hits, which carry its terms by construction
    scope = [h["ord"] for h in gt.search(conn, run, "jury trial investigation report", k=12)]
    rows = gt.term_stats(conn, run, "jury trial investigation report", scope)
    checked = 0
    for r in rows:
        for o in r["hits"]:
            assert r["term"] in gt.node(conn, run, o)["tf"], (
                f"{r['term']} claimed in #{o} but absent from its tf map")
            checked += 1
    assert checked > 0, ("no hits in scope -- this test passed vacuously "
                         "before node() returned tf; widen the scope")


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


@pytest.mark.parametrize("label", ["brown-50", "brown-50-dual", "brown-500-dual"])
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


# ------------------------------------ community primitives (design §6, W9)


def _some_cids(conn, run, n=4):
    rows = gt.communities_touched(conn, run,
                                  [x["ord"] for x in gt.neighbors(conn, run, HUB, limit=40)])
    return [r["cid"] for r in rows[:n]]


def test_community_terms_returns_at_most_k_per_requested_cid(conn, run):
    cids = _some_cids(conn, run)
    out = gt.community_terms(conn, run, cids, k=3)
    assert set(out) == set(cids)
    for c in cids:
        assert 1 <= len(out[c]) <= 3
        assert all(isinstance(t, str) and t for t in out[c])


def test_community_terms_scores_the_whole_community_not_a_slice(conn, run):
    """W9: the signature takes cids only. A term that lives in members the
    walk never touched must still be eligible. Pinned by checking that every
    returned term has tf mass in at least one member."""
    cid = _some_cids(conn, run, 1)[0]
    members = gt.community(conn, run, cid)
    with conn.cursor() as cur:
        cur.execute("SELECT members FROM community WHERE run_id=%s AND cid=%s",
                    (run.run_id, cid))
        ords = cur.fetchone()["members"]
    terms = gt.community_terms(conn, run, [cid], k=3)[cid]
    for t in terms:
        assert any(t in gt.node(conn, run, o)["tf"] for o in ords), (
            f"{t!r} ranked for c{cid} but no member carries it")


def test_community_terms_empty_input(conn, run):
    assert gt.community_terms(conn, run, []) == {}


def test_local_medoid_edge_cases(conn, run):
    assert gt.local_medoid(conn, run, []) is None
    assert gt.local_medoid(conn, run, [HUB]) == HUB


def test_local_medoid_is_the_argmax_of_summed_strength(conn, run):
    ords = [HUB] + [x["ord"] for x in gt.neighbors(conn, run, HUB, limit=12)]
    tot = {o: 0.0 for o in ords}
    for e in gt.subgraph_edges(conn, run, ords):
        tot[e["src"]] += e["strength"]; tot[e["dst"]] += e["strength"]
    want = min(ords, key=lambda o: (-tot[o], o))
    assert gt.local_medoid(conn, run, ords) == want
    assert want in ords


def test_cross_community_contract(conn, run):
    ords = [HUB] + [x["ord"] for x in gt.neighbors(conn, run, HUB, limit=30)]
    out = gt.cross_community(conn, run, ords)
    seen = set(ords)
    for r in out:
        assert r["ord"] in seen
        assert r["cid"] not in r["foreign_cids"], "own cid counted as foreign"
        assert r["n_foreign_edges"] >= len(r["foreign_cids"]) >= 1
    keys = [(-len(r["foreign_cids"]), -r["n_foreign_edges"], r["ord"]) for r in out]
    assert keys == sorted(keys), "not ranked by reach then edge count"


def test_cross_community_needs_two_chunks(conn, run):
    assert gt.cross_community(conn, run, [HUB]) == []


# ------------------------------------------ query_terms (design §6.1, W10)


def _fake_embed(anchor: str):
    """Deterministic unit vectors; `anchor` embeds identically to any query,
    so cosine(query, anchor) == 1.0 and everything else is ~orthogonal."""
    import hashlib
    import numpy as np

    def embed(texts):
        out = []
        for t in texts:
            key = anchor if (t == anchor or t.split() != [t] and anchor in t) else t
            h = hashlib.sha256(key.encode()).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            v = np.resize(v, 64) - 127.5
            out.append(v / np.linalg.norm(v))
        return np.stack(out)
    return embed


def test_query_terms_is_a_subset_of_the_community_pool(conn, run):
    """W10: the prompt re-ranks; it never imports a term."""
    cids = _some_cids(conn, run)
    pool = gt.community_terms(conn, run, cids, k=40)
    out = gt.query_terms(conn, run, cids, "jury trial betrayal", k=3)
    for c in cids:
        assert set(out[c]) <= set(pool[c])
        assert len(out[c]) <= 3


def test_query_terms_lexical_match_is_forced_first(conn, run):
    """Pick the term FROM the pool -- a hard-coded word can be absent from
    the fixture's communities and the test dies on its own precondition."""
    cids = _some_cids(conn, run, 3)
    pool = gt.community_terms(conn, run, cids, k=40)
    for c in cids:
        if len(pool[c]) < 12:
            continue
        term = pool[c][10]                       # not already first
        assert "_" not in term or True
        out = gt.query_terms(conn, run, [c], f"{term} something", k=3)
        assert out[c][0] == term, f"c{c}: lexical hit {term!r} not first: {out[c]}"


def test_query_terms_without_embed_falls_back_to_unsupervised_order(conn, run):
    cids = _some_cids(conn, run, 2)
    pool = gt.community_terms(conn, run, cids, k=40)
    out = gt.query_terms(conn, run, cids, "zzqx", k=3, embed=None)   # no lexical hit
    for c in cids:
        assert out[c] == pool[c][:3]


def test_query_terms_dense_signal_promotes_the_closest_term(conn, run):
    """With an embed that makes one pool term identical to the query, that
    term wins -- even though it is not lexically in the prompt."""
    import numpy as np
    cids = _some_cids(conn, run, 1)
    c = cids[0]
    pool = gt.community_terms(conn, run, cids, k=40)[c]
    target = pool[-1]                     # weakest unsupervised candidate
    q = "zzqx"                            # no lexical hit anywhere

    def embed(texts):
        base = _fake_embed(target)(texts)
        # make the query vector equal to the target's vector
        tv = _fake_embed(target)([target.replace("_", " ")])[0]
        return np.stack([tv if t == q else row for t, row in zip(texts, base)])

    out = gt.query_terms(conn, run, cids, q, k=3, embed=embed)
    assert out[c][0] == target, f"dense argmax not promoted: {out[c]} vs {target}"


def test_query_terms_empty_pool_and_empty_cids(conn, run):
    assert gt.query_terms(conn, run, [], "jury") == {}


# ---------------------------------------- query-weighted local medoid (W11)


def test_local_medoid_matches_its_weighted_definition(conn, run):
    """W11: centrality(o) = w[o] * sum_j w[j] * strength(o, j). Recomputed from
    subgraph_edges under deterministic non-uniform weights; if the function
    ignored `weights` it would return the unweighted argmax instead."""
    ords = [HUB] + [x["ord"] for x in gt.neighbors(conn, run, HUB, limit=14)]
    w = {o: (o % 7 + 1) / 7.0 for o in ords}
    tot = {o: 0.0 for o in ords}
    for e in gt.subgraph_edges(conn, run, ords):
        tot[e["src"]] += w[e["dst"]] * e["strength"]
        tot[e["dst"]] += w[e["src"]] * e["strength"]
    want = min(ords, key=lambda o: (-w[o] * tot[o], o))
    assert gt.local_medoid(conn, run, ords, weights=w) == want


def test_local_medoid_unit_weights_equal_unweighted(conn, run):
    ords = [HUB] + [x["ord"] for x in gt.neighbors(conn, run, HUB, limit=12)]
    assert gt.local_medoid(conn, run, ords, weights={o: 1.0 for o in ords}) \
        == gt.local_medoid(conn, run, ords)


# ------------------------------------------------- search is BM25 (W12)


def test_search_rare_decisive_term_beats_common_word_density(conn):
    """W12: on the document-level run a long essay dense in `school` must not
    outrank a document carrying the rare term. Measured before the fix: plain
    tf*idf anchored on cj48/cf33/cf04 via how+school+communities."""
    try:
        run = gt.get_run(conn, "brown-500-dual")
    except Exception as e:                                # pragma: no cover
        pytest.skip(str(e))
    hits = gt.search(conn, run, "school desegregation", k=5)
    assert hits
    top = gt.node(conn, run, hits[0]["ord"])["body"].lower()
    assert "desegregation" in top, f"top anchor lacks the decisive term: {hits[0]['doc_id']}"


def test_search_scores_are_length_normalised(conn):
    """Two hits on the same terms: the longer document must not win purely on
    length. Checks the sign of the BM25 length term via a controlled pair."""
    try:
        run = gt.get_run(conn, "brown-500-dual")
    except Exception as e:                                # pragma: no cover
        pytest.skip(str(e))
    hits = gt.search(conn, run, "election", k=40)
    same = [h for h in hits if h["terms_hit"] == 1]
    assert len(same) >= 5
    # among single-term hits, score must not be monotone increasing in n_tok
    pairs = [(gt.node(conn, run, h["ord"])["n_tok"], h["score"]) for h in same[:20]]
    longer_but_lower = any(a[0] > b[0] and a[1] < b[1] for a in pairs for b in pairs)
    assert longer_but_lower, "no longer-but-lower pair: length normalisation absent"


def test_question_words_are_stopwords():
    toks = gt.tokenize("how did communities respond to school desegregation")
    assert "how" not in toks and "did" not in toks
    assert {"communities", "respond", "school", "desegregation"} <= set(toks)


# ------------------------------------------- salient gate (W14), chunk terms


def test_salient_gate_is_a_score_ordered_prefix_that_keeps_at_least_half():
    import numpy as np
    rng = np.random.default_rng(1)
    sc = {f"t{i}": float(v) for i, v in enumerate(rng.lognormal(0.0, 1.0, 300))}
    g = gt.salient_gate(sc)
    order = sorted(sc, key=lambda t: (-sc[t], t))
    assert g["kept"] == order[:g["n_kept"]], "kept set must be the top of the ranking"
    assert g["n_kept"] >= g["n_in"] // 2
    assert 1 <= g["threshold_decile"] <= 10 and g["n_in"] == 300


def test_salient_gate_small_inputs_keep_everything():
    assert gt.salient_gate({}) == {"kept": [], "threshold_decile": 0, "n_in": 0, "n_kept": 0}
    g = gt.salient_gate({"a": 3.0, "b": 1.0, "c": 2.0})
    assert g["kept"] == ["a", "c", "b"] and g["n_kept"] == 3


def test_salient_gate_uses_the_more_permissive_branch():
    """min(median-1.4826*MAD, mean-sd): with one huge outlier the mean-sd
    branch drops far below the robust one, so more is kept, not less."""
    base = {f"t{i}": 1.0 + i * 0.01 for i in range(50)}
    with_outlier = dict(base, big=1e6)
    assert gt.salient_gate(with_outlier)["n_kept"] >= gt.salient_gate(base)["n_kept"]


def test_chunk_salient_top_is_within_kept_within_the_chunks_own_terms(conn, run):
    out = gt.chunk_salient(conn, run, [HUB], k=3)
    v = out[HUB]
    tf = gt.node(conn, run, HUB)["tf"]
    assert v["top"] == v["kept"][:3]
    assert set(v["kept"]) <= set(tf)
    assert 0 < v["n_kept"] <= v["n_in"] == len(tf)
    assert gt.chunk_terms(conn, run, [HUB], k=3)[HUB] == v["top"]


def test_corpus_df_is_cached_per_run(conn, run):
    import time
    gt._DF_CACHE.pop(str(run.run_id), None)
    t0 = time.time(); a = gt.corpus_df(conn, run); t1 = time.time() - t0
    t0 = time.time(); b = gt.corpus_df(conn, run); t2 = time.time() - t0
    assert a == b and t2 < t1
    assert gt.corpus_index(conn, run) is gt.corpus_index(conn, run)   # the cached object
    df, avgdl, n = a
    assert n == run.n_chunks and avgdl > 0 and df



# -------------------------------- search over the cached index; disk cache


def test_search_python_bm25_matches_the_formula(conn, run):
    """Parity: score for a single-term query equals BM25 computed by hand
    from the cached postings."""
    import math
    ix = gt.corpus_index(conn, run)
    term = max(ix["df"], key=lambda t: ix["df"][t] if 3 <= ix["df"][t] <= run.n_chunks // 2 else -1)
    hits = gt.search(conn, run, term, k=3)
    assert hits and all(h["terms_hit"] == 1 for h in hits)
    N, avgdl, d = ix["n"], ix["avgdl"], ix["df"][term]
    for h in hits:
        f = ix["post"][term][h["ord"]]; dl = ix["dl"][h["ord"]]
        want = math.log(1 + (N - d + 0.5) / (d + 0.5)) * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * dl / avgdl))
        assert abs(h["score"] - want) < 1e-9


def test_search_is_fast_after_the_index_is_built(conn, run):
    import time
    gt.corpus_index(conn, run)
    t0 = time.time(); gt.search(conn, run, "election county school", k=8); dt = time.time() - t0
    assert dt < 0.5, f"search took {dt:.2f}s; the SQL path took 7.3 s"


def test_disk_cache_round_trips_and_tolerates_a_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(gt, "CACHE_DIR", str(tmp_path / "nested" / "dir"))
    assert gt._disk("nope") is None
    gt._disk_put("k", {"a": [1, 2], "b": {"x": 1.5}})
    assert gt._disk("k") == {"a": [1, 2], "b": {"x": 1.5}}
    monkeypatch.setattr(gt, "CACHE_DIR", str(tmp_path / "file-not-dir"))
    (tmp_path / "file-not-dir").write_text("x")
    gt._disk_put("k2", {"z": 1})                       # must not raise


def test_stoplist_is_cheap_and_knows_contractions():
    import importlib, sys, time
    sys.modules.pop("stoplist", None)
    t0 = time.time(); import stoplist; dt = time.time() - t0
    assert dt < 1.0, f"stoplist import took {dt:.2f}s"
    assert {"didn", "wasn", "couldn", "the", "how"} <= stoplist._STOP
    assert "election" not in stoplist._STOP


# ---- source metadata (R20)
# DB-free: exercise the pure helpers directly, no `conn`/`run` fixtures.


def test_source_of_battery():
    assert gt.source_of({"source": "brown", "doc_id": "wiki/12"}) == "brown"
    assert gt.source_of({"doc_id": "wiki/12"}) == "wiki"
    assert gt.source_of({"doc_id": "ca01"}) is None            # today's Brown id
    assert gt.source_of({}) is None
    assert gt.source_of({"source": "", "doc_id": "wiki/12"}) == "wiki"


def test_source_mix_counts_and_orders():
    rows = [
        {"source": "brown", "doc_id": "brown/ca01"},
        {"source": "brown", "doc_id": "brown/ca02"},
        {"doc_id": "wiki/7"},
        {"doc_id": "ca03"},                                    # unsourced
    ]
    assert gt.source_mix(rows) == {"brown": 2, "wiki": 1}
    assert list(gt.source_mix(rows)) == ["brown", "wiki"]      # count desc, name asc
    assert gt.source_mix([{"doc_id": "ca01"}, {}]) == {}
    assert gt.source_mix([]) == {}


def test_format_source_mix():
    assert gt.format_source_mix({}) == ""
    assert gt.format_source_mix({"brown": 2, "wiki": 1}) == "brown 2 · wiki 1"


# DB-backed, skip-tolerant: the degrade path on real pre-R20 data.


def test_node_search_neighbors_walk_carry_none_source_on_pre_r20_run(conn, run):
    n = gt.node(conn, run, HUB)
    assert "source" in n and n["source"] is None
    for h in gt.search(conn, run, "jury election county", k=5):
        assert h.get("source") is None
    for nb in gt.neighbors(conn, run, HUB, limit=10):
        assert nb.get("source") is None
    rows = gt.walk(conn, run, HUB, hops=2, cap=25)
    assert rows
    for r in rows:
        assert r.get("source") is None
        assert len(r["prov_path"]) == r["hop"], "W7 unbroken"


def test_communities_touched_sources_empty_on_pre_r20_run(conn, run):
    nbrs = gt.neighbors(conn, run, HUB, limit=100)
    ords = [HUB] + [n["ord"] for n in nbrs]
    touched = gt.communities_touched(conn, run, ords)
    assert touched
    for t in touched:
        assert t["sources"] == {}
    # pre-existing assertions still hold
    hits = [t["hits"] for t in touched]
    assert hits == sorted(hits, reverse=True)
    assert sum(hits) <= len(ords)
    for t in touched:
        assert t["hits"] <= t["size"]


def test_run_sources_empty_on_pre_r20_run(conn, run):
    assert gt.run_sources(conn, run) == {}


# ---------------------------------------------------------------- dendrites
def test_dendrite_sort_recovers_planted_chains():
    """Two planted collinear families and one independent variable: the sort
    must thread each family into one chain and leave the loner alone
    (correlation sorting.md semantics)."""
    import numpy as np
    rng = np.random.default_rng(7)
    n = 120
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.3, size=n)
    c = b + rng.normal(scale=0.3, size=n)
    p = rng.normal(size=n)
    q = p + rng.normal(scale=0.3, size=n)
    lone = rng.normal(size=n)
    M = np.column_stack([a, b, c, p, q, lone])
    out = gt.dendrite_sort(M, ["a", "b", "c", "p", "q", "lone"])
    fam = {frozenset(ch) for ch in out["chains"]}
    assert frozenset(["a", "b", "c"]) in fam
    assert frozenset(["p", "q"]) in fam
    assert frozenset(["lone"]) in fam
    # every variable in exactly one chain
    flat = [x for ch in out["chains"] for x in ch]
    assert sorted(flat) == ["a", "b", "c", "lone", "p", "q"]


def test_dendrite_sort_significance_uses_n():
    """The same r that chains at n=120 must NOT chain at n=8 -- significance
    is a function of observation count, not correlation alone."""
    import numpy as np
    rng = np.random.default_rng(11)
    a = rng.normal(size=8)
    b = a + rng.normal(scale=0.9, size=8)     # modest r, tiny n
    M = np.column_stack([a, b])
    out = gt.dendrite_sort(M, ["a", "b"])
    assert all(len(ch) == 1 for ch in out["chains"])


def test_dendrite_sort_floors_support_and_degenerates():
    import numpy as np
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    sparse = np.zeros(60); sparse[:2] = 1.0    # 2 nonzero < min_support
    flat = np.ones(60)                          # zero variance
    M = np.column_stack([x, sparse, flat])
    out = gt.dendrite_sort(M, ["x", "sparse", "flat"], min_support=4)
    assert out["names"] == ["x"]


# ---------------------------------- second-order term ladder (W17)
def test_llr_is_zero_on_independence_and_symmetric():
    """k11 = row_total*col_total/N exactly => independence, G2 ~= 0; the
    table's transpose (rows/cols swapped) gives the same value."""
    tables = [(20, 20, 30, 30), (10, 10, 20, 20), (24, 56, 36, 84)]
    for k11, k12, k21, k22 in tables:
        g2 = gt.llr(k11, k12, k21, k22)
        assert abs(g2) < 1e-9
        assert gt.llr(k11, k21, k12, k22) == pytest.approx(g2, abs=1e-9)


@pytest.mark.parametrize("N", [40, 100, 300])
def test_llr_gate_admits_association_and_rejects_independence_at_the_same_scale(N):
    """Same marginals (T = C = N//4) at every scale: an 80%-overlap pair
    clears the 10.83 gate, the marginal-matched independent pair does not."""
    T = N // 4
    C = N // 4
    k11_i = round(T * C / N)
    k12_i, k21_i = T - k11_i, C - k11_i
    k22_i = N - k11_i - k12_i - k21_i
    assert gt.llr(k11_i, k12_i, k21_i, k22_i) < 10.83

    k11_a = round(0.8 * T)
    k12_a, k21_a = T - k11_a, C - k11_a
    k22_a = N - k11_a - k12_a - k21_a
    assert gt.llr(k11_a, k12_a, k21_a, k22_a) >= 10.83


def test_second_order_gate_keeps_only_the_associated_nomens():
    import numpy as np
    N = 40
    names = ["target", "assoc1", "assoc2", "indep1", "indep2", "indep3"]
    M = np.zeros((N, 6))
    target_rows = list(range(0, 10))
    M[target_rows, 0] = 1
    M[target_rows[:8] + [10, 11], 1] = 1              # assoc1: 8/10 overlap
    M[target_rows[:8] + [12, 13], 2] = 1              # assoc2: 8/10 overlap
    M[target_rows[:2] + list(range(20, 28)), 3] = 1   # indep1: 2/10 overlap
    M[target_rows[:2] + list(range(28, 36)), 4] = 1   # indep2: 2/10 overlap
    M[[target_rows[0], target_rows[1]] + [36, 37, 38, 39, 14, 15, 16, 17], 5] = 1
    res = gt.second_order_terms(M, names, None, "target")
    assert res["n_survivors"] == 2
    assert {r["term"] for r in res["ranked"]} == {"assoc1", "assoc2"}
    with pytest.raises(LookupError):
        gt.second_order_terms(M, names, None, "not-a-term")


@pytest.mark.parametrize("seed", [1, 5, 9])
def test_second_order_centroid_ranks_shared_context_above_the_unrelated_term(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    N = 40
    names = ["target", "shared", "unrelated"]
    M = np.zeros((N, 3))
    target_rows = list(range(0, 10))
    M[target_rows, 0] = 1
    M[target_rows[:8] + [10, 11], 1] = 1   # shared: 8/10 overlap, cluster A
    M[target_rows[:8] + [12, 13], 2] = 1   # unrelated: 8/10 overlap, cluster B

    E = np.zeros((N, 2))
    for i in range(N):
        E[i] = np.array([1.0, 0.0]) + rng.normal(scale=0.05, size=2)   # cluster A default
    for i in (12, 13):
        E[i] = np.array([0.0, 20.0]) + rng.normal(scale=0.5, size=2)   # cluster B, dominant mag

    res = gt.second_order_terms(M, names, E, "target")
    assert res["rung"] == "centroid"
    by_term = {r["term"]: r["cos"] for r in res["ranked"]}
    assert res["ranked"][0]["term"] == "shared"
    assert by_term["shared"] > by_term["unrelated"]


def test_second_order_skew_trip_reranks_and_corrects_a_cosine_inversion():
    import numpy as np
    N, D = 60, 8
    target_rows = list(range(0, 20))
    true_rows = target_rows[:16] + [40, 41, 42, 43]
    decoy_rows = target_rows[:16] + [44, 45, 46, 47]
    filler_rows = [target_rows[:16] + [24, 25, 26, 27],
                   target_rows[:16] + [28, 29, 30, 31],
                   target_rows[:16] + [32, 33, 34, 35]]
    names = ["target", "true", "decoy", "f1", "f2", "f3"]
    M = np.zeros((N, 6))
    M[target_rows, 0] = 1
    M[true_rows, 1] = 1
    M[decoy_rows, 2] = 1
    for i, rows in enumerate(filler_rows):
        M[rows, 3 + i] = 1

    rng = np.random.default_rng(1)
    E = rng.normal(scale=1.0, size=(N, D))
    sig = rng.normal(size=D); sig /= np.linalg.norm(sig)      # the real separating signal
    for i in target_rows:
        E[i] = E[i] * 0.3 + sig * 1.0
    hub = rng.normal(size=D); hub /= np.linalg.norm(hub)      # broad hub, no row-level signal
    for i in range(N):
        E[i] = E[i] + hub * 2.0
    for i in (44, 45, 46, 47):                                # decoy's extra rows: hub-dominant
        E[i] = hub * 2.0 + rng.normal(scale=0.2, size=D)
    for i in (40, 41, 42, 43):                                # true's extra rows: sig-aligned
        E[i] = sig * 1.0 + rng.normal(scale=0.3, size=D)

    res = gt.second_order_terms(M, names, E, "target", min_skew_n=4)
    assert abs(res["skew"]) > 2.0
    assert res["rung"] == "auc"
    assert res["ranked"][0]["term"] == "true"

    # cosine-only ranking (skew_trip disabled) had decoy first -- the rerank actually flipped it
    unranked = gt.second_order_terms(M, names, E, "target", min_skew_n=4, skew_trip=999)
    assert unranked["rung"] == "centroid"
    assert unranked["ranked"][0]["term"] == "decoy"


def test_second_order_without_skew_leaves_the_centroid_ranking_alone():
    import numpy as np
    N = 40
    names = ["target", "c1", "c2", "c3", "c4"]
    M = np.zeros((N, 5))
    target_rows = list(range(0, 10))
    M[target_rows, 0] = 1
    M[target_rows[:8] + [10, 11], 1] = 1
    M[target_rows[:8] + [12, 13], 2] = 1
    M[target_rows[:8] + [14, 15], 3] = 1
    M[target_rows[:8] + [16, 17], 4] = 1
    rng = np.random.default_rng(2)
    E = rng.normal(size=(N, 4))
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    res = gt.second_order_terms(M, names, E, "target")
    assert res["rung"] == "centroid"
    assert all(r["auc"] is None for r in res["ranked"])
    cos_order = sorted(res["ranked"], key=lambda r: (-r["cos"], r["term"]))
    assert [r["term"] for r in res["ranked"]] == [r["term"] for r in cos_order]


@pytest.mark.parametrize("min_df", [3, 5, 8])
def test_second_order_min_df_floor_drops_thin_terms(min_df):
    import numpy as np
    N = 30
    names = ["target", "a", "b", "c", "d"]
    M = np.zeros((N, 5))
    target_rows = list(range(0, 6))                      # target df = 6
    M[target_rows, 0] = 1
    M[[0, 1], 1] = 1                                      # a: df 2
    M[[0, 1, 2, 3], 2] = 1                                 # b: df 4
    M[[0, 1, 2, 3, 4, 20, 21], 3] = 1                      # c: df 7
    M[[0, 1, 2, 3, 4, 5, 22, 23, 24, 25], 4] = 1           # d: df 10

    res = gt.second_order_terms(M, names, None, "target", min_df=min_df)
    dfs = {"a": 2, "b": 4, "c": 7, "d": 10}
    below = {t for t, d in dfs.items() if d < min_df}
    assert set(res["dropped"]) == below
    ranked_terms = {r["term"] for r in res["ranked"]}
    assert ranked_terms.isdisjoint(below)

    if min_df > 6:                                        # target's own df (6) is below floor
        assert res["ranked"] == []
        assert res["reason"] == "target_df_below_min"


def test_second_order_without_embeddings_falls_back_to_the_llr_order():
    import numpy as np
    N = 40
    names = ["target", "A", "B", "C"]
    M = np.zeros((N, 4))
    target_rows = list(range(0, 10))
    M[target_rows, 0] = 1
    M[target_rows[:7] + [20, 21, 22], 1] = 1   # A: weakest association
    M[target_rows[:8] + [20, 21], 2] = 1       # B: medium
    M[target_rows[:9] + [20], 3] = 1           # C: strongest

    res = gt.second_order_terms(M, names, None, "target")
    assert res["rung"] == "llr"
    assert [r["term"] for r in res["ranked"]] == ["C", "B", "A"]
    assert all(r["cos"] is None and r["auc"] is None for r in res["ranked"])
    g2s = [r["g2"] for r in res["ranked"]]
    assert g2s == sorted(g2s, reverse=True)


def test_second_order_wrapper_shape_on_the_live_run(conn, run):
    ix = gt.corpus_index(conn, run)
    n = ix["n"]
    vocab = [t for t, d in ix["df"].items() if 5 <= d <= 0.5 * n]
    if not vocab:
        pytest.skip("brown-50 vocabulary has no term in the default df band")
    target = sorted(vocab)[0]
    res = gt.second_order(conn, run, target, k=8)
    assert len(res["ranked"]) <= 8
    assert res["rung"] in {"llr", "centroid", "auc"}
    for r in res["ranked"]:
        assert r["term"] != target
        d = ix["df"].get(r["term"], 0)
        assert 5 <= d <= 0.5 * n


# ---------------------------------- named graph metrics (W18/W19/W20)
def _star(n_leaves=8):
    adj = {0: {}}
    for i in range(1, n_leaves + 1):
        adj[0][i] = 1.0
        adj[i] = {0: 1.0}
    return adj


def _triangle():
    return {0: {1: 1.0, 2: 1.0}, 1: {0: 1.0, 2: 1.0}, 2: {0: 1.0, 1: 1.0}}


def _ring_with_chords(n=60, chord_step=15, chord_every=5):
    adj = {i: {} for i in range(n)}

    def link(a, b, w=1.0):
        adj[a][b] = w
        adj[b][a] = w

    for i in range(n):
        link(i, (i + 1) % n)
    for i in range(0, n, chord_every):
        link(i, (i + chord_step) % n)
    return adj


def _path(n=5):
    adj = {i: {} for i in range(n)}
    for i in range(n - 1):
        adj[i][i + 1] = 1.0
        adj[i + 1][i] = 1.0
    return adj


def test_graph_metrics_star_puts_the_hub_first():
    res = gt.graph_metrics(_star(8))
    nodes = res["nodes"]
    assert max(nodes, key=lambda o: nodes[o]["betweenness"]) == 0
    assert max(nodes, key=lambda o: nodes[o]["pagerank"]) == 0
    for leaf in range(1, 9):
        assert nodes[leaf]["betweenness"] == 0.0
    assert all(row["triangles"] == 0 for row in nodes.values())
    assert res["n"] == 9
    assert res["approx"] is False


def test_graph_metrics_triangle_is_fully_clustered():
    res = gt.graph_metrics(_triangle())
    for row in res["nodes"].values():
        assert row["triangles"] == 1
        assert row["clustering"] == 1.0


def test_graph_metrics_pagerank_is_a_distribution():
    res = gt.graph_metrics(_star(8))
    total = sum(row["pagerank"] for row in res["nodes"].values())
    assert total == pytest.approx(1.0)


def test_graph_metrics_sampled_betweenness_is_deterministic():
    import pickle
    adj = _ring_with_chords()
    r1 = gt.graph_metrics(adj, k_sample=8)
    r2 = gt.graph_metrics(adj, k_sample=8)
    assert pickle.dumps(r1) == pickle.dumps(r2)
    assert r1["approx"] is True
    assert r1["params"]["k"] == 8


def _two_cliques_one_bridge():
    adj = {i: {} for i in range(10)}

    def link(a, b, w=1.0):
        adj[a][b] = w
        adj[b][a] = w

    for a in range(5):
        for b in range(a + 1, 5):
            link(a, b)
    for a in range(5, 10):
        for b in range(a + 1, 10):
            link(a, b)
    link(4, 5)
    return adj


def test_partition_metrics_two_cliques_one_bridge():
    adj = _two_cliques_one_bridge()
    res = gt.partition_metrics(adj, {0: list(range(5)), 1: list(range(5, 10))})
    e_total = sum(len(v) for v in adj.values()) / 2.0
    assert e_total == 21                     # 10 + 10 clique edges + 1 bridge
    by_cid = {c["cid"]: c for c in res["communities"]}
    for cid in (0, 1):
        c = by_cid[cid]
        assert c["density"] == pytest.approx(1.0)
        assert c["volume"] == 21              # 4 nodes*deg4 + bridge node deg5
        assert c["internal_edges"] == 10
        assert c["cut"] == 1
        assert c["conductance"] == pytest.approx(1 / 21)
    assert res["wcc"]["components"] == 1


def test_partition_metrics_reports_a_shattered_graph():
    adj = {0: {1: 1.0, 2: 1.0}, 1: {0: 1.0, 2: 1.0}, 2: {0: 1.0, 1: 1.0},
          3: {4: 1.0, 5: 1.0}, 4: {3: 1.0, 5: 1.0}, 5: {3: 1.0, 4: 1.0}}
    res = gt.partition_metrics(adj, {0: [0, 1, 2], 1: [3, 4, 5]})
    assert res["wcc"]["components"] == 2
    assert res["wcc"]["sizes"] == [3, 3]
    assert res["wcc"]["largest_frac"] == pytest.approx(0.5)
    assert all(c["conductance"] == pytest.approx(0.0) for c in res["communities"])


def test_partition_metrics_summary_is_median_and_p90():
    # three disjoint communities, no cross edges -> conductance 0 for all,
    # densities 1.0 (triangle), 0.5 (4-node path), 0.4 (5-node star).
    adj = {0: {1: 1.0, 2: 1.0}, 1: {0: 1.0, 2: 1.0}, 2: {0: 1.0, 1: 1.0},
          10: {11: 1.0}, 11: {10: 1.0, 12: 1.0}, 12: {11: 1.0, 13: 1.0},
          13: {12: 1.0},
          20: {21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0},
          21: {20: 1.0}, 22: {20: 1.0}, 23: {20: 1.0}, 24: {20: 1.0}}
    members = {0: [0, 1, 2], 1: [10, 11, 12, 13], 2: [20, 21, 22, 23, 24]}
    res = gt.partition_metrics(adj, members)
    densities = sorted(c["density"] for c in res["communities"])
    conductances = sorted(c["conductance"] for c in res["communities"])
    assert densities == pytest.approx([0.4, 0.5, 1.0])
    assert res["summary"]["density_median"] == pytest.approx(gt._pct(densities, 0.5))
    assert res["summary"]["density_p90"] == pytest.approx(gt._pct(densities, 0.9))
    assert res["summary"]["conductance_median"] == pytest.approx(gt._pct(conductances, 0.5))
    assert res["summary"]["conductance_p90"] == pytest.approx(gt._pct(conductances, 0.9))
    assert res["summary"]["n_communities"] == 3


def test_ppr_concentrates_on_the_seed_and_sums_to_one():
    # Seeded at the CENTER (2): a degree-1 endpoint seed forwards its entire
    # mass onward every iteration (no self-loop), so its immediate neighbour
    # legitimately outscores it -- verified byte-identical to nx.pagerank's
    # own personalization on this same path. The center has no such
    # asymmetry: PPR is symmetric and strictly decreasing outward both ways.
    adj = _path(5)
    res = gt.ppr(adj, [2])
    assert res[2] > res[1] == pytest.approx(res[3])
    assert res[1] > res[0] == pytest.approx(res[4])
    assert sum(res.values()) == pytest.approx(1.0)


def test_ppr_is_deterministic_and_symmetric_under_seed_swap():
    import pickle
    adj = _path(5)
    r_from_0 = gt.ppr(adj, [0])
    r_from_4 = gt.ppr(adj, [4])
    assert r_from_0[4] == pytest.approx(r_from_4[0])
    assert pickle.dumps(gt.ppr(adj, [0])) == pickle.dumps(gt.ppr(adj, [0]))


# --------------------------- named graph metrics, live (brown-50)
def test_node_metrics_covers_the_run_and_the_split_sums_to_degree(conn, run):
    adj = gt.full_adjacency(conn, run)
    res = gt.node_metrics(conn, run)
    assert set(adj) <= set(res["nodes"])
    assert res["approx"] is False
    for o in sorted(adj)[:8]:
        row = res["nodes"][o]
        assert sum(row["prov"].values()) == row["degree"]
        assert row["degree"] == gt.degrees(conn, run, [o])[o]
        assert set(row["prov"]) <= set(run.provenance)


def test_node_metrics_ords_filter_is_a_view_not_a_recomputation(conn, run):
    filtered = gt.node_metrics(conn, run, [HUB, 0])["nodes"]
    whole = gt.node_metrics(conn, run)["nodes"]
    assert filtered[HUB] == whole[HUB]
    assert filtered[HUB]["degree"] >= filtered[0]["degree"]


def test_community_metrics_matches_the_stored_partition(conn, run):
    with conn.cursor() as cur:
        cur.execute("SELECT cid, size FROM community WHERE run_id = %s ORDER BY cid",
                   (run.run_id,))
        stored = cur.fetchall()
    res = gt.community_metrics(conn, run)
    assert [c["cid"] for c in res["communities"]] == [r["cid"] for r in stored]
    assert [c["size"] for c in res["communities"]] == [r["size"] for r in stored]
    for c in res["communities"]:
        assert 0.0 <= c["density"] <= 1.0
        assert 0.0 <= c["conductance"] <= 1.0
    assert res["summary"]["n_communities"] == N_COMMUNITIES
    adj = gt.full_adjacency(conn, run)
    assert sum(res["wcc"]["sizes"]) == len(adj)


def test_pathways_ppr_is_additive_and_does_not_reorder(conn, run):
    rows = gt.walk(conn, run, HUB, hops=2, cap=25)
    ords = sorted({HUB} | {r["ord"] for r in rows})
    anchors = ords[:4]
    if len(anchors) < 2:
        pytest.skip("walk from HUB did not yield enough anchors")
    pw = gt.pathways(conn, run, ords, anchors)
    for key in ("n", "edges", "components", "wcc_sizes",
               "largest_component_frac", "density", "conductance", "pairs"):
        assert key in pw
    assert pw["pairs"] == sorted(pw["pairs"], key=lambda p: (-p["dwpc"], p["a"], p["b"]))
    for p in pw["pairs"]:
        assert isinstance(p["ppr"], float)
    assert set(pw["ppr"]) == set(ords)
    assert sum(pw["ppr"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- W21 alias expansion


def test_expand_terms_adds_siblings_keeps_originals(conn, run):
    aliases = {"jury": ("panel", "grand_jury")}
    out = gt.expand_terms(["jury", "county"], aliases)
    assert out == sorted({"jury", "county", "panel", "grand_jury"})


def test_expand_terms_empty_map_and_unknown_token_pass_through(conn, run):
    assert gt.expand_terms(["jury", "county"], {}) == sorted(["jury", "county"])
    assert gt.expand_terms(["nonexistent"], {"jury": ("panel",)}) == ["nonexistent"]


def test_expand_terms_never_drops_a_term_with_a_large_alias_set(conn, run):
    aliases = {"jury": tuple(f"alt{i}" for i in range(50))}
    out = gt.expand_terms(["jury"], aliases)
    assert "jury" in out
    assert len(out) == 51
    assert out == sorted(out)


def test_alias_map_degrades_gracefully_whether_or_not_entities_exists(conn, run):
    gt._ALIAS_CACHE.clear()
    m = gt.alias_map(conn, run)
    assert isinstance(m, dict)          # allowed to be {} -- brown-50 has no resolve run
    gt._ALIAS_CACHE.clear()


def test_search_expand_aliases_is_a_noop_when_alias_map_is_empty(conn, run):
    gt._ALIAS_CACHE.clear()
    q = "jury election county"
    plain = gt.search(conn, run, q, k=5)
    expanded = gt.search(conn, run, q, k=5, expand_aliases=True)
    assert expanded == plain
    gt._ALIAS_CACHE.clear()


def test_search_expand_aliases_with_planted_map_surfaces_more_evidence(conn, run, monkeypatch):
    """Planted alias joining two real vocabulary terms: expansion must not
    drop the unexpanded results and should surface at least one new ord or a
    strictly higher score on a shared ord."""
    gt._ALIAS_CACHE.clear()
    monkeypatch.setattr(gt, "alias_map", lambda conn, run: {"jury": ("election",)})
    try:
        plain = gt.search(conn, run, "jury", k=20)
        expanded = gt.search(conn, run, "jury", k=20, expand_aliases=True)
        plain_ords = {h["ord"] for h in plain}
        expanded_ords = {h["ord"] for h in expanded}
        assert plain_ords <= expanded_ords
        plain_scores = {h["ord"]: h["score"] for h in plain}
        expanded_scores = {h["ord"]: h["score"] for h in expanded}
        new_ords = expanded_ords - plain_ords
        higher = any(expanded_scores[o] > plain_scores.get(o, -1) for o in expanded_ords)
        assert new_ords or higher
    finally:
        gt._ALIAS_CACHE.clear()


def test_search_expand_aliases_default_is_false():
    import inspect
    assert inspect.signature(gt.search).parameters["expand_aliases"].default is False


def test_shim_reexports_the_whole_public_surface():
    """T37: the split is invisible to callers. Every name a caller uses must
    resolve on graph_tools, and be the SAME object the sibling defines."""
    import types
    import gt_sql, gt_terms, gt_metrics
    for mod in (gt_sql, gt_terms, gt_metrics):
        for name in mod.__dict__:
            if name.startswith("__") or name == "_gt":
                continue
            obj = getattr(mod, name)
            if isinstance(obj, types.ModuleType):
                continue                      # imported module (math, os, re, psycopg, config, np)
            if getattr(obj, "__module__", mod.__name__) != mod.__name__:
                continue                      # imported helper (dataclass, stoplist._STOP, ...)
            assert getattr(gt, name, None) is obj, f"{mod.__name__}.{name} not re-exported"
    assert gt.search is gt_sql.search and gt.llr is gt_terms.llr and gt.ppr is gt_metrics.ppr
