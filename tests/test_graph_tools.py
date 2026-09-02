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
N_CHUNKS = 51
N_EDGES = 115
N_COMMUNITIES = 5
HUB = 43          # joint-maximum degree node
HUB_DEGREE = 12


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
