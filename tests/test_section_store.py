"""| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| FIXTURE | Create small relational section graphs inside a forced rollback transaction. | [mandatory] | R1 |
| COMPARE | Compare SQL neighbors with AGE Cypher neighbors on varied graph shapes. | [mandatory] | R2 |
| BULK | Load 81,473 vertices and 10,000 relationships from relational COPY output and check counts/null scores. | [mandatory] | R3 |
| CLEANUP | Roll back each fixture and AGE graph after assertions. | [contract] | R1 |

R1 Tests use an isolated disposable database and force rollback; no live rows persist.
R2 The parity fixtures cover tied weights, high degree, multiple seeds, exclusions, disconnected nodes, null scores, and caps.
R3 The scaled loader fixture compares source and graph counts, preserves null scores, and leaves no server CSV files. Neighbor behavior is covered by the varied small fixtures.
Preconditions: the disposable database has pgvector and Apache AGE installed. Failure modes: any mismatch or AGE query failure fails the fixture.

Run: pytest tests/test_section_store.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import section_store as st
import age_graph


def test_or_query_ors_the_words_once_and_drops_punctuation():
    assert st.or_query("How does GRPO remove the value-function? GRPO!") == "how | does | grpo | remove | the | value | function"
    assert st.or_query("?? 1 -") == ""


def test_vec_literal():
    assert st.vec(np.array([1.0, -0.5, 0.25], np.float32)) == "[1,-0.5,0.25]"


def test_legacy_build_scoped_dense_rows_migrate_to_global_record_id_bridge(conn):
    schema = "pytest_dense_migration"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET LOCAL search_path TO {schema}, public")
    conn.execute("CREATE TABLE sect_build (build_id bigint PRIMARY KEY, tag text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), live boolean NOT NULL, n_nodes int, n_edges int, n_communities int, params jsonb NOT NULL DEFAULT '{}')")
    conn.execute("CREATE TABLE sect_node (build_id bigint NOT NULL REFERENCES sect_build, ord int NOT NULL, doc_id text NOT NULL, section_idx int NOT NULL, section_title text, community int NOT NULL, x real, y real, genre_frac real, text text NOT NULL, tsv tsvector, PRIMARY KEY(build_id,ord))")
    conn.execute("CREATE TABLE sect_embedding (build_id bigint NOT NULL, record_id int NOT NULL, emb vector(256) NOT NULL, emb_b vector(256), PRIMARY KEY(build_id,record_id))")
    conn.execute("INSERT INTO sect_build(build_id,tag,live) VALUES (1,'xpa',false),(2,'xpa',true)")
    vector = "[1," + ",".join(["0"] * 255) + "]"
    for build_id in (1, 2):
        conn.execute("INSERT INTO sect_node(build_id,ord,doc_id,section_idx,community,text) VALUES (%s,0,%s,0,0,'section')", (build_id, f"arxiv/{build_id}"))
        conn.execute("INSERT INTO sect_embedding(build_id,record_id,emb) VALUES (%s,0,%s::vector)", (build_id, vector))

    st.ensure_schema(conn)

    rows = conn.execute("SELECT n.build_id,n.ord,n.record_id,v.record_id FROM sect_node n JOIN sect_embedding v USING(record_id) ORDER BY n.build_id").fetchall()
    assert len(rows) == 2 and rows[0][2] != rows[1][2]
    assert [row[:2] for row in rows] == [(1, 0), (2, 0)]
    assert st.seeds_dense(conn, 2, np.eye(1, st.DIM, 0, dtype=np.float32)[0], 1) == [0]
    cols = {r[0] for r in conn.execute("SELECT attname FROM pg_attribute WHERE attrelid='sect_embedding'::regclass AND attnum>0 AND NOT attisdropped").fetchall()}
    assert cols == {"record_id", "emb", "emb_b"}
    assert conn.execute("SELECT count(*) FROM sect_embedding_build_scoped").fetchone()[0] == 2


@pytest.fixture()
def conn():
    import sparsevec_store as ss
    c = ss.connect()                                                     # autocommit: force_rollback makes everything below one transaction that is never committed
    try:
        with c.transaction(force_rollback=True):
            yield c
    finally:
        c.close()


def world():
    """6 sections: 0,1 in community 0 about speculative decoding; 2,3 in community 1 about agent memory; 4,5 in community 2 about image captions.
    Edges (a<b): 0-1 weight 3, 0-2 weight 1, 2-3 weight 2, 4-5 weight 2, 1-2 weight 0.5."""
    recs = [{"doc_id": "arxiv/p%d" % i, "section_idx": i, "section_title": "T%d" % i, "text": "## T%d\n%s" % (i, body)}
            for i, body in enumerate(["speculative decoding drafts tokens with a small model", "speculative decoding verifies draft tokens",
                                      "agent memory stores episodes for retrieval", "long term memory for agents",
                                      "a woman standing in a green field", "an image of a man sitting"])]
    X = np.zeros((6, st.DIM), np.float32)
    for i, c in enumerate([0, 0, 1, 1, 2, 2]):
        X[i, c] = 1.0
        X[i, 10 + i] = 0.1
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    lab = np.array([0, 0, 1, 1, 2, 2])
    ij = [(0, 1, 3.0), (0, 2, 1.0), (2, 3, 2.0), (4, 5, 2.0), (1, 2, 0.5)]
    n = len(ij)
    fused = {"names": ["dense", "sparse"], "i": np.array([a for a, _, _ in ij]), "j": np.array([b for _, b, _ in ij]), "w": np.array([w for _, _, w in ij], np.float32),
             "s": {"dense": np.full(n, .8, np.float32), "sparse": np.full(n, .3, np.float32)}, "z": {"dense": np.full(n, 4, np.float32), "sparse": np.full(n, 2, np.float32)},
             "prov": np.full(n, 3), "back": np.zeros(n, np.int64)}
    comm = {c: {"size": 2, "genre_score": .1 * c, "genre_flag": c == 2, "title": "C%d" % c, "summary": "summary %d" % c, "status": "draft", "terms": ["t%d" % c]} for c in range(3)}
    S = np.zeros((3, st.DIM), np.float32)
    for c in range(3):
        S[c, c] = 1.0
    return recs, X, lab, np.zeros((6, 2), np.float32), np.full(6, .5, np.float32), fused, comm, S


def build(conn, tag="pytest_section_store"):
    recs, X, lab, XY, frac, fused, comm, S = world()
    return st.persist(conn, tag, recs, X, lab, XY, frac, fused, comm, S, {"mu": [0.0] * st.DIM}), X


def test_persist_writes_everything_and_marks_the_build_live(conn):
    b, _ = build(conn)
    assert st.live_build(conn, "pytest_section_store")[0] == b
    assert conn.execute("SELECT count(*) FROM sect_node WHERE build_id = %s", (b,)).fetchone()[0] == 6
    assert conn.execute("SELECT count(*) FROM sect_edge WHERE build_id = %s", (b,)).fetchone()[0] == 5
    assert conn.execute("SELECT count(*) FROM sect_community WHERE build_id = %s", (b,)).fetchone()[0] == 3
    assert conn.execute("SELECT text FROM sect_node WHERE build_id = %s AND ord = 4", (b,)).fetchone()[0].startswith("## T4\n")      # D2: whole text, heading kept for display


def test_a_second_build_supersedes_the_first_and_deletes_nothing(conn):
    a, _ = build(conn)
    b, _ = build(conn)
    assert st.live_build(conn, "pytest_section_store")[0] == b != a
    assert conn.execute("SELECT live FROM sect_build WHERE build_id = %s", (a,)).fetchone()[0] is False
    assert conn.execute("SELECT count(*) FROM sect_node WHERE build_id = %s", (a,)).fetchone()[0] == 6
    old_ids = {r[0] for r in conn.execute("SELECT record_id FROM sect_node WHERE build_id=%s", (a,)).fetchall()}
    new_ids = {r[0] for r in conn.execute("SELECT record_id FROM sect_node WHERE build_id=%s", (b,)).fetchall()}
    assert len(old_ids) == len(new_ids) == 6 and old_ids.isdisjoint(new_ids)
    assert conn.execute("SELECT count(*) FROM sect_embedding WHERE record_id=ANY(%s)", (list(old_ids | new_ids),)).fetchone()[0] == 12
    columns = {r[0] for r in conn.execute("SELECT attname FROM pg_attribute WHERE attrelid='sect_embedding'::regclass AND attnum>0 AND NOT attisdropped").fetchall()}
    assert columns == {"record_id", "emb", "emb_b"}


def test_seeds_dense_orders_by_cosine(conn):
    b, X = build(conn)
    assert st.seeds_dense(conn, b, X[4], 3)[:2] == [4, 5] or st.seeds_dense(conn, b, X[4], 3)[:2] == [5, 4]
    assert st.seeds_dense(conn, b, X[2], 1) == [2]


def test_word_df_lists_indexed_words_rarest_first_and_drops_stop_and_unseen_words(conn):
    b, _ = build(conn)
    assert st.word_df(conn, b, "how does speculative decoding with a small model work") == [("small", 1), ("model", 1), ("speculative", 2), ("decoding", 2)]
    assert st.word_df(conn, b, "how does it work") == []                                       # stop words and unseen words carry no frequency


def test_idf_weighting_lets_a_rare_word_outrank_a_common_one(conn):
    b, _ = build(conn)
    assert st.seeds_lexical(conn, b, "memory small", 5)[0] == 0                                # 'small' is in one section (idf ln 6), 'memory' in two (idf ln 3)
    assert st.seeds_lexical(conn, b, "speculative decoding with a small model", 5)[:2] == [0, 1]    # the section matching all four words, then the one matching two


def test_seeds_lexical_matches_body_any_word_and_not_the_heading(conn):
    b, _ = build(conn)
    assert set(st.seeds_lexical(conn, b, "how does speculative decoding work", 5)) == {0, 1}
    assert st.seeds_lexical(conn, b, "agent memory", 5)[0] in (2, 3)
    assert st.seeds_lexical(conn, b, "T4", 5) == []                      # D3: "T4" is only in section 4's heading line
    assert st.seeds_lexical(conn, b, "?? --", 5) == []


def test_neighbours_rank_by_weight_exclude_seeds_and_cap(conn):
    b, _ = build(conn)
    out = st.neighbours(conn, b, [0, 2], 2)
    assert [n["ord"] for n in out[0]] == [1]                              # 0's edges: 1 (w3) and 2 (w1, a seed, excluded)
    assert [n["ord"] for n in out[2]] == [3, 1]                          # 2's edges: 3 (w2), 1 (w.5), 0 is a seed
    assert out[2][0]["weight"] == 2.0 and abs(out[2][0]["dense_sim"] - 0.8) < 1e-6
    assert [n["ord"] for n in st.neighbours(conn, b, [2], 1)[2]] == [3]


@pytest.mark.parametrize(
    "build_id,node_count,edges,seeds,cap",
    [
        (9201, 6, [
            (0, 1, 0.9, 0.85, 0.75, 1), (0, 2, 0.9, 0.84, 0.76, 1),
            (0, 3, 0.7, 0.65, 0.55, 2), (0, 4, 0.6, 0.56, 0.50, 0),
            (2, 5, 0.8, 0.79, 0.61, 3),
        ], [0, 2, 5], 2),
        (9202, 4, [
            (0, 1, 0.4, 0.30, 0.20, 1), (0, 2, 0.4, 0.31, 0.19, 1),
            (1, 3, 0.9, 0.88, 0.82, 3), (2, 3, 0.9, 0.87, 0.83, 3),
        ], [0, 1], 3),
        (9203, 7, [
            (0, 1, 0.1, 0.2, 0.3, 0), (2, 3, 0.1, None, 0.2, 0),
            (3, 4, 0.5, 0.45, None, 2), (5, 6, 0.5, 0.55, 0.65, 2),
        ], [2, 5, 1], 5),
    ],
)
def test_age_neighbors_match_sql_on_varied_graphs(conn, build_id, node_count, edges, seeds, cap):
    conn.execute("CREATE TEMP TABLE sect_build (build_id bigint PRIMARY KEY, n_nodes int, n_edges int)")
    conn.execute("CREATE TEMP TABLE sect_node (build_id bigint, ord int, PRIMARY KEY(build_id, ord))")
    conn.execute("CREATE TEMP TABLE sect_edge (build_id bigint, a int, b int, weight real, dense_sim real, sparse_sim real, sig smallint)")
    conn.execute("CREATE TEMP VIEW sect_edge_sym AS "
                 "SELECT build_id, a AS src, b AS dst, weight, dense_sim, sparse_sim, sig FROM sect_edge "
                 "UNION ALL SELECT build_id, b, a, weight, dense_sim, sparse_sim, sig FROM sect_edge")
    conn.execute("INSERT INTO sect_build VALUES (%s, %s, %s)", (build_id, node_count, len(edges)))
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO sect_node VALUES (%s, %s)", [(build_id, ord_) for ord_ in range(node_count)])
        cur.executemany(
            "INSERT INTO sect_edge VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [(build_id, *edge) for edge in edges],
        )

    age_graph.rebuild_from_relational(conn, build_id)
    assert age_graph.neighbours(conn, build_id, seeds, cap) == st.neighbours(conn, build_id, seeds, cap)


def test_age_bulk_loader_scales_and_matches_relational_counts(conn):
    """Check that server-side COPY feeds AGE's loader at the measured graph size.

    Require: `conn` is the isolated rollback-only AGE test connection.
    Guarantee: 81,473 vertices and 10,000 relationships load with exact counts.
    Maintain: no fixture rows, AGE graph, or temporary CSV files survive the test transaction.
    Assert: loaded counts match source counts and nullable scores remain null.

    Args:
        conn: Disposable PostgreSQL connection supplied by the test fixture.

    Returns:
        None.

    Raises:
        AssertionError: If loaded counts, null properties, or cleanup differ.

    Called by:
        pytest.

    Depends on:
        `age_graph.rebuild_from_relational` and AGE's graph label tables.
    """
    build_id, node_count, edge_count = 9204, 81473, 10000
    conn.execute("CREATE TEMP TABLE sect_build (build_id bigint PRIMARY KEY, n_nodes int, n_edges int)")
    conn.execute("CREATE TEMP TABLE sect_node (build_id bigint, ord int, PRIMARY KEY(build_id, ord))")
    conn.execute("CREATE TEMP TABLE sect_edge (build_id bigint, a int, b int, weight real, dense_sim real, sparse_sim real, sig smallint)")
    conn.execute("CREATE TEMP VIEW sect_edge_sym AS "
                 "SELECT build_id, a AS src, b AS dst, weight, dense_sim, sparse_sim, sig FROM sect_edge "
                 "UNION ALL SELECT build_id, b, a, weight, dense_sim, sparse_sim, sig FROM sect_edge")
    conn.execute("INSERT INTO sect_build VALUES (%s, %s, %s)", (build_id, node_count, edge_count))
    conn.execute(
        "INSERT INTO sect_node SELECT %s, ord FROM generate_series(0, %s - 1) AS ord",
        (build_id, node_count),
    )
    conn.execute(
        "INSERT INTO sect_edge "
        "SELECT %s, i - 1, i, 0.5, CASE WHEN i = 1 THEN NULL::real ELSE 0.4 END, "
        "CASE WHEN i = 2 THEN NULL::real ELSE 0.6 END, 3 "
        "FROM generate_series(1, %s) AS i",
        (build_id, edge_count),
    )

    age_graph.rebuild_from_relational(conn, build_id)
    assert conn.execute("SELECT count(*) FROM sect_node WHERE build_id = %s", (build_id,)).fetchone()[0] == node_count
    assert conn.execute("SELECT count(*) FROM sect_edge WHERE build_id = %s", (build_id,)).fetchone()[0] == edge_count
    with age_graph._age_context(conn):
        vertex_count = conn.execute(
            sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(age_graph.graph_name(build_id)), sql.Identifier("Section"),
            )
        ).fetchone()[0]
        relationship_count = conn.execute(
            sql.SQL("SELECT count(*) FROM {}.{}").format(
                sql.Identifier(age_graph.graph_name(build_id)), sql.Identifier("SIMILAR_TO"),
            )
        ).fetchone()[0]
        null_dense = conn.execute(
            sql.SQL("SELECT count(*) FROM {}.{} WHERE properties->'\"dense_sim\"'::ag_catalog.agtype = 'null'::ag_catalog.agtype").format(
                sql.Identifier(age_graph.graph_name(build_id)), sql.Identifier("SIMILAR_TO"),
            )
        ).fetchone()[0]
        null_sparse = conn.execute(
            sql.SQL("SELECT count(*) FROM {}.{} WHERE properties->'\"sparse_sim\"'::ag_catalog.agtype = 'null'::ag_catalog.agtype").format(
                sql.Identifier(age_graph.graph_name(build_id)), sql.Identifier("SIMILAR_TO"),
            )
        ).fetchone()[0]
    assert (vertex_count, relationship_count) == (node_count, edge_count)
    assert null_dense == 1
    assert null_sparse == 1
    staged_files = conn.execute(
        "SELECT count(*) FROM pg_ls_dir(%s) AS files(name) WHERE name LIKE %s",
        (age_graph.AGE_IMPORT_DIR, f"age_section_{build_id}_%"),
    ).fetchone()[0]
    assert staged_files == 0


def test_genre_vectors_are_stored_with_their_axes_and_seed_the_dense_arm_from_emb_b(conn):
    b, X = build(conn)
    XB = np.roll(X, 1, axis=0)                                           # a different geometry: section i's emb_b is section i-1's vector
    V = np.zeros((st.DIM, 2), np.float32)
    V[0, 0] = V[1, 1] = 1.0
    assert st.set_genre_vectors(conn, b, XB, V) == 6
    assert st.live_build(conn, "pytest_section_store")[1]["genre_axes"][0][0] == 1.0
    assert st.seeds_dense(conn, b, XB[3], 1, col="emb_b") == [3]
    assert st.seeds_dense(conn, b, X[3], 1, col="emb") == [3]
    assert st.seeds_dense(conn, b, X[2], 2, col="emb_b")[0] == 3         # the vector of section 2 now sits at section 3 in emb_b
    with pytest.raises(AssertionError):
        st.seeds_dense(conn, b, X[0], 1, col="text")


SURF = [("NapMem", 1), ("Metis", 1), ("Fig", 1)]
MENT = [(0, 0, 2), (2, 0, 1), (2, 1, 1), (3, 1, 4)] + [(o, 2, 1) for o in range(6)]          # entity 0 in sections 0,2; entity 1 in 2,3; entity 2 (a hub) in all six


def test_mentions_are_stored_with_their_document_frequency_and_a_second_set_replaces_the_first(conn):
    b, _ = build(conn)
    assert st.persist_mentions(conn, b, SURF, MENT) == (10, 3)
    assert dict(conn.execute("SELECT ent_id, df FROM sect_entity WHERE build_id = %s", (b,)).fetchall()) == {0: 2, 1: 2, 2: 6}
    assert st.persist_mentions(conn, b, SURF, [(0, 0, 1), (1, 0, 1)]) == (2, 1)
    assert dict(conn.execute("SELECT ent_id, df FROM sect_entity WHERE build_id = %s", (b,)).fetchall()) == {0: 2}
    assert conn.execute("SELECT count(*) FROM sect_mention WHERE build_id = %s", (b,)).fetchone()[0] == 2


def test_entity_walk_bridges_through_rare_entities_only_and_each_hop_starts_from_what_the_last_added(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    hops = st.entity_walk(conn, b, [0], hops=3, n=5, max_df=3)
    assert [[h["ord"] for h in hop] for hop in hops] == [[2], [3], []]                         # 0 -(NapMem)-> 2 -(Metis)-> 3; the hub bridges nothing; nothing is left at hop 3
    assert abs(hops[0][0]["score"] - np.log(6 / 2)) < 1e-9 and hops[0][0]["shared"] == 1
    assert [h["ord"] for h in st.entity_hop(conn, b, [0], [0], max_df=3, n=5)] == [2]


def test_entity_walk_with_the_hub_allowed_reaches_every_section_and_n_cuts_each_hop(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    hop1 = st.entity_walk(conn, b, [0], hops=1, n=5, max_df=6)[0]
    assert [h["ord"] for h in hop1] == [2, 1, 3, 4, 5]                                          # section 2 first (NapMem + hub), then the hub-only sections by ord
    assert [h["ord"] for h in st.entity_walk(conn, b, [0], hops=1, n=2, max_df=6)[0]] == [2, 1]
    assert st.entity_walk(conn, b, [], hops=2, n=5, max_df=6) == [[], []]                       # no subgraph, no hops


def test_arxiv_id_of_maps_doc_ids_including_methods_extracts_and_old_ids_and_books_to_none():
    assert st.arxiv_id_of("arxiv/2504_07891") == "2504.07891" and st.arxiv_id_of("arxiv/2606_17276_methods") == "2606.17276"
    assert st.arxiv_id_of("arxiv/hep-th_9901001") == "hep-th/9901001" and st.arxiv_id_of("arxiv/ISLP") is None


def test_titles_are_saved_once_the_first_source_wins_and_books_have_none(conn):
    st.ensure_schema(conn)                                    # ids 9999.* are synthetic: the live table holds the real papers' titles, which a test must not collide with
    new = st.save_titles(conn, [("9999.00001", "Speculative Decoding Survey", "A thesis.", "enriched.csv"), ("9999.00002", "DFlash", None, "enriched.csv")])
    assert new == 2
    assert st.save_titles(conn, [("9999.00001", "A different title", None, "api"), ("9999.00003", "New one", None, "api")]) == 1               # the held title is not overwritten
    assert dict(conn.execute("SELECT arxiv_id, title FROM paper_title WHERE arxiv_id LIKE '9999.%' ORDER BY arxiv_id").fetchall()) == {
        "9999.00001": "Speculative Decoding Survey", "9999.00002": "DFlash", "9999.00003": "New one"}
    assert conn.execute("SELECT thesis, source FROM paper_title WHERE arxiv_id = '9999.00001'").fetchone() == ("A thesis.", "enriched.csv")
    got = st.paper_titles(conn, ["arxiv/9999_00001", "arxiv/9999_00002_methods", "arxiv/ISLP", "arxiv/9999_99999"])
    assert got == {"arxiv/9999_00001": "Speculative Decoding Survey", "arxiv/9999_00002_methods": "DFlash"}                                   # a `_methods` extract shows its paper; a book and an unknown paper are absent


def test_community_catalogue_lists_draft_communities_largest_first(conn):
    b, _ = build(conn)
    assert st.community_catalogue(conn, b) == [(0, 2, "C0"), (1, 2, "C1"), (2, 2, "C2")]                  # equal sizes: by community
    conn.execute("UPDATE sect_community SET status = 'stale', size = 9 WHERE build_id = %s AND cid = 1", (b,))
    assert st.community_catalogue(conn, b) == [(0, 2, "C0"), (2, 2, "C2")]                                 # a community without a draft is not offered


def test_subgraph_entities_rank_by_sections_times_rarity_and_the_hub_comes_last(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    assert st.subgraph_entities(conn, b, [0, 2], max_df=6) == [("NapMem", 2), ("Metis", 1), ("Fig", 2)]      # 2 x ln(6/2) > 1 x ln(6/2) > 2 x ln(6/6) = 0
    assert st.subgraph_entities(conn, b, [0, 2], max_df=3) == [("NapMem", 2), ("Metis", 1)]                  # the hub is over the cap
    assert st.subgraph_entities(conn, b, [0, 2], top=1, max_df=6) == [("NapMem", 2)]
    assert st.subgraph_entities(conn, b, [], max_df=6) == []


def test_subgraph_entities_skip_a_surface_whose_words_contain_or_are_inside_a_better_one(conn):
    b, _ = build(conn)
    surf = [("speculative decoding", 2), ("speculative", 1), ("speculative decoding methods", 3), ("draft model", 2)]
    mention = [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1), (0, 2, 1), (1, 2, 1), (3, 3, 1), (0, 3, 1), (4, 3, 1)]
    st.persist_mentions(conn, b, surf, mention)
    # scores, N = 6: speculative decoding 2 x ln3 = 2.197; speculative decoding methods 2 x ln3 = 2.197 (tie, by surface after the shorter); draft model 3 x ln2 = 2.079; speculative 3 x ln2 = 2.079
    assert st.subgraph_entities(conn, b, [0, 1, 2, 3, 4], max_df=6, top=6) == [("speculative decoding", 2), ("draft model", 3)]      # the superset and the subset of the first are skipped


def test_entities_by_community_inside_a_subgraph_count_the_subgraphs_sections_only(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    # scores, N = 6: community 0 holds sections 0, 1: NapMem 1 x ln3 = 1.10, Fig 2 x ln1 = 0; community 1 holds 2, 3: Metis 2 x ln3 = 2.20, NapMem 1.10, Fig 0
    assert st.subgraph_entities_by_community(conn, b, [0, 1, 2, 3], max_df=6) == [
        {"cid": 0, "n": 2, "entities": [("NapMem", 1), ("Fig", 2)]}, {"cid": 1, "n": 2, "entities": [("Metis", 2), ("NapMem", 1), ("Fig", 2)]}]
    assert st.subgraph_entities_by_community(conn, b, [0, 1, 2, 3], max_df=3)[1]["entities"] == [("Metis", 2), ("NapMem", 1)]               # the hub is over the cap
    assert [c["cid"] for c in st.subgraph_entities_by_community(conn, b, [0, 1, 2, 3], top_comms=1, max_df=6)] == [0]
    assert st.subgraph_entities_by_community(conn, b, [2], max_df=6) == [{"cid": 1, "n": 1, "entities": [("Metis", 1), ("NapMem", 1), ("Fig", 1)]}]      # only section 2 counts: 1 each; Metis and NapMem tie at ln3 and break by surface, the hub scores 0


def test_community_entity_stats_count_covered_sections_mentions_and_distinct_entities(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    assert st.community_entity_stats(conn, b) == {0: {"size": 2, "covered": 2, "mentions": 4, "distinct": 2}, 1: {"size": 2, "covered": 2, "mentions": 8, "distinct": 3},
                                                  2: {"size": 2, "covered": 2, "mentions": 2, "distinct": 1}}
    assert st.community_entity_stats(conn, b, [1]) == {1: {"size": 2, "covered": 2, "mentions": 8, "distinct": 3}}
    st.persist_mentions(conn, b, SURF, [(0, 0, 1)])
    assert st.community_entity_stats(conn, b)[2] == {"size": 2, "covered": 0, "mentions": 0, "distinct": 0}                               # no mention: zeros, not absent


def test_community_entity_lines_print_the_top_stored_entities_with_their_counts(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    st.community_entities(conn, b, top=8, min_in=1)
    assert st.community_entity_lines(conn, b, top=3) == {0: "NapMem 1", 1: "Metis 2 · NapMem 1"}                                           # community 2 has none: absent
    assert st.community_entity_lines(conn, b, top=1) == {0: "NapMem 1", 1: "Metis 2"}


def test_shared_entities_name_what_two_sections_have_in_common_rarest_first_and_skip_the_hub(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    assert st.shared_entities(conn, b, 0, [2, 3, 1], max_df=3) == {2: ["NapMem"]}                 # 0 and 2 share NapMem; 3 and 1 share nothing with 0 once the hub is over the cap
    assert st.shared_entities(conn, b, 2, [0, 3], max_df=6) == {0: ["NapMem", "Fig"], 3: ["Metis", "Fig"]}      # rarest first: df 2 before the hub's 6
    assert st.shared_entities(conn, b, 2, [0, 3], max_df=6, top=1) == {0: ["NapMem"], 3: ["Metis"]}
    assert st.shared_entities(conn, b, 0, [], max_df=6) == {}


def test_entity_edges_keep_positive_npmi_pairs_only_and_a_hub_in_every_section_links_to_nothing(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    assert st.entity_edges(conn, b, min_shared=1, max_df=6) == 1                                # NapMem+Metis share section 2; the hub is independent of both (NPMI 0)
    a, bb, kind, n, npmi = conn.execute("SELECT a, b, kind, n_shared, npmi FROM sect_entity_edge WHERE build_id = %s", (b,)).fetchone()
    assert (a, bb, kind, n) == (0, 1, "co_mention", 1) and abs(npmi - np.log(1.5) / np.log(6)) < 1e-6
    assert st.entity_edges(conn, b, min_shared=1, max_df=6) == 1                                # a second call replaces, never doubles
    assert st.entity_edges(conn, b, min_shared=2, max_df=6) == 0                                # the one pair shares a single section
    assert st.entity_edges(conn, b, min_shared=1, max_df=1) == 0                                # every entity is over the cap


def test_community_entities_rank_by_g2_keep_over_represented_entities_only_and_leave_others_empty(conn):
    b, _ = build(conn)
    st.persist_mentions(conn, b, SURF, MENT)
    assert st.community_entities(conn, b, top=8, min_in=1) == 2
    got = dict(conn.execute("SELECT cid, entities FROM sect_community WHERE build_id = %s", (b,)).fetchall())
    assert [e[0] for e in got[1]] == ["Metis", "NapMem"]                                        # Metis is in both of community 1's sections and nowhere else; the hub (in every section) is not over-represented
    assert got[1][0][1:3] == [2, 2] and abs(got[1][0][3] - 7.6) < 1e-9 and abs(got[1][1][3] - 0.4) < 1e-9
    assert [e[0] for e in got[0]] == ["NapMem"] and got[2] is None
    assert st.community_entities(conn, b, top=1, min_in=1) == 2 and len(dict(conn.execute("SELECT cid, entities FROM sect_community WHERE build_id = %s", (b,)).fetchall())[1]) == 1
    assert st.community_entities(conn, b, top=8, min_in=3) == 0                                 # no entity reaches 3 sections of a community


def test_centrepoint_is_the_member_nearest_its_communitys_mean_and_similarity_is_cosine(conn):
    b, X = build(conn)
    XB = X.copy()
    XB[1] = XB[0] * 0.8 + XB[1] * 0.2                                    # community 0 = sections 0 and 1: make 1 closer to the pair's mean than 0 is
    XB[0] = XB[0] + 0.5 * XB[1]
    XB /= np.linalg.norm(XB, axis=1, keepdims=True)
    st.set_genre_vectors(conn, b, XB, np.zeros((st.DIM, 1), np.float32))
    cp = st.centrepoints(conn, b, [0, 1, 2], col="emb_b")
    mean0 = XB[:2].mean(0)
    assert cp[0] == int(np.argmax(XB[:2] @ mean0))                       # the member with the larger cosine to the mean
    assert set(cp) == {0, 1, 2} and cp[1] in (2, 3) and cp[2] in (4, 5)
    sim = st.similarity(conn, b, [0, 4], XB[0], col="emb_b")
    assert abs(sim[0] - 1.0) < 1e-5 and abs(sim[4] - float(XB[4] @ XB[0])) < 1e-5
    assert st.centrepoints(conn, b, [99], col="emb_b") == {}             # an unknown community is absent


def test_communities_by_summary_and_genre_exclusion(conn):
    b, _ = build(conn)
    q = np.zeros(st.DIM, np.float32)
    q[2] = 1.0                                                           # nearest to community 2's summary vector, which is genre-flagged
    assert st.communities_by_summary(conn, b, q, 1) == [2]
    assert st.communities_by_summary(conn, b, q, 3, exclude_genre=True) == [0, 1] or set(st.communities_by_summary(conn, b, q, 3, exclude_genre=True)) == {0, 1}
    assert 2 not in st.communities_by_summary(conn, b, q, 3, exclude_genre=True)


def test_lookups_return_whole_rows(conn):
    b, _ = build(conn)
    assert st.community_of(conn, b, [0, 3, 5]) == {0: 0, 3: 1, 5: 2}
    assert st.summaries(conn, b, [0, 2])[2]["title"] == "C2" and st.summaries(conn, b, [0, 2])[2]["genre_flag"] is True
    sec = st.sections(conn, b, [1])[1]
    assert sec["doc_id"] == "arxiv/p1" and sec["community"] == 0 and "verifies draft tokens" in sec["text"]
