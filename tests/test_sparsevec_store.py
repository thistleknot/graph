"""Pins sparsevec_store.py's V7 and V8 guards: the junk flag and text hash on chunk metadata, the
appending ingest, the cosine and dense views, and the build record with its communities,
assignments, exemplars and draft summaries.

Real Postgres, in a database this suite owns (`graph_sparsevec_test`), so no dev table is touched.
Skipped, with the reason, only when no server answers.

Run:  pytest tests/test_sparsevec_store.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sparsevec_store as ss

TEST_DB = "graph_sparsevec_test"
LABEL, SDIM, DDIM = "zz_store", 12, 4


def _dsn() -> str:
    return ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB


@pytest.fixture(scope="module")
def conn():
    try:
        with psycopg.connect(ss.DSN, autocommit=True) as admin:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone():
                admin.execute(f'CREATE DATABASE "{TEST_DB}"')
        c = psycopg.connect(_dsn(), autocommit=True)
    except Exception as exc:                                            # pragma: no cover
        pytest.skip(f"no Postgres at {ss.DSN}: {exc}")
    ss.ensure_schema(c, LABEL, SDIM)
    ss.ensure_chunk_meta(c)
    ss.ensure_view_schema(c, LABEL, SDIM, DDIM)
    ss.ensure_build_schema(c)
    c.execute("DELETE FROM lex_build WHERE label LIKE %s", (LABEL + "%",))        # the database outlives a run
    yield c
    c.execute("DELETE FROM lex_build WHERE label LIKE %s", (LABEL + "%",))
    c.close()


def _rec(doc, si, ci, text, junk=False, ref=False):
    return {"doc_id": doc, "section_idx": si, "chunk_idx": ci, "section_title": "T", "is_reference": ref,
            "is_junk": junk, "text": text}


def test_meta_keeps_the_junk_flag_and_hashes_the_text_and_append_continues_the_ords(conn):
    first = ss.write_chunk_meta(conn, LABEL, [_rec("arxiv/a", 0, 0, "alpha text"), _rec("arxiv/a", 1, 0, "x", junk=True)])
    assert first == 2
    start = ss.append_chunk_meta(conn, LABEL, [_rec("arxiv/b", 0, 0, "beta text"), _rec("arxiv/b", 1, 0, "gamma", ref=True)])
    assert start == 2
    rows = conn.execute("SELECT ord, doc_id, is_junk, is_reference, text_md5 = md5(text) FROM lex_chunk_meta "
                        "WHERE label = %s ORDER BY ord", (LABEL,)).fetchall()
    assert rows == [(0, "arxiv/a", False, False, True), (1, "arxiv/a", True, False, True),
                    (2, "arxiv/b", False, False, True), (3, "arxiv/b", False, True, True)]
    assert ss.stored_versions(conn, LABEL) == {"arxiv/a": 1, "arxiv/b": 1}                  # no version given: v1


def test_known_text_md5_finds_stored_texts_only_and_tolerates_empty_input(conn):
    ss.write_chunk_meta(conn, LABEL, [_rec("arxiv/a", 0, 0, "alpha text"), _rec("arxiv/a", 1, 0, "beta text")])
    import hashlib
    h = lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()
    assert ss.known_text_md5(conn, LABEL, ["alpha text", "never stored", "beta text"]) == {h("alpha text"), h("beta text")}
    assert ss.known_text_md5(conn, LABEL, ["never stored"]) == set()
    assert ss.known_text_md5(conn, LABEL, []) == set()
    assert ss.known_text_md5(conn, "another_label", ["alpha text"]) == set()


def test_the_version_is_a_secondary_key_stored_with_each_chunk_and_read_back_as_the_highest_held(conn):
    recs = [{**_rec("arxiv/p", 0, 0, "p one"), "version": 3}, {**_rec("arxiv/p", 1, 0, "p two"), "version": 3},
            {**_rec("arxiv/q", 0, 0, "q one"), "version": 2}, _rec("arxiv/r", 0, 0, "r one")]
    ss.write_chunk_meta(conn, LABEL, recs)
    assert ss.stored_versions(conn, LABEL) == {"arxiv/p": 3, "arxiv/q": 2, "arxiv/r": 1}
    assert conn.execute("SELECT version FROM lex_chunk_meta WHERE label = %s AND doc_id = 'arxiv/p'", (LABEL,)).fetchall() == [(3,), (3,)]
    assert ss.stored_versions(conn, "never_ingested") == {}


def test_a_paper_is_overwritten_wholly_across_every_table_and_nothing_else_is_touched(conn):
    build = ss.new_build(conn, LABEL + "_ow", 4, {})
    ss.write_chunk_meta(conn, LABEL, [_rec("arxiv/old", 0, 0, "old zero"), _rec("arxiv/old", 1, 0, "old one"),
                                      _rec("arxiv/keep", 0, 0, "keep zero"), _rec("arxiv/keep", 1, 0, "keep one")])
    for kind in ("chunk", "cos"):
        ss.write_chunks(conn, LABEL, SDIM, [(o, "arxiv/old" if o < 2 else "arxiv/keep", "arxiv", {o: 1.0}) for o in range(4)], kind=kind)
    ss.write_dense(conn, LABEL, range(4), np.eye(4))
    ss.write_assignments(conn, build, [(o, 0, "consensus", 0.0, 0.0) for o in range(4)])
    ss.write_exemplars(conn, build, [(0, 0, "medoid", 0.0, 1), (0, 1, "z=1/2", 0.5, 3)])
    assert ss.delete_papers(conn, LABEL, ["arxiv/old"], build) == 2
    meta = conn.execute("SELECT ord FROM lex_chunk_meta WHERE label = %s ORDER BY ord", (LABEL,)).fetchall()
    assert meta == [(2,), (3,)]
    for kind in ("chunk", "cos", "dense"):
        assert conn.execute(f"SELECT ord FROM {ss._table(LABEL, kind)} ORDER BY ord").fetchall() == [(2,), (3,)], kind
    assert conn.execute("SELECT ord FROM lex_assign WHERE build_id = %s ORDER BY ord", (build,)).fetchall() == [(2,), (3,)]
    assert conn.execute("SELECT ord FROM lex_exemplar WHERE build_id = %s", (build,)).fetchall() == [(3,)]
    assert ss.delete_papers(conn, LABEL, ["arxiv/old"], build) == 0                                  # idempotent
    assert ss.delete_papers(conn, LABEL, [], build) == 0
    assert ss.append_chunk_meta(conn, LABEL, [_rec("arxiv/old", 0, 0, "new zero")]) == 4              # after the highest kept ord
    ss.delete_papers(conn, LABEL, ["arxiv/old"], build)                                                 # the top of the range is empty again
    assert ss.append_chunk_meta(conn, LABEL, [_rec("arxiv/old", 0, 0, "again")]) == 4
    assert ss.append_chunk_meta(conn, LABEL, [_rec("arxiv/new", 0, 0, "floor")], min_ord=10) == 10      # never below the build's size
    conn.execute("DELETE FROM lex_chunk_meta WHERE label = %s AND doc_id IN ('arxiv/old', 'arxiv/new')", (LABEL,))
    conn.execute("DELETE FROM lex_build WHERE label = %s", (LABEL + "_ow",))


def test_delete_papers_tolerates_a_label_whose_view_tables_were_never_made(conn):
    lab = LABEL + "_bare"
    ss.write_chunk_meta(conn, lab, [_rec("arxiv/a", 0, 0, "alone")])
    assert ss.delete_papers(conn, lab, ["arxiv/a"]) == 1
    assert ss.stored_versions(conn, lab) == {}


def test_normalising_moves_a_version_out_of_the_doc_id_and_refuses_to_merge_two_papers(conn):
    ss.write_chunk_meta(conn, LABEL, [_rec("arxiv/2403_19889v1", 0, 0, "a0"), _rec("arxiv/2403_19889v1", 1, 0, "a1"),
                                      _rec("arxiv/2404_00001v3_methods", 0, 0, "m0"), _rec("arxiv/2405_10739", 0, 0, "plain"),
                                      _rec("arxiv/Machine-Learning-Systems", 0, 0, "book"), _rec("arxiv/draft_v2", 0, 0, "odd")])
    ss.write_chunks(conn, LABEL, SDIM, [(0, "arxiv/2403_19889v1", "arxiv", {0: 1.0}), (2, "arxiv/2404_00001v3_methods", "arxiv", {1: 1.0})])
    assert ss.normalise_versioned_doc_ids(conn, LABEL) == 3
    assert ss.stored_versions(conn, LABEL) == {"arxiv/2403_19889": 1, "arxiv/2404_00001_methods": 3, "arxiv/2405_10739": 1,
                                               "arxiv/Machine-Learning-Systems": 1, "arxiv/draft_v2": 1}
    assert conn.execute(f"SELECT ord, doc_id FROM {ss._table(LABEL)} ORDER BY ord").fetchall() == [(0, "arxiv/2403_19889"), (2, "arxiv/2404_00001_methods")]
    assert ss.normalise_versioned_doc_ids(conn, LABEL) == 0                                           # nothing left to repair
    ss.write_chunk_meta(conn, LABEL, [_rec("arxiv/2601_00007v2", 0, 0, "v2 text"), _rec("arxiv/2601_00007", 0, 0, "bare text")])
    with pytest.raises(AssertionError, match="would merge stored papers"):
        ss.normalise_versioned_doc_ids(conn, LABEL)
    assert set(ss.stored_versions(conn, LABEL)) == {"arxiv/2601_00007v2", "arxiv/2601_00007"}          # untouched on refusal


def test_append_adds_rows_and_replace_empties_the_table_first(conn):
    rows = lambda ords: [(o, "arxiv/a", "arxiv", {o % SDIM: 1.0 + o}) for o in ords]
    assert ss.write_chunks(conn, LABEL, SDIM, rows([0, 1, 2]))["written"] == 3
    assert ss.write_chunks(conn, LABEL, SDIM, rows([3, 4]), replace=False)["written"] == 2
    assert conn.execute(f"SELECT count(*) FROM {ss._table(LABEL)}").fetchone()[0] == 5
    ss.write_chunks(conn, LABEL, SDIM, rows([9]))
    assert conn.execute(f"SELECT count(*) FROM {ss._table(LABEL)}").fetchone()[0] == 1


def test_each_view_gets_the_operator_class_its_job_needs_and_cosine_returns_the_row_itself_first(conn):
    ss.write_chunks(conn, LABEL, SDIM, [(i, "arxiv/a", "arxiv", {i: 1.0, (i + 1) % SDIM: 0.5}) for i in range(6)])
    ss.write_chunks(conn, LABEL, SDIM, [(i, "arxiv/a", "arxiv", {i: 1.0, (i + 3) % SDIM: 0.25}) for i in range(6)],
                    kind="cos")
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(6, DDIM))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert ss.write_dense(conn, LABEL, range(6), emb) == 6
    for kind, ops in (("chunk", "sparsevec_ip_ops"), ("cos", "sparsevec_cosine_ops"), ("dense", "vector_cosine_ops")):
        ss.create_hnsw(conn, LABEL, kind=kind)
        d = conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = %s AND indexname LIKE '%%_hnsw'",
                         (ss._table(LABEL, kind),)).fetchone()[0]
        assert ops in d, (kind, d)
    conn.execute("SET enable_indexscan = off")
    near = conn.execute(f"SELECT ord FROM {ss._table(LABEL, 'dense')} ORDER BY emb <=> %s::vector LIMIT 1",
                        ("[" + ",".join(map(str, emb[4])) + "]",)).fetchone()[0]
    assert near == 4
    cos_near = conn.execute(f"SELECT ord FROM {ss._table(LABEL, 'cos')} ORDER BY vec <=> "
                            f"(SELECT vec FROM {ss._table(LABEL, 'cos')} WHERE ord = 2) LIMIT 1").fetchone()[0]
    assert cos_near == 2


def test_a_rebuilt_vocabulary_can_change_the_dimension_after_the_tables_are_reset(conn):
    lab = LABEL + "_reset"
    ss.ensure_schema(conn, lab, 8)
    ss.ensure_view_schema(conn, lab, 8, DDIM)
    ss.write_chunks(conn, lab, 8, [(0, "arxiv/a", "arxiv", {0: 1.0})])
    with pytest.raises(psycopg.errors.DataException):                   # the old dimension rejects a wider vocabulary
        ss.write_chunks(conn, lab, 20, [(1, "arxiv/a", "arxiv", {19: 1.0})], replace=False)
    ss.reset_label_tables(conn, lab)
    for kind in ("chunk", "cos", "dense"):
        assert conn.execute("SELECT to_regclass(%s)", (ss._table(lab, kind),)).fetchone()[0] is None
    ss.ensure_schema(conn, lab, 20)
    assert ss.write_chunks(conn, lab, 20, [(1, "arxiv/a", "arxiv", {19: 1.0})])["written"] == 1
    ss.reset_label_tables(conn, lab)


def test_a_new_build_supersedes_the_live_one_without_deleting_it(conn):
    assert ss.live_build(conn, LABEL + "_b") is None
    b1 = ss.new_build(conn, LABEL + "_b", 100, {"resolution": 6.16})
    b2 = ss.new_build(conn, LABEL + "_b", 140, {"resolution": 5.0})
    assert b2 > b1
    assert ss.live_build(conn, LABEL + "_b") == (b2, 140, {"resolution": 5.0})
    assert conn.execute("SELECT live FROM lex_build WHERE build_id = %s", (b1,)).fetchone()[0] is False
    assert ss.live_build(conn, LABEL + "_other") is None


def test_stages_add_their_params_to_one_build_and_clear_derived_keeps_the_summaries(conn):
    b = ss.new_build(conn, LABEL + "_p", 3, {"bm25": {"avgdl": 80.5}})
    merged = ss.update_build_params(conn, b, {"dense": {"dim": 4}, "resolution": 6.16})
    assert merged == {"bm25": {"avgdl": 80.5}, "dense": {"dim": 4}, "resolution": 6.16}
    assert ss.update_build_params(conn, b, {"resolution": 5.0})["resolution"] == 5.0
    ss.write_communities(conn, b, [(0, 3, None)])
    ss.write_assignments(conn, b, [(0, 0, "consensus", 0.0, 0.0)])
    ss.write_exemplars(conn, b, [(0, 0, "medoid", 0.0, 0)])
    ss.write_summaries(conn, b, [{"community": 0, "status": "draft", "title": "t", "summary": "s", "chunk_keys": []}])
    ss.clear_derived(conn, b)
    count = lambda t: conn.execute(f"SELECT count(*) FROM {t} WHERE build_id = %s", (b,)).fetchone()[0]
    assert (count("lex_community"), count("lex_assign"), count("lex_exemplar"), count("lex_summary")) == (0, 0, 0, 1)
    with pytest.raises(AssertionError):
        ss.update_build_params(conn, b + 10_000, {"x": 1})


def test_the_derived_rows_round_trip_and_the_checks_hold(conn):
    b = ss.new_build(conn, LABEL + "_d", 6, {})
    ss.write_communities(conn, b, [(0, 4, [0.5, 0.5, 0.0, 0.0]), (1, 2, None)])
    ss.write_assignments(conn, b, [(0, 0, "consensus", 0.1, 0.2), (1, 1, "consensus", 1.5, -2.0),
                                   (2, 0, "nearest-centroid", None, None)])
    ss.write_exemplars(conn, b, [(0, 0, "medoid", 0.0, 0), (0, 1, "z=1/2", 0.48, 2)])
    n = ss.write_summaries(conn, b, [
        {"community": 0, "status": "draft", "title": "A title", "summary": "A summary.", "model": "m", "provider": "p",
         "prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001, "chunk_keys": [["arxiv/a", 0, 0], ["arxiv/a", 1, 0]]},
        {"community": 1, "status": "failed", "reason": "api down", "chunk_keys": [["arxiv/b", 0, 0]]}])
    assert n == 2
    assert conn.execute("SELECT how, count(*) FROM lex_assign WHERE build_id = %s GROUP BY how ORDER BY how",
                        (b,)).fetchall() == [("consensus", 2), ("nearest-centroid", 1)]
    assert conn.execute("SELECT vector_dims(centroid) FROM lex_community WHERE build_id = %s AND cid = 0", (b,)).fetchone()[0] == 4
    s = conn.execute("SELECT status, title, chunk_keys FROM lex_summary WHERE build_id = %s AND cid = 0", (b,)).fetchone()
    assert s == ("draft", "A title", [["arxiv/a", 0, 0], ["arxiv/a", 1, 0]])
    with pytest.raises(psycopg.errors.CheckViolation):
        ss.write_assignments(conn, b, [(3, 0, "guessed", None, None)])
    with pytest.raises(psycopg.errors.CheckViolation):
        ss.write_summaries(conn, b, [{"community": 2, "status": "approved", "chunk_keys": []}])
    conn.execute("DELETE FROM lex_build WHERE build_id = %s", (b,))                    # the children go with the build
    assert conn.execute("SELECT count(*) FROM lex_summary WHERE build_id = %s", (b,)).fetchone()[0] == 0
