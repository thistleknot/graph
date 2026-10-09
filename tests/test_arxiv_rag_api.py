"""Pins arxiv_rag_api.py: hybrid search that returns WHOLE sections (Q1-Q3), the reference filter (Q4), visible
degradation (Q5), and the HTTP surface OpenWebUI reads.

Real Postgres in a database this suite owns (`graph_sparsevec_test`); a fake query embedder; no GPU. The library is
two papers, one with a 400-paragraph section that the chunker splits into several chunks, so "whole section" has to
rebuild across continuation chunks.

Run:  pytest tests/test_arxiv_rag_api.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import arxiv_rag_api as api
import sparsevec_store as ss
from arxiv_graph_service import frozen
from domain_corpora import chunk_arxiv, with_junk_flag
from ingest_arxiv_sparsevec import B, K1, full_views
from salient_grams import bm25_matrix
from stoplist import tokenize

TEST_DB, LABEL = "graph_sparsevec_test", "zz_api"
FIT = {"lcap": 1992, "m": 10, "hi": 151}


def _paras(tag: str, n: int, special: dict | None = None) -> str:
    special = special or {}
    return "\n\n".join(special.get(i, "alpha graph %s paragraph %d community edge network" % (tag, i)) for i in range(n))


P1 = ("## Introduction\n\n" + _paras("intro", 8) + "\n\n## Long results\n\n"
      + _paras("filler", 400, {250: "alpha graph platypus zebrafish quokka spotted here"})
      + "\n\n## References\n\n[1] A. Author. A survey of platypus zebrafish quokka habitats. 2020.")
P2 = "## Sparse retrieval\n\n" + "\n\n".join("beta sparse vector index query ranking detail %d" % i for i in range(8))


def fake_query(text: str) -> np.ndarray:
    v = np.array([1.0, 0, 0, 0] if "alpha" in text else [0, 1.0, 0, 0], np.float32)
    return v


def fake_embed_docs(texts):
    return np.array([[1.0, 0, 0, 0] if "alpha" in t else [0, 1.0, 0, 0] for t in texts], np.float32)


@pytest.fixture(scope="module")
def conn():
    try:
        with psycopg.connect(ss.DSN, autocommit=True) as admin:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone():
                admin.execute(f'CREATE DATABASE "{TEST_DB}"')
        c = psycopg.connect(ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB, autocommit=True)
    except Exception as exc:                                            # pragma: no cover
        pytest.skip(f"no Postgres at {ss.DSN}: {exc}")
    build_library(c)
    yield c
    c.execute("DELETE FROM lex_build WHERE label = %s", (LABEL,))
    c.execute("DELETE FROM lex_chunk_meta WHERE label = %s", (LABEL,))
    ss.reset_label_tables(c, LABEL)
    c.close()


def build_library(c) -> None:
    recs, _, _ = chunk_arxiv(["arxiv/2601_0001", "arxiv/2601_0002"], [P1, P2], fit=FIT)
    with_junk_flag(recs)
    tokens = [[] if r["is_junk"] else tokenize(r["text"]) for r in recs]
    BM, TF, terms, df = bm25_matrix(tokens, K1, B)
    N, dim = BM.shape
    sat_t, cos_t, idf, whales = full_views(BM.tocsr(), df, N)
    c.execute("DELETE FROM lex_build WHERE label = %s", (LABEL,))
    ss.reset_label_tables(c, LABEL)
    ss.ensure_schema(c, LABEL, dim)
    ss.ensure_chunk_meta(c)
    ss.ensure_view_schema(c, LABEL, dim, 4)
    ss.ensure_build_schema(c)
    ss.write_vocab(c, LABEL, [str(t) for t in terms], idf=idf)
    ss.write_chunk_meta(c, LABEL, recs)

    def rows(M):
        return [(i, recs[i]["doc_id"], "arxiv", dict(zip(M.indices[M.indptr[i]:M.indptr[i + 1]].tolist(),
                                                           M.data[M.indptr[i]:M.indptr[i + 1]].tolist()))) for i in range(N)]
    ss.write_chunks(c, LABEL, dim, rows(sat_t))
    ss.write_chunks(c, LABEL, dim, rows(cos_t), kind="cos")
    keep = [i for i, r in enumerate(recs) if not r["is_reference"] and not r["is_junk"]]
    E = fake_embed_docs([recs[i]["text"] for i in keep])
    ss.write_dense(c, LABEL, keep, E)
    build = ss.new_build(c, LABEL, N, {
        "bm25": {"k1": K1, "b": B, "avgdl": float(np.mean([len(t) for t in tokens])), "n_docs": N, "dim": dim, "whale_cols": whales.tolist()},
        "fit": FIT, "dense_dim": 4, "dense": {"model": "fake", "dim": 4, "mean": [0.0, 0.0, 0.0, 0.0]}, "communities": {"n": 2}})
    ss.write_communities(c, build, [(0, 1, [1.0, 0, 0, 0]), (1, 1, [0, 1.0, 0, 0])])
    ss.write_assignments(c, build, [(i, int(np.argmax(e)), "consensus", 0.0, 0.0) for i, e in zip(keep, E)])
    ss.write_summaries(c, build, [{"community": 0, "status": "draft", "title": "Graph communities", "summary": "s", "chunk_keys": []},
                                  {"community": 1, "status": "draft", "title": "Sparse retrieval", "summary": "s", "chunk_keys": []}])


@pytest.fixture(scope="module")
def st(conn):
    return frozen(conn, LABEL)


# ---------------------------------------------------------------------- pure ----
def test_arxiv_url_links_papers_and_methods_extracts_and_leaves_books_alone():
    assert api.arxiv_url("arxiv/2404_08634") == "https://arxiv.org/abs/2404.08634"
    assert api.arxiv_url("arxiv/1301_3781") == "https://arxiv.org/abs/1301.3781"
    assert api.arxiv_url("arxiv/2606_17276_methods") == "https://arxiv.org/abs/2606.17276"
    assert api.arxiv_url("arxiv/hep-th_9901001") == "https://arxiv.org/abs/hep-th/9901001"
    assert api.arxiv_url("arxiv/math.GT_0309136") == "https://arxiv.org/abs/math.GT/0309136"
    assert api.arxiv_url("arxiv/Machine-Learning-Systems") is None and api.arxiv_url("arxiv/ISLP") is None


def test_rrf_rewards_agreement_between_arms_and_breaks_ties_by_item():
    fused = api.rrf([[7, 3, 9], [3, 5, 7]])
    assert [i for i, _ in fused][:2] == [3, 7]                         # in both lists beats in one
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61) and dict(fused)[9] == pytest.approx(1 / 63)
    assert api.rrf([[4], [2]]) == [(2, pytest.approx(1 / 61)), (4, pytest.approx(1 / 61))]    # equal scores: lower item first
    assert api.rrf([]) == [] and api.rrf([[]]) == []


def test_say_survives_a_character_the_console_code_page_cannot_encode(monkeypatch):
    import io
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", write_through=True))
    with pytest.raises(UnicodeEncodeError):
        print("policy π and ∑")                                # the crash this replaces
    api.say("policy π and ∑, plain text intact")
    assert b"policy \\u03c0 and \\u2211, plain text intact" in raw.getvalue()


def test_a_cut_is_reported_and_an_uncut_text_is_not():
    assert api._cut("abcdef", 4) == ("abcd", True, 6)
    assert api._cut("abcdef", 6) == ("abcdef", False, 6) and api._cut("abcdef", None) == ("abcdef", False, 6)


# -------------------------------------------------------------------- search ----
def test_a_hit_returns_the_whole_multi_chunk_section_not_just_the_matching_chunk(conn, st):
    out = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=3, mode="sparse", max_chars=None)
    top = out["results"][0]
    assert top["doc_id"] == "arxiv/2601_0001" and top["section_title"] == "Long results" and top["matched_by"] == ["sparse"]
    chunks, omitted = api.section_chunks(conn, LABEL, "arxiv/2601_0001", top["section_idx"])
    assert len(chunks) >= 3 and omitted == 0                          # the section really is split across chunks
    assert "filler paragraph 0 " in top["text"] and "filler paragraph 399 " in top["text"]      # both ends of it
    assert top["text"].count("## Long results") == 1                  # continuation headers stripped
    assert len(top["matched_ords"]) == 1 and not top["truncated"] and top["chars_total"] == len(top["text"])
    assert top["url"] == "https://arxiv.org/abs/2601.0001" and top["version"] == 1
    assert out["warnings"] == [] and out["build"] == st["build"]


def test_the_sparse_arm_returns_only_chunks_that_contain_a_query_term(conn, st):
    n_chunks = conn.execute("SELECT count(*) FROM lex_chunk_meta WHERE label = %s", (LABEL,)).fetchone()[0]
    hits = api.sparse_ranked(conn, LABEL, st, "platypus zebrafish quokka", pool=50)
    assert n_chunks > 5 and 1 <= len(hits) < n_chunks                 # not padded out to the pool with zero-score chunks
    texts = {o: t for o, t in conn.execute("SELECT ord, text FROM lex_chunk_meta WHERE label = %s", (LABEL,)).fetchall()}
    assert all(any(w in texts[o] for w in ("platypus", "zebrafish", "quokka")) for o in hits)


def test_references_and_junk_are_left_out_unless_asked_for(conn, st):
    default = api.search(conn, LABEL, st, "platypus zebrafish quokka survey habitats", k=5, mode="sparse")
    assert default["results"] and not any(r["is_reference"] or r["section_title"] == "References" for r in default["results"])
    asked = api.search(conn, LABEL, st, "platypus zebrafish quokka survey habitats", k=5, mode="sparse", include_references=True)
    refs = [r for r in asked["results"] if r["is_reference"]]
    assert refs and refs[0]["section_title"] == "References" and "A survey of platypus" in refs[0]["text"]


def test_hits_in_one_section_fold_into_one_result_and_k_bounds_the_count(conn, st):
    out = api.search(conn, LABEL, st, "alpha graph community edge network", k=2, mode="sparse")
    keys = [(r["doc_id"], r["section_idx"]) for r in out["results"]]
    assert len(keys) == len(set(keys)) and len(keys) <= 2
    assert [r["rank"] for r in out["results"]] == list(range(1, len(keys) + 1))


def test_the_default_returns_the_entire_section_not_a_prefix_of_it(conn, st):
    whole = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=1, mode="sparse", max_chars=None)["results"][0]
    default = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=1, mode="sparse")["results"][0]
    assert api.DEFAULT_MAX_CHARS == 200_000 and 20_000 < whole["chars_total"] < api.DEFAULT_MAX_CHARS      # ~22k chars here
    assert default["text"] == whole["text"] and default["truncated"] is False                              # the old 20k cut is gone
    assert "filler paragraph 399 " in default["text"]


def test_a_junk_chunk_inside_a_section_is_left_out_of_its_text_and_counted(conn, st):
    top = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=1, mode="sparse", max_chars=None)["results"][0]
    chunks, _ = api.section_chunks(conn, LABEL, top["doc_id"], top["section_idx"])
    victim = next(c for c in chunks if c[1] > 0 and "platypus" not in c[2])      # a continuation chunk that is not the hit itself
    marker = next(p for p in victim[2].split("\n\n")[1:] if p.strip())[:45]      # a paragraph carrying a unique number
    assert "filler paragraph" in marker
    assert marker in top["text"]
    conn.execute("UPDATE lex_chunk_meta SET is_junk = true WHERE label = %s AND ord = %s", (LABEL, victim[0]))
    try:
        again = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=1, mode="sparse", max_chars=None)["results"][0]
        assert again["junk_chunks_omitted"] == 1 and marker not in again["text"]
        assert again["chars_total"] < top["chars_total"] and "filler paragraph 399 " in again["text"]
        assert api.get_section(conn, LABEL, top["doc_id"], top["section_idx"])["junk_chunks_omitted"] == 1
    finally:
        conn.execute("UPDATE lex_chunk_meta SET is_junk = false WHERE label = %s AND ord = %s", (LABEL, victim[0]))


def test_a_cut_text_says_so_and_keeps_the_full_length(conn, st):
    out = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=1, mode="sparse", max_chars=500)
    top = out["results"][0]
    assert len(top["text"]) == 500 and top["truncated"] is True and top["chars_total"] > 5000


def test_hybrid_fuses_both_arms_and_marks_which_ranked_each_hit(conn, st):
    out = api.search(conn, LABEL, st, "beta sparse vector index query ranking", k=3, mode="hybrid", embed_query=fake_query)
    top = out["results"][0]
    assert top["doc_id"] == "arxiv/2601_0002" and top["matched_by"] == ["sparse", "dense"]
    assert top["community"] == {"cid": 1, "how": "consensus", "title": "Sparse retrieval", "status": "draft"}
    dense_only = api.search(conn, LABEL, st, "beta anything", k=3, mode="dense", embed_query=fake_query)
    assert dense_only["results"][0]["doc_id"] == "arxiv/2601_0002" and dense_only["results"][0]["matched_by"] == ["dense"]


def test_the_community_of_a_hit_is_its_placement_and_its_title_a_draft(conn, st):
    out = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=1, mode="sparse")
    assert out["results"][0]["community"] == {"cid": 0, "how": "consensus", "title": "Graph communities", "status": "draft"}


def test_a_dense_arm_that_cannot_run_is_reported_and_hybrid_still_answers_from_the_sparse_arm(conn, st):
    def broken(_):
        raise RuntimeError("model not loadable")
    hybrid = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=2, mode="hybrid", embed_query=broken)
    assert hybrid["results"] and hybrid["results"][0]["matched_by"] == ["sparse"]
    assert any("dense arm unavailable (RuntimeError: model not loadable)" in w and "sparse arm" in w for w in hybrid["warnings"])
    dense = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=2, mode="dense", embed_query=broken)
    assert dense["results"] == [] and any("dense arm unavailable" in w for w in dense["warnings"])


def test_a_query_with_no_known_term_says_the_sparse_arm_is_empty_and_hybrid_falls_back_to_dense(conn, st):
    sparse = api.search(conn, LABEL, st, "qqqq zzzz wwww", k=3, mode="sparse")
    assert sparse["results"] == [] and any("vocabulary" in w for w in sparse["warnings"])
    hybrid = api.search(conn, LABEL, st, "qqqq zzzz wwww", k=3, mode="hybrid", embed_query=fake_query)
    assert hybrid["results"] and all(r["matched_by"] == ["dense"] for r in hybrid["results"])
    assert any("vocabulary" in w for w in hybrid["warnings"])


def test_get_section_returns_the_same_text_search_does_and_none_for_an_unknown_key(conn, st):
    top = api.search(conn, LABEL, st, "platypus zebrafish quokka", k=1, mode="sparse")["results"][0]
    got = api.get_section(conn, LABEL, top["doc_id"], top["section_idx"])
    assert got["text"] == top["text"] and got["section_title"] == "Long results" and len(got["chunk_ords"]) >= 3
    assert api.get_section(conn, LABEL, "arxiv/2601_0001", 9999) is None and api.get_section(conn, LABEL, "arxiv/nope", 0) is None
    assert api.get_section(conn, LABEL, top["doc_id"], top["section_idx"], max_chars=600)["truncated"] is True


# ------------------------------------------------------------------- the API ----
@pytest.fixture(scope="module")
def client(conn):
    from fastapi.testclient import TestClient
    app = api.create_app(label=LABEL, dsn=ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB, embed_query=fake_query)
    with TestClient(app) as c:
        yield c


def test_the_search_endpoint_returns_whole_sections_in_the_documented_shape(client):
    r = client.get("/search", params={"query": "platypus zebrafish quokka", "k": 2, "mode": "sparse", "max_chars": 100000})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"query", "mode", "build", "results", "warnings"}
    top = body["results"][0]
    assert top["doc_id"] == "arxiv/2601_0001" and "filler paragraph 399 " in top["text"]
    assert set(top) >= {"rank", "score", "doc_id", "version", "url", "section_idx", "section_title", "text", "truncated",
                        "chars_total", "matched_by", "matched_ords", "community", "is_reference"}


def test_the_endpoints_refuse_bad_parameters_and_unknown_sections(client):
    assert client.get("/search", params={"query": "x", "k": 0}).status_code == 422
    assert client.get("/search", params={"query": "x", "k": api.MAX_K + 1}).status_code == 422
    assert client.get("/search", params={"query": "x", "mode": "bogus"}).status_code == 422
    assert client.get("/search", params={"query": "x", "max_chars": 10}).status_code == 422
    assert client.get("/search", params={"query": "x", "max_chars": api.CEILING_MAX_CHARS + 1}).status_code == 422
    assert client.get("/search", params={"query": ""}).status_code == 422
    assert client.get("/section", params={"doc_id": "arxiv/2601_0001", "section_idx": 9999}).status_code == 404
    assert client.get("/section", params={"doc_id": "arxiv/2601_0001", "section_idx": -1}).status_code == 422


def test_the_section_health_and_community_endpoints(client):
    top = client.get("/search", params={"query": "platypus zebrafish quokka", "mode": "sparse", "k": 1}).json()["results"][0]
    sec = client.get("/section", params={"doc_id": top["doc_id"], "section_idx": top["section_idx"]}).json()
    assert sec["text"] == top["text"]
    health = client.get("/health").json()
    assert health["label"] == LABEL and health["papers"] == 2 and health["chunks"] >= 5
    rows = client.get("/communities").json()
    assert {r["title"] for r in rows} == {"Graph communities", "Sparse retrieval"} and rows == sorted(rows, key=lambda r: -r["size"])


def test_openwebui_can_read_the_surface_from_openapi(client):
    spec = client.get("/openapi.json").json()
    ops = {p: {m: d["operationId"] for m, d in item.items()} for p, item in spec["paths"].items()}
    assert ops == {"/search": {"get": "search_arxiv"}, "/section": {"get": "get_section"},
                   "/communities": {"get": "list_communities"}, "/health": {"get": "health"}}
    assert "WHOLE SECTION" in spec["info"]["description"]
    params = {p["name"]: p for p in spec["paths"]["/search"]["get"]["parameters"]}
    assert params["query"]["required"] is True and params["mode"]["schema"]["enum"] == ["hybrid", "sparse", "dense"]


def test_a_service_with_no_live_build_says_so_with_a_503(conn):
    from fastapi.testclient import TestClient
    app = api.create_app(label="zz_api_nobuild", dsn=ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB, embed_query=fake_query)
    with TestClient(app) as c:
        r = c.get("/search", params={"query": "x"})
        assert r.status_code == 503 and "no live build" in r.json()["detail"]


def test_a_rebuild_is_picked_up_without_a_restart(conn, client):
    before = client.get("/health").json()["build"]
    stats = conn.execute("SELECT n_chunks, params FROM lex_build WHERE label = %s AND live", (LABEL,)).fetchone()
    import json
    newer = ss.new_build(conn, LABEL, stats[0], json.loads(json.dumps(stats[1])) if not isinstance(stats[1], str) else json.loads(stats[1]))
    try:
        ss.write_communities(conn, newer, [(0, 1, [1.0, 0, 0, 0])])
        assert client.get("/health").json()["build"] == newer != before
    finally:
        conn.execute("DELETE FROM lex_build WHERE build_id = %s", (newer,))
        conn.execute("UPDATE lex_build SET live = true WHERE build_id = %s", (before,))
