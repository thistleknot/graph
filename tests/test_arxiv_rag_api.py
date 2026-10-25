"""| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| WARM | Verify API startup warms the build, dense encoder, and cross-encoder. | [mandatory] | T4 |
| SEARCH | Pin hybrid whole-section retrieval and reference filtering. | [mandatory] | T1 |
| AGENT | Verify selected-model forwarding and separate text-bearing evidence records. | [mandatory] | T2 |
| AGGREGATE | Verify restart recovery reuses committed provider responses through the API. | [mandatory] | T6 |
| HEALTH | Verify live build changes and health metadata. | [mandatory] | T3 |
| BOUND | Verify model failures return explicit HTTP errors and close both connections. | [mandatory] | T5 |

Real Postgres in a database this suite owns (`graph_sparsevec_test`); a fake query embedder; no GPU. The library is
two papers, one with a 400-paragraph section that the chunker splits into several chunks, so "whole section" has to
rebuild across continuation chunks.

T1 Search returns complete sections, omits references by default, and reports sparse/dense degradation.
T2 `/agent` forwards the selected model, returns the final answer separately, and places bounded passage text on evidence rows rather than metadata-only sections.
T3 Health reports the active build; replacing it is picked up on the next request.
T4 Startup verifies the live build and runs both local retrieval models before readiness.
T5 Budget exhaustion returns 504; empty and unfinished model output returns 502.
T6 A restarted API recovers an incomplete job, resumes it within its original budget, and does not repeat committed calls.

Run:  pytest tests/test_arxiv_rag_api.py -v
Preconditions: test PostgreSQL is available. Failure modes: schema, query, agent, or warmup contract drift fails here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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


def test_agent_request_keeps_sql_default_and_accepts_age_backend():
    assert api.AgentRequest(question="graph query").graph_backend == "sql"
    assert api.AgentRequest(question="graph query", graph_backend="age").graph_backend == "age"
    with pytest.raises(ValueError):
        api.AgentRequest(question="graph query", graph_backend="missing")


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
    build_library(c, "arxiv_sect_bpe")
    c.execute("CREATE TABLE IF NOT EXISTS sect_build (build_id bigserial PRIMARY KEY, tag text NOT NULL, live boolean NOT NULL DEFAULT true, params jsonb NOT NULL DEFAULT '{}')")
    c.execute("INSERT INTO sect_build (tag, params) VALUES ('xpa', '{}'::jsonb)")
    yield c
    c.execute("DELETE FROM lex_build WHERE label = %s", (LABEL,))
    c.execute("DELETE FROM lex_build WHERE label = %s", ("arxiv_sect_bpe",))
    c.execute("DELETE FROM lex_chunk_meta WHERE label = %s", (LABEL,))
    c.execute("DELETE FROM lex_chunk_meta WHERE label = %s", ("arxiv_sect_bpe",))
    ss.reset_label_tables(c, LABEL)
    ss.reset_label_tables(c, "arxiv_sect_bpe")
    c.execute("DELETE FROM sect_build WHERE tag = 'xpa'")
    c.close()


def build_library(c, label=LABEL) -> None:
    recs, _, _ = chunk_arxiv(["arxiv/2601_0001", "arxiv/2601_0002"], [P1, P2], fit=FIT)
    with_junk_flag(recs)
    tokens = [[] if r["is_junk"] else tokenize(r["text"]) for r in recs]
    BM, TF, terms, df = bm25_matrix(tokens, K1, B)
    N, dim = BM.shape
    sat_t, cos_t, idf, whales = full_views(BM.tocsr(), df, N)
    c.execute("DELETE FROM lex_build WHERE label = %s", (label,))
    ss.reset_label_tables(c, label)
    ss.ensure_schema(c, label, dim)
    ss.ensure_chunk_meta(c)
    ss.ensure_view_schema(c, label, dim, 4)
    ss.ensure_build_schema(c)
    ss.write_vocab(c, label, [str(t) for t in terms], idf=idf)
    ss.write_chunk_meta(c, label, recs)

    def rows(M):
        return [(i, recs[i]["doc_id"], "arxiv", dict(zip(M.indices[M.indptr[i]:M.indptr[i + 1]].tolist(),
                                                           M.data[M.indptr[i]:M.indptr[i + 1]].tolist()))) for i in range(N)]
    ss.write_chunks(c, label, dim, rows(sat_t))
    ss.write_chunks(c, label, dim, rows(cos_t), kind="cos")
    keep = [i for i, r in enumerate(recs) if not r["is_reference"] and not r["is_junk"]]
    E = fake_embed_docs([recs[i]["text"] for i in keep])
    ss.write_dense(c, label, keep, E)
    build = ss.new_build(c, label, N, {
        "bm25": {"k1": K1, "b": B, "avgdl": float(np.mean([len(t) for t in tokens])), "n_docs": N, "dim": dim, "whale_cols": whales.tolist()},
        "fit": FIT, "dense_dim": 4, "dense": {"model": "fake", "dim": 4, "mean": [0.0, 0.0, 0.0, 0.0]}, "communities": {"n": 2}})
    c.execute("UPDATE lex_chunk_meta SET build_id = %s WHERE label = %s", (build, label))
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
    import section_store as store
    store.ensure_schema(conn)
    graph_build = conn.execute("SELECT build_id FROM sect_build WHERE tag = 'xpa' AND live").fetchone()[0]
    conn.execute(
        "UPDATE lex_build SET params = params || jsonb_build_object('source_graph_build', %s) "
        "WHERE label = 'arxiv_sect_bpe' AND live",
        (graph_build,),
    )
    conn.execute(
        "INSERT INTO sect_community (build_id,cid,size,title,summary,status) VALUES "
        "(%s,0,2,'Graph communities','s','draft'),(%s,1,2,'Sparse retrieval','s','draft') "
        "ON CONFLICT (build_id,cid) DO UPDATE SET title=excluded.title,summary=excluded.summary,status=excluded.status",
        (graph_build, graph_build),
    )
    app = api.create_app(label="arxiv_sect_bpe", dsn=ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB,
                         warm_start=False, bm25_dsn=ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB)
    with TestClient(app) as c:
        yield c


def test_startup_warms_database_dense_encoder_and_cross_encoder(conn, monkeypatch):
    from fastapi.testclient import TestClient
    import section_graphrag as graph

    calls = []

    def warm_cross(question, texts):
        calls.append(("cross", question, texts))
        return [0.0] * len(texts)

    class WarmGraph:
        build = 123

        def __init__(self, conn, tag, chunk_label=None, bm25_conn=None):
            assert tag == "xpa" and chunk_label == LABEL and bm25_conn is not None

        def embed(self, text):
            calls.append(("graph", text))
            return fake_query(text)

    monkeypatch.setattr(graph, "SectionGraph", WarmGraph)
    monkeypatch.setattr(graph, "cross_scores", warm_cross)
    app = api.create_app(label=LABEL, dsn=ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB,
                         bm25_dsn=ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
    assert calls == [
        ("graph", "Open WebUI startup warmup"),
        ("cross", "Open WebUI startup warmup", ["Warmup evidence section."]),
    ]


def test_the_search_endpoint_returns_whole_sections_in_the_documented_shape(client, conn, monkeypatch):
    monkeypatch.setattr(api, "search_section_graph", lambda db, bm25, query, k, mode, refs, max_chars, chunk_label:
                        api.search(db, LABEL, frozen(db, LABEL), query, k, mode, refs, max_chars, fake_query))
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


def test_the_section_health_and_community_endpoints(client, monkeypatch):
    monkeypatch.setattr(api, "search_section_graph", lambda db, bm25, query, k, mode, refs, max_chars, chunk_label:
                        api.search(db, LABEL, frozen(db, LABEL), query, k, mode, refs, max_chars, fake_query))
    monkeypatch.setattr(api, "get_section_from_graph", lambda db, bm25, doc_id, section_idx, max_chars, chunk_label:
                        api.get_section(db, LABEL, doc_id, section_idx, max_chars))
    top = client.get("/search", params={"query": "platypus zebrafish quokka", "mode": "sparse", "k": 1}).json()["results"][0]
    sec = client.get("/section", params={"doc_id": top["doc_id"], "section_idx": top["section_idx"]}).json()
    assert sec["text"] == top["text"]
    health = client.get("/health").json()
    assert health["label"] == "arxiv_sect_bpe" and health["papers"] == 2 and health["chunks"] >= 5
    rows = client.get("/communities").json()
    assert {r["title"] for r in rows} == {"Graph communities", "Sparse retrieval"} and rows == sorted(rows, key=lambda r: -r["size"])


def test_openwebui_can_read_the_surface_from_openapi(client):
    spec = client.get("/openapi.json").json()
    ops = {p: {m: d["operationId"] for m, d in item.items()} for p, item in spec["paths"].items()}
    assert ops == {"/search": {"get": "search_arxiv"}, "/agent": {"post": "answer_with_section_graph"},
                   "/section": {"get": "get_section"},
                   "/communities": {"get": "list_communities"}, "/community-terms": {"get": "list_community_terms"},
                   "/health": {"get": "health"}}
    assert "WHOLE SECTION" in spec["info"]["description"]
    params = {p["name"]: p for p in spec["paths"]["/search"]["get"]["parameters"]}
    assert params["query"]["required"] is True and params["mode"]["schema"]["enum"] == ["hybrid", "sparse", "dense"]


@pytest.mark.parametrize("backend", ["sql", "age"])
def test_agent_endpoint_returns_json_safe_graph_report_and_selected_model(client, monkeypatch, backend):
    import section_graphrag as graph
    import section_store as store
    import summarize_clusters

    calls = {}

    class FakeConnection:
        def close(self):
            calls["closed"] = calls.get("closed", 0) + 1

    class FakeGraph:
        def __init__(self, conn, tag, bm25_conn=None, graph_backend="sql"):
            self.conn, self.build = conn, 91
            calls["tag"] = tag
            calls["bm25_conn"] = bm25_conn is not None
            calls["graph_backend"] = graph_backend

    def fake_agent(sg, question, **kwargs):
        calls["question"] = question
        calls["model"] = kwargs["model"]
        calls["agent_model"] = kwargs["agent_model"]
        calls["deadline"] = kwargs["deadline"]
        return {
            "question": question,
            "seeds": [42],
            "subgraph": [{"ord": 42}, {"ord": 43}],
            "partials": [[7, "draft"]],
            "partials_rejected": [],
            "global_answer": "global",
            "final": "final",
            "evidence": [{"key": "arxiv/2401_00001#4", "ord": 42, "role": "read by the agent"}],
            "traversal": {"by_community": [{"cid": 7, "n": 1, "entities": []}]},
            "section_scores": [{"ord": 42, "key": "arxiv/2401_00001#4", "community": 7, "score": np.float32(0.5)}],
            "section_threshold": 0.0,
            "question_rejected": "",
            "agent": {"nodes": {"k": {"ord": 42}}, "trace": [], "stop": "answered", "read": {"k": "large"}},
            "qd": np.array([0.1, 0.2]),
        }

    monkeypatch.setattr(api.ss, "connect", lambda *args, **kwargs: FakeConnection())
    monkeypatch.setattr(graph, "SectionGraph", FakeGraph)
    monkeypatch.setattr(graph, "agent_answer", fake_agent)
    monkeypatch.setattr(summarize_clusters, "chat", lambda model, prompt, **kwargs: (
        calls.update(rewrite_model=model, rewrite_prompt=prompt, rewrite_deadline=kwargs["deadline"]) or {"content": "standalone graph question"}
    ))
    monkeypatch.setattr(store, "sections", lambda conn, build, ords: {
        42: {"doc_id": "arxiv/2401_00001", "section_idx": 4, "section_title": "Graph methods",
             "community": 7, "text": "Evidence"},
        43: {"doc_id": "arxiv/2401_00002", "section_idx": 5, "section_title": "Neighbor",
             "community": 7, "text": "Not returned as evidence"},
    })
    monkeypatch.setattr(store, "paper_titles", lambda conn, doc_ids: {
        "arxiv/2401_00001": "Paper title", "arxiv/2401_00002": "Neighbor paper"
    })
    monkeypatch.setattr(store, "summaries", lambda conn, build, cids: {
        7: {"community": 7, "title": "Graph retrieval", "size": 1, "summary": "draft"}
    })

    response = client.post("/agent", json={
        "question": "What about that?",
        "model": "openrouter/google/test",
        "history": [{"role": "user", "content": "Earlier graph question"}],
        "graph_backend": backend,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "standalone graph question"
    assert body["original_question"] == "What about that?" and body["final"] == "final"
    assert body["section_scores"][0]["score"] == 0.5
    assert body["sections"]["42"]["paper_title"] == "Paper title"
    assert body["evidence"][0]["text"] == "Evidence"
    assert body["evidence"][0]["truncated"] is False
    assert "text" not in body["sections"]["42"]
    assert "text" not in body["sections"]["43"]
    assert body["communities"]["7"]["title"] == "Graph retrieval"
    assert body["graph_backend"] == backend and calls["graph_backend"] == backend
    assert "read" not in body["agent"]
    assert calls["tag"] == "xpa" and calls["model"] == calls["agent_model"] == "google/test"
    assert calls["question"] == body["question"] and calls["rewrite_model"] == "google/test"
    assert "Earlier graph question" in calls["rewrite_prompt"] and calls["closed"] == 2
    assert calls["bm25_conn"] is True
    assert calls["deadline"] == calls["rewrite_deadline"]


def test_agent_age_missing_graph_fails_without_sql_fallback(client, monkeypatch):
    import section_graphrag as graph

    closed = []

    class Connection:
        def close(self):
            closed.append(True)

    def unavailable(conn, tag, bm25_conn=None, graph_backend="sql"):
        assert graph_backend == "age"
        raise RuntimeError("AGE graph arxiv_section_91 is missing")

    monkeypatch.setattr(api.ss, "connect", lambda *args, **kwargs: Connection())
    monkeypatch.setattr(graph, "SectionGraph", unavailable)
    response = client.post("/agent", json={"question": "graph query", "graph_backend": "age"})

    assert response.status_code == 503
    assert "AGE graph arxiv_section_91 is missing" in response.json()["detail"]
    assert len(closed) == 2


@pytest.mark.parametrize("failure,status,detail", [
    (TimeoutError("budget exhausted"), 504, "180-second request budget"),
    (ValueError("empty model reply"), 502, "empty model reply"),
    (ValueError("finish_reason=length"), 502, "finish_reason=length"),
])
def test_agent_reports_model_failure_and_closes_both_connections(monkeypatch, failure, status, detail):
    """A4. Guarantee: upstream failures stay explicit and close request resources."""
    import section_graphrag as graph
    from fastapi.testclient import TestClient
    closed, deadlines = [], []

    class Connection:
        def close(self):
            closed.append(True)

    def fail(sg, question, **kwargs):
        deadlines.append(kwargs["deadline"])
        raise failure

    monkeypatch.setattr(api.ss, "connect", lambda *a, **kw: Connection())
    monkeypatch.setattr(graph, "SectionGraph", lambda *a, **kw: object())
    monkeypatch.setattr(graph, "agent_answer", fail)
    with TestClient(api.create_app(warm_start=False)) as client:
        response = client.post("/agent", json={"question": "A bounded research question"})
    assert response.status_code == status and detail in response.json()["detail"]
    assert len(closed) == 2 and len(deadlines) == 1


def test_a_service_with_no_live_build_says_so_with_a_503(conn):
    from fastapi.testclient import TestClient
    app = api.create_app(label="zz_api_nobuild", dsn=ss.DSN.rsplit("/", 1)[0] + "/" + TEST_DB,
                         warm_start=False)
    with TestClient(app) as c:
        r = c.get("/search", params={"query": "x"})
        assert r.status_code == 503 and "build is missing" in r.json()["detail"]


def test_a_rebuild_is_picked_up_without_a_restart(conn, client):
    before = client.get("/health").json()["build"]
    stats = conn.execute("SELECT n_chunks, params FROM lex_build WHERE label = %s AND live", ("arxiv_sect_bpe",)).fetchone()
    import json
    newer = ss.new_build(conn, "arxiv_sect_bpe", stats[0], json.loads(json.dumps(stats[1])) if not isinstance(stats[1], str) else json.loads(stats[1]))
    try:
        ss.write_communities(conn, newer, [(0, 1, [1.0, 0, 0, 0])])
        assert client.get("/health").json()["build"] == newer != before
    finally:
        conn.execute("DELETE FROM lex_build WHERE build_id = %s", (newer,))
        conn.execute("UPDATE lex_build SET live = true WHERE build_id = %s", (before,))


def test_global_aggregate_job_submits_pins_and_completes_without_provider_access(monkeypatch, tmp_path):
    """A5: API pins the live population and returns durable result state."""
    import global_aggregate as ga
    import time
    from fastapi.testclient import TestClient
    from aggregate_jobs import AggregateJobs

    class Connection:
        def close(self):
            pass

    population = {"paper-a": 1, "paper-b": 1, "paper-c": 2, "paper-d": 2}
    monkeypatch.setattr(api.ss, "connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(api, "AggregateJobs", lambda *_args, **_kwargs: AggregateJobs(tmp_path / "job.sqlite"))
    monkeypatch.setattr(api.store, "live_build", lambda *_args: (456, {}))
    monkeypatch.setattr(api, "aggregate_population", lambda *_args: population)
    monkeypatch.setattr(ga, "load_paper_texts", lambda _conn, _build, ids, _cap: {
        paper_id: {"text": paper_id, "truncated": False} for paper_id in ids
    })
    monkeypatch.setattr(ga, "classify_predicate", lambda *_args, **_kwargs: {"label": "yes"})
    with TestClient(api.create_app(warm_start=False)) as client:
        response = client.post("/agent", json={
            "question": "How many papers contain an abstract?",
            "answer_kind": "auto",
        })
        assert response.status_code == 202, response.text
        receipt = response.json()["aggregate_job"]
        assert response.headers["location"] == f"/aggregate/{receipt['job_id']}"
        assert receipt["manifest"]["source_build_id"] == 456
        assert receipt["manifest"]["seed"] == 0
        assert receipt["manifest"]["draw_order"]
        deadline = time.monotonic() + 5
        snapshot = client.get(f"/aggregate/{receipt['job_id']}").json()
        while snapshot["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = client.get(f"/aggregate/{receipt['job_id']}").json()
        assert snapshot["status"] == "completed", snapshot
        assert snapshot["result"]["calls"] <= 4
        assert snapshot["result"]["interval"]


def test_auto_aggregate_intent_requires_explicit_global_count_wording():
    """Q16: count phrasing dispatches to tally while ordinary research stays retrieval."""
    assert api.infer_aggregate_predicate("How many papers discuss graph attention?") == "graph attention"
    assert api.infer_aggregate_predicate("What percentage of papers use retrieval augmented generation?") == \
        "retrieval augmented generation"
    assert api.infer_aggregate_predicate("How do papers discuss graph attention?") is None


def test_api_restart_resume_reuses_completed_provider_calls(monkeypatch, tmp_path):
    """T6: API startup exposes an interrupted job and resume completes without provider repetition."""
    import hashlib
    import json
    import random
    import sqlite3
    import time
    import global_aggregate as ga
    from aggregate_jobs import AggregateJobs
    from fastapi.testclient import TestClient

    class Connection:
        def close(self):
            pass

    population = {"paper-a": 1, "paper-b": 1, "paper-c": 2, "paper-d": 2}
    source_id = 456
    seed = 21
    draw_order = {}
    for community_id in sorted(set(population.values())):
        ids = sorted(pid for pid, cid in population.items() if cid == community_id)
        random.Random(f"{seed}:{community_id}").shuffle(ids)
        draw_order[str(community_id)] = ids
    population_digest = hashlib.sha256("\n".join(
        f"{pid}\t{population[pid]}" for pid in sorted(population)).encode()).hexdigest()
    classifier_version = ga.interpret.OPENROUTER_MODEL + ":" + ga.PREDICATE_PROMPT_VERSION
    manifest = {
        "source_build_id": source_id,
        "population_sha256": population_digest,
        "population_size": len(population),
        "community_count": 2,
        "seed": seed,
        "draw_order": draw_order,
        "budget": 4,
        "pilot_budget": 4,
        "target_halfwidth": 0.02,
        "classifier_version": classifier_version,
        "prompt_version": ga.PREDICATE_PROMPT_VERSION,
    }
    database_path = tmp_path / "jobs.sqlite"
    jobs = AggregateJobs(database_path)
    job_id = jobs.create({"predicate": "discuss graph attention", "model": None}, manifest)
    for paper_id in sorted(population):
        paper = {"text": paper_id, "truncated": False}
        call_key = f"{paper_id}:{ga._semantic_label_digest(paper)}"
        assert jobs.call(job_id, call_key, lambda: {"label": "yes"}) == {"label": "yes"}
    with sqlite3.connect(database_path) as db:
        db.execute("UPDATE aggregate_job SET status='running' WHERE job_id=?", (job_id,))
    jobs.close()

    monkeypatch.setattr(api.ss, "connect", lambda *_args, **_kwargs: Connection())
    monkeypatch.setattr(api.store, "live_build", lambda *_args: (source_id, {}))
    monkeypatch.setattr(api, "aggregate_population", lambda *_args: population)
    monkeypatch.setattr(ga, "load_paper_texts", lambda _conn, _build, ids, _cap: {
        paper_id: {"text": paper_id, "truncated": False} for paper_id in ids
    })
    provider_calls = []
    monkeypatch.setattr(ga, "classify_predicate", lambda *_args, **_kwargs:
                        provider_calls.append("unexpected") or {"label": "no"})
    with TestClient(api.create_app(warm_start=False, aggregate_jobs_path=database_path)) as client:
        recovered = client.get(f"/aggregate/{job_id}").json()
        assert recovered["status"] == "incomplete"
        assert recovered["calls_completed"] == 4
        assert client.post(f"/aggregate/{job_id}/resume").status_code == 202
        deadline = time.monotonic() + 5
        recovered = client.get(f"/aggregate/{job_id}").json()
        while recovered["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
            recovered = client.get(f"/aggregate/{job_id}").json()
        assert recovered["status"] == "completed", json.dumps(recovered)
        assert recovered["calls_completed"] == 4
        assert recovered["result"]["calls"] == 4
        assert recovered["result"]["provider_attempts_this_run"] == 0
        assert recovered["result"]["durable_responses_reused"] == 4
    assert provider_calls == []
