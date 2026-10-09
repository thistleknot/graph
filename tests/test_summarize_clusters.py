"""Pins summarize_clusters.py's never-truncate guards (S1-S4), its reply parser and its
failure isolation.

The claim the whole tool rests on: every exemplar chunk reaches the model in full, and any
silent cut (a window that is too small, a capped answer) becomes a loud, recorded failure.
No network: the API call is replaced where it is exercised.

Run:  pytest tests/test_summarize_clusters.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import summarize_clusters as sc
from summarize_clusters import MAX_FILL, MAX_TOKENS, assert_not_truncated, build_prompt, collapse_ws, fits, parse

NONWS = lambda s: re.sub(r"\s+", "", s)


def test_collapse_ws_shrinks_padding_and_keeps_every_non_whitespace_character():
    padded = "| a" + " " * 5000 + "| b" + " " * 4000 + "|   \n## Next\n\ttext\t\tmore"
    out = collapse_ws(padded)
    assert len(out) < 80 and NONWS(out) == NONWS(padded)


def test_build_prompt_contains_every_chunk_in_full_even_a_huge_one():
    big = "## T\n\n" + " ".join("word%d" % i for i in range(60000))         # ~400k chars, no cutting allowed
    ex = [{"doc_id": "arxiv/1", "section_title": "T", "text": big},
          {"doc_id": "arxiv/2", "section_title": "U", "text": "## U\n\nsmall chunk body"}]
    p = build_prompt(7, 123, ex, "2 distinct documents; the largest, 1, supplies 90% of the chunks.")
    assert NONWS(big) in NONWS(p) and "word59999" in p and "small chunk body" in p
    assert "cluster 7; 123 chunks in all" in p and "chunk 1 of 2" in p
    assert "supplies 90% of the chunks" in p


def test_provenance_states_the_measured_paper_spread_for_the_model():
    one = sc.provenance({"n_papers": 1, "top_paper": "arxiv/Machine-Learning-Systems", "top_share": 1.0})
    assert one == "1 distinct document; the largest, Machine-Learning-Systems, supplies 100% of the chunks."
    many = sc.provenance({"n_papers": 412, "top_paper": "arxiv/2404_08634", "top_share": 0.011})
    assert many.startswith("412 distinct documents;") and "1% of the chunks" in many


def test_fits_refuses_a_prompt_the_window_cannot_hold_instead_of_cutting_it():
    window = 100_000
    limit_chars = int((MAX_FILL * window - MAX_TOKENS) * 2)
    assert fits(1000, window) and fits(limit_chars, window)
    assert not fits(limit_chars + 10, window)
    assert not fits(10 ** 7, window)


def test_a_prompt_that_fills_the_window_is_a_failure_not_a_summary():
    ok = {"finish_reason": "stop", "prompt_tokens": 2000}
    assert_not_truncated(ok, 8192)
    with pytest.raises(AssertionError, match="truncate silently"):
        assert_not_truncated({**ok, "prompt_tokens": 8192}, 8192)
    with pytest.raises(AssertionError, match="truncate silently"):
        assert_not_truncated({**ok, "prompt_tokens": int(0.91 * 8192)}, 8192)


def test_a_capped_answer_is_a_failure():
    with pytest.raises(AssertionError, match="answer cut"):
        assert_not_truncated({"finish_reason": "length", "prompt_tokens": 10}, 8192)


def test_parse_takes_the_last_block_not_a_draft_earlier_in_the_reply():
    reply = "TITLE: Draft title\nSUMMARY: A first attempt.\n\nTITLE: Final title\nSUMMARY: The real answer."
    assert parse(reply) == ("Final title", "The real answer.")


def test_parse_reads_title_and_summary_and_rejects_anything_else():
    t, s = parse("TITLE: Retrieval-augmented generation\nSUMMARY: Papers on RAG.\nThey cover retrievers.")
    assert t == "Retrieval-augmented generation" and s == "Papers on RAG. They cover retrievers."
    with pytest.raises(AssertionError, match="TITLE/SUMMARY"):
        parse("This cluster is about transformers.")


def test_log_survives_a_title_the_console_codepage_cannot_encode(capsys):
    sc.log("community 7 | τ-bench and ∑ notation")
    out = capsys.readouterr().out
    assert "\\u03c4-bench" in out and "\\u2211" in out


def test_load_inputs_indexes_the_same_list_the_map_does_so_a_row_is_the_chunk_the_map_showed(tmp_path, monkeypatch):
    import json
    import pickle
    recs = [{"doc_id": "arxiv/%d" % i, "section_idx": 0, "chunk_idx": 0, "section_title": "T",
             "is_reference": i == 1, "text": "## T\n\n" + ("ordinary prose number %d about graphs" % i)} for i in range(5)]
    recs[2]["text"] = "## S\n\n" + "\n\n".join("abcdefghij"[k % 10] for k in range(30))        # junk, flag absent
    pkl, exj = tmp_path / "c.pkl", tmp_path / "e.json"
    pickle.dump({"records": recs}, open(pkl, "wb"))
    exj.write_text(json.dumps({"0": {"exemplars": [{"row": 2, "key": ["arxiv/3", 0, 0]}]}}), encoding="utf-8")
    monkeypatch.setattr(sc, "CACHE", str(pkl))
    monkeypatch.setattr(sc, "EXEMPLARS", str(exj))
    got, ex = sc.load_inputs()
    assert [r["doc_id"] for r in got] == ["arxiv/0", "arxiv/3", "arxiv/4"]          # reference and junk are out
    assert got[ex[0]["exemplars"][0]["row"]]["doc_id"] == "arxiv/4"                  # row 2 of THIS list


def test_persist_stores_only_summaries_written_for_the_builds_current_exemplars():
    import psycopg
    import sparsevec_store as ss
    test_db = "graph_sparsevec_test"
    try:
        with psycopg.connect(ss.DSN, autocommit=True) as admin:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_db,)).fetchone():
                admin.execute(f'CREATE DATABASE "{test_db}"')
        conn = psycopg.connect(ss.DSN.rsplit("/", 1)[0] + "/" + test_db, autocommit=True)
    except Exception as exc:                                              # pragma: no cover
        pytest.skip("no Postgres: %s" % exc)
    label = "zz_summ"
    ss.ensure_build_schema(conn)
    conn.execute("DELETE FROM lex_build WHERE label = %s", (label,))
    with pytest.raises(AssertionError, match="no live build"):
        sc.persist({}, {}, label=label, conn=conn)
    build = ss.new_build(conn, label, 3, {})
    exem = {0: {"exemplars": [{"key": ["arxiv/a", 0, 0]}]}, 1: {"exemplars": [{"key": ["arxiv/b", 1, 0]}]}}
    done = {0: {"community": 0, "status": "draft", "title": "Kept", "summary": "s", "model": "m", "provider": "p",
                "prompt_tokens": 10, "output_tokens": 4, "cost": 0.001, "chunk_keys": [["arxiv/a", 0, 0]]},
            1: {"community": 1, "status": "draft", "title": "Stale", "summary": "s", "model": "m",
                "chunk_keys": [["arxiv/OLD", 9, 9]]},                                   # written for other exemplars
            7: {"community": 7, "status": "draft", "title": "Gone", "summary": "s", "model": "m",
                "chunk_keys": [["arxiv/a", 0, 0]]}}                                      # a community this build no longer has
    assert sc.persist(done, exem, label=label, conn=conn) == 1
    assert conn.execute("SELECT cid, title, completion_tokens, status FROM lex_summary WHERE build_id = %s", (build,)).fetchall() \
        == [(0, "Kept", 4, "draft")]
    assert sc.persist(done, exem, label=label, conn=conn) == 1                         # a second run replaces, never doubles
    assert conn.execute("SELECT count(*) FROM lex_summary WHERE build_id = %s", (build,)).fetchone()[0] == 1
    conn.execute("DELETE FROM lex_build WHERE label = %s", (label,))


def _ex():
    return [["arxiv/1", 2, 0]]                                          # the chunk keys a community's exemplars carry


def test_summarize_one_records_a_draft_with_its_chunk_keys(monkeypatch):
    monkeypatch.setattr(sc, "chat", lambda m, p: {"content": "TITLE: X\nSUMMARY: Y.", "finish_reason": "stop", "prompt_tokens": 50,
                                                  "completion_tokens": 9, "cost": 0.0001, "provider": "Google"})
    r = sc.summarize_one("m", 1_000_000, 5, 10, _ex(), "a prompt")
    assert (r["status"], r["title"], r["summary"], r["chunk_keys"]) == ("draft", "X", "Y.", [["arxiv/1", 2, 0]])


def test_one_failing_community_is_recorded_and_does_not_raise(monkeypatch):
    def boom(m, p):
        raise RuntimeError("api down")
    monkeypatch.setattr(sc, "chat", boom)
    r = sc.summarize_one("m", 1_000_000, 5, 10, _ex(), "a prompt")
    assert r["status"] == "failed" and "api down" in r["reason"]


def test_the_dunning_terms_reach_the_prompt_and_only_the_ones_that_survive_into_the_reply_are_bolded_T164(monkeypatch):
    ex = [{"doc_id": "arxiv/1", "section_title": "T", "text": "## T\n\nbody"}]
    terms = ["retrieval", "reinforcement learning", "rag", "kv cache"]
    p = build_prompt(3, 10, ex, "prov", terms)
    assert "retrieval, reinforcement learning, rag, kv cache" in p and "exactly as written" in p
    assert "Dunning" not in build_prompt(3, 10, ex, "prov")                           # no terms (chunk mode): the prompt is as it was
    seen = {}
    def fake(m, prompt):
        seen["prompt"] = prompt
        return {"content": "TITLE: Retrieval for LLMs\nSUMMARY: Covers Retrieval-augmented generation (RAG), and reinforcement learning of the retriever.",
                "finish_reason": "stop", "prompt_tokens": 50, "completion_tokens": 20, "cost": 0.0, "provider": "p"}
    monkeypatch.setattr(sc, "chat", fake)
    r = sc.summarize_one("m", 1_000_000, 3, 10, _ex(), p, terms)
    assert r["status"] == "draft" and "Dunning terms" in seen["prompt"]
    assert r["title_bold"] == "**Retrieval** for LLMs"
    assert r["summary_bold"] == "Covers **Retrieval**-augmented generation (**RAG**), and **reinforcement learning** of the retriever."
    assert r["terms_surviving"] == ["retrieval", "reinforcement learning", "rag"] and "kv cache" not in r["summary_bold"]
    assert r["summary"] == "Covers Retrieval-augmented generation (RAG), and reinforcement learning of the retriever."   # the raw text stays beside the bolded one


def test_bold_terms_takes_whole_words_once_longest_first_and_changes_nothing_without_a_match():
    from term_salience import bold_terms
    assert bold_terms("graph neural networks and graph cuts", ["graph", "graph neural networks"]) == (
        "**graph neural networks** and **graph** cuts", ["graph", "graph neural networks"])
    assert bold_terms("paragraphs", ["graph"]) == ("paragraphs", [])                     # inside a word is not a match
    assert bold_terms("nothing here", []) == ("nothing here", [])
    assert bold_terms("The GNN", ["gnn"])[0] == "The **GNN**"                              # case-insensitive, shown as written


def test_an_oversized_prompt_is_recorded_not_summarized_and_never_sent(monkeypatch):
    monkeypatch.setattr(sc, "chat", lambda m, p: pytest.fail("an oversized prompt must not be sent"))
    r = sc.summarize_one("m", 8192, 5, 10, _ex(), "x" * 100_000)
    assert r["status"] == "not_summarized" and "never truncated" in r["reason"]
