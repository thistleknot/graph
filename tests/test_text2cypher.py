"""tests/test_text2cypher.py -- text2cypher.py's loop tested offline: `call=` fakes
the model backend, `post=` fakes export_neo4j._tx, both injectable so generate ->
check_read_only -> execute -> retry runs with no database, no neo4j and no API key.
One live test exercises the real mirror and skips cleanly when it is unavailable.

Spec: .spec/specs/graph-explorer/design.md 6.18 (Q1-Q7)
Task: playbook.md T20
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import export_neo4j
import text2cypher
from text2cypher import (MAX_ATTEMPTS, ask, build_prompt, check_read_only,
                          load_fewshots)


def _fake_call(*replies):
    """Fake `call=`: returns each reply in turn (a dict is JSON-dumped, a str is sent
    raw), recording every (system, user) pair it saw on .prompts."""
    it = iter(replies)
    prompts = []

    def call(system, user, timeout, **kw):
        prompts.append((system, user))
        reply = next(it)
        text = json.dumps(reply) if isinstance(reply, dict) else reply
        return text, "fake-backend"

    call.prompts = prompts
    return call


def _fake_post(*outcomes):
    """Fake `post=`: each outcome is either an exception to raise or a
    (columns, rows) pair returned in _tx's payload shape. Records statements on
    .statements -- if it is never called for an attempt, that attempt never went
    over the wire."""
    it = iter(outcomes)
    statements = []

    def post(stmts, url=None, auth=None, db=None):
        statements.append(stmts)
        outcome = next(it)
        if isinstance(outcome, BaseException):
            raise outcome
        columns, rows = outcome
        return [{"columns": columns, "data": [{"row": list(r)} for r in rows]}]

    post.statements = statements
    return post


# ------------------------------------------------------------------- cookbook parsing

def test_the_cookbook_parses_into_six_shots():
    shots = load_fewshots()
    assert len(shots) == 6
    for s in shots:
        assert s["question"]
        assert "MATCH" in s["query"]
        assert "//" not in s["query"]
        assert not s["query"].endswith(";")
    assert shots[0]["question"].startswith("What are the strongest highlighted paths")


def test_every_cookbook_query_passes_the_read_only_check():
    shots = load_fewshots()
    assert len(shots) == 6
    for s in shots:
        assert check_read_only(s["query"]) is None, (s["n"], s["query"])


# ------------------------------------------------------------------ read-only checks

@pytest.mark.parametrize("query", [
    "MATCH (c:Chunk) MERGE (d:Chunk {id: c.id}) RETURN d",
    "CREATE (c:Chunk {id: '1'}) RETURN c",
    "MATCH (c:Chunk) DELETE c",
    "MATCH (c:Chunk) DETACH DELETE c",
    "MATCH (c:Chunk) SET c.text = 'x' RETURN c",
    "MATCH (c:Chunk) REMOVE c.text RETURN c",
    "MATCH (c:Chunk) RETURN c.id DROP INDEX chunk_embedding",
    "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
    "CALL apoc.periodic.iterate('MATCH (c) RETURN c', 'DELETE c', {})",
])
def test_write_clauses_are_rejected(query):
    reason = check_read_only(query)
    assert reason is not None


def test_a_write_word_inside_a_string_literal_is_not_rejected():
    query = "MATCH (c:Chunk) WHERE c.text CONTAINS 'we should create a merge' RETURN c.id"
    assert check_read_only(query) is None


def test_a_write_hidden_behind_a_comment_is_still_caught():
    query = "MATCH (c:Chunk) RETURN c.id\n// harmless\nMERGE (x:Evil)"
    assert check_read_only(query) is not None


def test_a_second_statement_is_rejected():
    query = "MATCH (c:Chunk) RETURN c.id; MATCH (d) DELETE d"
    reason = check_read_only(query)
    assert reason is not None
    assert "statement" in reason


# ------------------------------------------------------------------------ prompt shape

def test_the_prompt_carries_the_schema_and_the_cookbook():
    shots = load_fewshots()
    system, _user = build_prompt("who bridges two sources?", shots)
    assert "(:Walk {prompt" in system
    assert "PATHWAY" in system
    assert "NEXT_IN_CHAIN" in system
    assert "STRING" in system
    assert shots[0]["question"] in system
    i_rel = system.index('"relationships"')
    i_reason = system.index('"reasoning"')
    i_query = system.index('"query"')
    assert i_rel < i_reason < i_query


# --------------------------------------------------------------------------- the loop

def test_ask_returns_rows_and_the_query_it_ran():
    call = _fake_call({"relationships": ["Chunk", "PATHWAY"], "reasoning": "ok",
                        "query": "MATCH (a:Chunk)-[p:PATHWAY]->(b) RETURN a.id, p.dwpc"})
    post = _fake_post((["a.id", "p.dwpc"], [["3439", 0.138]]))
    params = {"prompt": "test prompt"}
    res = ask("which chunks bridge two sources?", params=params, call=call, post=post)
    assert res["ok"] is True
    assert res["n_attempts"] == 1
    assert res["rows"] == [{"a.id": "3439", "p.dwpc": 0.138}]
    assert res["query"] == "MATCH (a:Chunk)-[p:PATHWAY]->(b) RETURN a.id, p.dwpc"
    assert post.statements[0][0]["parameters"] == params


def test_a_neo4j_error_is_fed_back_and_the_retry_succeeds():
    query = "MATCH (x)-[]->(y) RETURN x"
    call = _fake_call(
        {"relationships": ["Chunk"], "reasoning": "r1", "query": query},
        {"relationships": ["Chunk"], "reasoning": "r2", "query": "MATCH (c:Chunk) RETURN c.id"},
    )
    err = RuntimeError("neo4j tx error Neo.ClientError.Statement.SyntaxError: "
                        "Variable `x` not defined")
    post = _fake_post(err, (["c.id"], [["1"]]))
    res = ask("q?", call=call, post=post)
    assert res["ok"] is True
    assert res["n_attempts"] == 2
    assert len(res["attempts"]) == 1
    assert res["attempts"][0]["stage"] == "execute"
    second_user = call.prompts[1][1]
    assert query in second_user
    assert "Variable `x` not defined" in second_user


def test_a_generated_write_is_never_posted_and_is_retried():
    call = _fake_call(
        {"relationships": ["Chunk"], "reasoning": "bad",
         "query": 'MERGE (x:Chunk {id:"1"}) RETURN x'},
        {"relationships": ["Chunk"], "reasoning": "good", "query": "MATCH (c:Chunk) RETURN c.id"},
    )
    post = _fake_post((["c.id"], [["1"]]))
    res = ask("q?", call=call, post=post)
    assert res["ok"] is True
    assert len(post.statements) == 1                        # only the good attempt posted
    assert res["attempts"][0]["stage"] == "reject"
    second_user = call.prompts[1][1]
    assert "read-only" in second_user


def test_three_failures_return_an_honest_structured_failure():
    call = _fake_call(
        {"relationships": [], "reasoning": "r1", "query": "MATCH (c:Chunk) RETURN c.id"},
        {"relationships": [], "reasoning": "r2", "query": "MATCH (c:Chunk) RETURN c.id"},
        {"relationships": [], "reasoning": "r3", "query": "MATCH (c:Chunk) RETURN c.id"},
    )
    post = _fake_post(RuntimeError("boom1"), RuntimeError("boom2"), RuntimeError("boom3"))
    res = ask("q?", call=call, post=post)
    assert res["ok"] is False
    assert res["n_attempts"] == 3
    assert len(res["attempts"]) == 3
    assert "boom1" in res["error"] and "boom2" in res["error"] and "boom3" in res["error"]
    assert res["rows"] == []


def test_unparseable_reply_counts_as_an_attempt_not_a_crash():
    call = _fake_call("not json, just prose", "still prose", "more prose")
    post = _fake_post()
    res = ask("q?", call=call, post=post)
    assert res["ok"] is False
    assert all(a["stage"] == "generate" for a in res["attempts"])
    assert len(post.statements) == 0


# ---------------------------------------------------------------------------- live test

@pytest.fixture(scope="module")
def neo4j_up():
    try:
        urllib.request.urlopen(export_neo4j.NEO4J_HTTP, timeout=2).read()
    except (urllib.error.URLError, OSError) as e:            # pragma: no cover
        pytest.skip(f"no neo4j at {export_neo4j.NEO4J_HTTP}: {e}")


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no API key")
def test_live_asks_a_cookbook_question_against_the_mirror(neo4j_up):
    results = export_neo4j._tx(
        [{"statement": "MATCH (w:Walk) RETURN w.prompt LIMIT 1", "parameters": {}}])
    data = results[0]["data"]
    if not data:
        pytest.skip("no Walk written")
    prompt = data[0]["row"][0]

    shots = load_fewshots()
    question = shots[5]["question"]                          # cheapest question in the bank
    res = ask(question, params={"prompt": prompt})

    assert res["ok"]
    assert check_read_only(res["query"]) is None
    assert isinstance(res["rows"], list)
    assert res["n_attempts"] <= MAX_ATTEMPTS
