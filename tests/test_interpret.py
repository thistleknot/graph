"""Pins interpret.py: deterministic rendering (I3), JSON parsing (I6), the
citation contract on both channels (I1), transport failure (I4), backend
order (I5), the rerank no-op (I7), and -- only when OPENROUTER_API_KEY is set
-- one live call whose verdicts must cover the shown chunks.

Run:  pytest tests/test_interpret.py -v      (needs docker compose up -d)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

import graph_tools as gt
import interpret
import sampler

Q = "a colonial power leaves and the country falls apart"


@pytest.fixture(scope="module")
def live():
    try:
        conn = gt.connect()
        run = gt.get_run(conn, "brown-50-dual")
    except (psycopg.OperationalError, LookupError) as e:   # pragma: no cover
        pytest.skip(f"no live run: {e}")
    yield conn, run
    conn.close()


@pytest.fixture(scope="module")
def walk(live):
    conn, run = live
    b, _ = sampler.ef_evidence(conn, run, Q)
    cids = [t["cid"] for t in gt.communities_touched(conn, run, b.sampled)]
    return (b, gt.query_terms(conn, run, cids, Q, k=3),
            gt.community_terms(conn, run, cids, k=3))


# --------------------------------------------------------------- I3 render


def test_render_is_deterministic_and_complete(live, walk):
    conn, run = live
    b, terms, concept = walk
    a = interpret.render_bundle(conn, run, b, terms, concept, max_chunks_per_community=1000)
    assert a == interpret.render_bundle(conn, run, b, terms, concept, max_chunks_per_community=1000)
    assert a.startswith(f"PROMPT: {Q}")
    for o in b.sampled:
        assert f"[id={o}] (" in a
    assert "as the prompt sees it:" in a and "local medoid #" in a


def test_render_respects_a_reranked_subset(live, walk):
    conn, run = live
    b, terms, concept = walk
    sub = b.sampled[:5]
    text = interpret.render_bundle(conn, run, b, terms, concept, ords=sub)
    for o in sub:
        assert f"[id={o}] (" in text
    for o in b.sampled[5:]:
        assert f"[id={o}] (" not in text
    assert f"VALID IDS ({len(sub)} chunks" in text


# ---------------------------------------------------------------- I6 parse


def test_parse_reply_accepts_bare_fenced_and_thinking_wrapped_json():
    body = {"verdicts": [{"id": 5, "verdict": "ENTAILS", "why": "x"},
                         {"ord": 6, "verdict": "neutral"},
                         {"ord": "bad", "verdict": "entails"},
                         {"ord": 7, "verdict": "maybe"}],
            "answer": "Because #5."}
    raw = json.dumps(body)
    for text in (raw, "```json\n" + raw + "\n```", "<think>hmm</think>\n" + raw,
                 "Here you go: " + raw + " done."):
        p = interpret.parse_reply(text)
        assert p is not None, text[:30]
        assert [v["ord"] for v in p["verdicts"]] == [5, 6], "bad rows must be dropped"
        assert p["verdicts"][0]["verdict"] == "entails"     # lower-cased
        assert p["answer"] == "Because #5."


def test_parse_reply_rejects_non_json_and_wrong_shape():
    assert interpret.parse_reply("I think the answer is #5.") is None
    assert interpret.parse_reply('{"answer": "no verdicts key"}') is None
    assert interpret.parse_reply("") is None


# ------------------------------------------------------------- I1 checks


def test_check_separates_entailed_foreign_and_self_contradiction():
    shown = [1, 2, 3, 4]
    parsed = {"verdicts": [{"ord": 1, "verdict": "entails", "why": ""},
                           {"ord": 2, "verdict": "neutral", "why": ""},
                           {"ord": 3, "verdict": "contradicts", "why": ""},
                           {"ord": 99, "verdict": "entails", "why": ""}],
              "answer": "Claim #1. Claim #2. Claim #42."}
    r = interpret.check(parsed, shown)
    assert r["entailed"] == [1]
    assert r["contradicts"] == [3]
    assert r["cited"] == [1, 2]
    assert r["foreign"] == [99, 42]
    assert r["self_contradicting"] == [2], "cited but not entailed must be flagged"
    assert r["coverage"] == 0.75                           # 3 of 4 shown judged


def test_check_empty_reply_is_zero_coverage():
    r = interpret.check({"verdicts": [], "answer": ""}, [1, 2])
    assert r["coverage"] == 0.0 and r["entailed"] == [] and r["foreign"] == []


# --------------------------------------------------------- I4 / I5 backends


def test_no_backend_returns_ok_false_not_raise(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:1")
    res = interpret.answer(conn, run, b, terms, concept, timeout=2.0, use_rerank=False)
    assert res["ok"] is False and res["backend"] is None
    assert "OPENROUTER_API_KEY unset" in res["error"]
    assert "_via_ollama" in res["error"]
    assert res["evidence"].startswith("PROMPT:")
    assert res["rerank_note"] == "rerank: skipped"


def test_openrouter_is_tried_before_ollama(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    calls = []
    monkeypatch.setattr(interpret, "_via_openrouter",
                        lambda s, u, t: (calls.append("or"), (json.dumps(
                            {"verdicts": [{"ord": b.sampled[0], "verdict": "entails", "why": "t"}],
                             "answer": f"see #{b.sampled[0]}"}), "openrouter:fake"))[1])
    monkeypatch.setattr(interpret, "_via_ollama",
                        lambda s, u, t: (calls.append("ol"), ("", "ollama:fake"))[1])
    res = interpret.answer(conn, run, b, terms, concept, use_rerank=False)
    assert calls == ["or"], "fell through to ollama although openrouter answered"
    assert res["ok"] and res["backend"] == "openrouter:fake"
    assert res["entailed"] == [b.sampled[0]] and res["self_contradicting"] == []
    assert res["fallback_reason"] is None


def test_zero_coverage_reply_is_not_an_answer_and_falls_through(live, walk, monkeypatch):
    """I6: a reply whose every ord is foreign judged nothing. Measured on the
    4b model: verdicts for ords 0..12, none in the bundle, ok=True before."""
    conn, run = live
    b, terms, concept = walk
    monkeypatch.setattr(interpret, "_via_openrouter", lambda s, u, t: (json.dumps(
        {"verdicts": [{"ord": i, "verdict": "neutral", "why": ""} for i in range(13)],
         "answer": ""}), "openrouter:fake"))
    monkeypatch.setattr(interpret, "_via_ollama", lambda s, u, t: (json.dumps(
        {"verdicts": [{"ord": b.sampled[0], "verdict": "entails", "why": "y"}],
         "answer": f"#{b.sampled[0]}"}), "ollama:fake"))
    res = interpret.answer(conn, run, b, terms, concept, use_rerank=False)
    assert res["ok"] and res["backend"] == "ollama:fake"
    assert "judged 0 of" in res["fallback_reason"] and "13 foreign" in res["fallback_reason"]


def test_unparseable_first_backend_falls_through(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    monkeypatch.setattr(interpret, "_via_openrouter", lambda s, u, t: ("prose, no json", "openrouter:fake"))
    monkeypatch.setattr(interpret, "_via_ollama", lambda s, u, t: (
        json.dumps({"verdicts": [{"ord": b.sampled[1], "verdict": "neutral", "why": ""}],
                    "answer": ""}), "ollama:fake"))
    res = interpret.answer(conn, run, b, terms, concept, use_rerank=False)
    assert res["ok"] and res["backend"] == "ollama:fake"
    assert "not the JSON object" in res["fallback_reason"]


# ------------------------------------------------------------------ I7


def test_rerank_is_a_noop_without_a_model(live, walk, monkeypatch):
    conn, run = live
    b, _, _ = walk
    monkeypatch.setattr(interpret, "RERANK_MODEL", None)
    ords, note = interpret.rerank(conn, run, Q, list(b.sampled))
    assert ords == list(b.sampled)
    assert note.startswith("rerank: off")


# ---------------------------------------------------------------- live


@pytest.mark.skipif(not __import__("os").environ.get("OPENROUTER_API_KEY"),
                    reason="OPENROUTER_API_KEY not set")
def test_live_openrouter_judges_the_shown_chunks(live, walk):
    conn, run = live
    b, terms, concept = walk
    res = interpret.answer(conn, run, b, terms, concept)
    assert res["ok"], res["error"]
    assert res["backend"].startswith("openrouter:")
    assert res["foreign"] == [], f"model named ords outside the bundle: {res['foreign']}"
    assert res["coverage"] >= 0.5, f"judged only {res['coverage']:.0%} of shown chunks"
    assert res["verdicts"], "no verdicts"
