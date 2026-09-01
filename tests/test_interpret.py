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
        run = gt.get_run(conn, "brown-500-dual")
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
    assert "ollama" in res["error"]            # ran and timed out, or skipped as oversized
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
    monkeypatch.setattr(interpret, "OLLAMA_MAX_CHARS", 10**9)   # exercise the fallback path
    b, terms, concept = walk
    monkeypatch.setattr(interpret, "_via_openrouter", lambda s, u, t: (json.dumps(
        {"verdicts": [{"ord": 10**6 + i, "verdict": "neutral", "why": ""} for i in range(13)],   # never real ords
         "answer": ""}), "openrouter:fake"))
    monkeypatch.setattr(interpret, "_via_ollama", lambda s, u, t: (json.dumps(
        {"verdicts": [{"ord": b.sampled[0], "verdict": "entails", "why": "y"}],
         "answer": f"#{b.sampled[0]}"}), "ollama:fake"))
    res = interpret.answer(conn, run, b, terms, concept, use_rerank=False)
    assert res["ok"] and res["backend"] == "ollama:fake"
    assert "judged 0 of" in res["fallback_reason"] and "13 foreign" in res["fallback_reason"]


def test_unparseable_first_backend_falls_through(live, walk, monkeypatch):
    conn, run = live
    monkeypatch.setattr(interpret, "OLLAMA_MAX_CHARS", 10**9)   # exercise the fallback path
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


# ----------------------------------------------- I5 transport: retry shape


def _fake_urlopen(script):
    """script: list of callables/exceptions consumed per attempt."""
    import io, json
    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(req, timeout=None):
        calls.append(timeout)
        step = script[len(calls) - 1]
        if isinstance(step, Exception):
            raise step
        return _Resp(json.dumps(step).encode("utf-8"))
    return urlopen, calls


def test_openrouter_retries_transient_failures_then_succeeds(monkeypatch):
    import urllib.request
    good = {"choices": [{"finish_reason": "stop",
                         "message": {"content": '{"verdicts": [], "answer": "ok"}'}}]}
    urlopen, calls = _fake_urlopen([TimeoutError("handshake"), good])
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    text, name = interpret._via_openrouter("s", "u", 30.0)
    assert text == '{"verdicts": [], "answer": "ok"}' and name.startswith("openrouter:")
    assert len(calls) == 2 and all(t == 30.0 for t in calls)


def test_openrouter_retries_empty_content_and_gives_up_after_attempts(monkeypatch):
    import urllib.request
    empty = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
    urlopen, calls = _fake_urlopen([empty, empty])
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(RuntimeError, match="exhausted 2 attempts"):
        interpret._via_openrouter("s", "u", 30.0)
    assert len(calls) == 2


def test_openrouter_auth_errors_do_not_retry(monkeypatch):
    import urllib.error, urllib.request
    err = urllib.error.HTTPError("u", 401, "unauthorized", {}, None)
    urlopen, calls = _fake_urlopen([err, err])
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(urllib.error.HTTPError):
        interpret._via_openrouter("s", "u", 30.0)
    assert len(calls) == 1, "401 must fail fast, not burn retries"


def test_openrouter_truncated_json_is_retried_as_a_failure(monkeypatch):
    import urllib.request
    cut = {"choices": [{"finish_reason": "length", "message": {"content": '{"verdicts": [{"id": 1'}}]}
    good = {"choices": [{"finish_reason": "stop", "message": {"content": '{"verdicts": [], "answer": ""}'}}]}
    urlopen, calls = _fake_urlopen([cut, good])
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    text, _ = interpret._via_openrouter("s", "u", 30.0)
    assert text.endswith("}") and len(calls) == 2


# -------------------------------------------------- I8 excerpt by the prompt

NL2 = chr(10) * 2


def _body(*paras):
    return NL2.join(paras)


def test_excerpt_prefers_the_paragraph_that_carries_the_prompt_terms():
    body = _body("Opening line about the weather .",
                 "The electoral procedure in Morocco set registration and voting rules .",
                 "A closing paragraph about dinner .")
    out = interpret.excerpt(body, "how were Morocco's first elections organized", 200)
    assert out.startswith("The electoral procedure")
    assert "weather" not in out and "dinner" not in out


def test_excerpt_keeps_source_order_and_stays_within_budget():
    body = _body("Elections were announced in the spring .",
                 "Filler about crops .",
                 "Voting districts were fixed by the ministry before the elections .",
                 "More filler about crops .")
    out = interpret.excerpt(body, "elections voting districts", 200)
    assert out.index("Elections were announced") < out.index("Voting districts")
    assert "crops" not in out
    assert len(out) <= 200 + 2


def test_excerpt_falls_back_to_the_opening_when_nothing_hits():
    body = _body("Opening line .", "Second paragraph .", "Third paragraph .")
    out = interpret.excerpt(body, "quokka wombat", 30)
    assert out.startswith("Opening line")


def test_excerpt_handles_empty_and_stopword_only_queries():
    body = _body("Only paragraph here .")
    assert interpret.excerpt("", "elections", 100) == ""
    assert interpret.excerpt(body, "the of and", 100).startswith("Only paragraph")


def test_render_uses_prompt_excerpts_not_openings(live, walk):
    """I8 end to end. The contract is per document: wherever the body carries
    a prompt term ANYWHERE, the rendered excerpt must carry one too. Documents
    reached through edges may carry none at all; their opening fallback is
    correct, not a miss -- so they are excluded, not counted against."""
    conn, run = live
    b, terms, concept = walk
    text = interpret.render_bundle(conn, run, b, terms, concept)
    qt = set(gt.tokenize(Q))
    shown = {}
    for l in text.splitlines():
        s_ = l.strip()
        if s_.startswith("[id="):
            shown.setdefault(int(s_[4:s_.index("]")]), s_)   # first rendering per id
    assert shown
    eligible, hit = 0, 0
    for o in b.sampled:
        body_terms = set(gt.tokenize(gt.node(conn, run, o)["body"]))
        if body_terms & qt and o in shown:
            eligible += 1
            if set(gt.tokenize(shown[o])) & qt:
                hit += 1
    assert eligible >= 5, "fixture: too few documents carry a prompt term"
    assert hit == eligible, f"{eligible - hit} of {eligible} term-bearing documents rendered an opening instead"


# ------------------------------------ I8 dense signal, budgets; I6 two-stage


def test_excerpt_dense_signal_promotes_a_lexically_weak_paragraph():
    """The intro carries the prompt words; the mechanics paragraph carries the
    answer with no lexical overlap. A dense signal that points at the mechanics
    paragraph must lift it into the excerpt."""
    import numpy as np
    intro = "Morocco elections elections Morocco first elections overview ."
    mech = "Registration of voters and fixing of districts were done by the ministry ."
    filler = "Crops and weather this season ."
    body = _body(intro, mech, filler)

    def embed(texts):                       # query and `mech` share a direction
        out = []
        for t in texts:
            if t.startswith("Registration") or "organized" in t:
                out.append(np.array([1.0, 0.0]))
            else:
                out.append(np.array([0.0, 1.0]))
        return np.stack(out)
    lexical = interpret.excerpt(body, "how were Morocco's elections organized", 200)
    dense = interpret.excerpt(body, "how were Morocco's elections organized", 200, embed=embed)
    assert "Registration" not in lexical
    assert "Registration" in dense


def test_excerpt_fits_three_capped_paragraphs_in_the_budget():
    paras = [f"elections paragraph number {i} " + "word " * 120 + "." for i in range(3)]
    out = interpret.excerpt(_body(*paras), "elections", 1500)
    assert all(f"elections paragraph number {i}" in out for i in range(3)), out[:200]
    assert len(out) <= 1500 + 3


def test_render_gives_anchors_a_larger_budget(live, walk):
    conn, run = live
    b, terms, concept = walk
    text = interpret.render_bundle(conn, run, b, terms, concept)
    by = {}
    for l in text.splitlines():
        s = l.strip()
        if s.startswith("[id="):
            o = int(s[4:s.index("]")]); by[o] = len(s)
    anchors = [o for o in b.anchors if o in by]
    others = [o for o in by if o not in set(b.anchors)]
    assert anchors and others
    assert max(by[o] for o in anchors) > max(by[o] for o in others), "anchor excerpt not larger"


def test_followup_answer_runs_when_entailed_but_answer_empty(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    target = b.sampled[0]
    calls = []

    def fake_or(system, user, timeout):
        calls.append("followup" if "ENTAILED EXCERPTS" in user else "judge")
        if "ENTAILED EXCERPTS" in user:
            return json.dumps({"answer": f"It was so #{target}. Also #999999."}), "openrouter:fake"
        return json.dumps({"verdicts": [{"id": target, "verdict": "entails", "why": "y"}],
                           "answer": ""}), "openrouter:fake"
    monkeypatch.setattr(interpret, "_via_openrouter", fake_or)
    monkeypatch.setattr(interpret, "_via_ollama", lambda s, u, t: ("", "ollama:fake"))
    res = interpret.answer(conn, run, b, terms, concept, use_rerank=False)
    assert calls == ["judge", "followup"]
    assert res["answer_stage"] == "followup"
    assert res["answer"].startswith("It was so")
    assert res["cited"] == [target]
    assert res["self_contradicting"] == [999999], "citation outside the entailed set must be flagged"


def test_no_followup_when_answer_already_present(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    target = b.sampled[0]
    calls = []
    monkeypatch.setattr(interpret, "_via_openrouter", lambda s, u, t: (calls.append(1), (json.dumps(
        {"verdicts": [{"id": target, "verdict": "entails", "why": "y"}], "answer": f"Yes #{target}."}), "openrouter:fake"))[1])
    res = interpret.answer(conn, run, b, terms, concept, use_rerank=False)
    assert len(calls) == 1 and res["answer_stage"] == "single"


# ------------------------------------- 6.6 reason over community evidence


def test_briefs_are_deterministic_and_carry_both_medoids(live, walk):
    conn, run = live
    b, terms, concept = walk
    a = interpret.community_briefs(conn, run, b, terms, concept)
    c = interpret.community_briefs(conn, run, b, terms, concept)
    assert a == c and a
    assert [x["hits"] for x in a] == sorted([x["hits"] for x in a], reverse=True)
    for x in a:
        assert x["local"]["ord"] in set(b.sampled)
        assert x["global"]["ord"] == gt.community(conn, run, x["cid"])["medoid"]
        assert x["local"]["excerpt"] and x["global"]["excerpt"]
    text = interpret.render_briefs(b, a)
    assert text.startswith(f"PROMPT: {Q}") and "VALID IDS" in text
    assert "LOCAL medoid" in text and "GLOBAL medoid" in text


def _staged_backend(b_local, b_global, calls, *, foreign=False, contradict=False, bad_cite=False):
    """One fake backend that answers each stage by the system prompt it gets."""
    def be(system, user, timeout, retries=None, images=None, max_tokens=None):
        if system is interpret.HYP_SYSTEM or "Propose up to three" in system:
            calls.append("hypothesis")
            return json.dumps({"hypotheses": ["H0", "H1"], "chosen": 1, "why": "because"}), "openrouter:fake"
        if "premises the hypothesis needs" in system:
            calls.append("premises")
            ids0 = [b_local] + ([10**6] if foreign else [])
            return json.dumps({"premises": [{"text": "P0", "ids": ids0},
                                            {"text": "P1", "ids": [b_global]},
                                            {"text": "P2", "ids": []}]}), "openrouter:fake"
        if "For each PREMISE" in system:
            calls.append("evaluate")
            return json.dumps({"evaluations": [
                {"index": 0, "verdict": "supports", "why": "yes"},
                {"index": 1, "verdict": "contradicts" if contradict else "insufficient", "why": "no"},
                {"index": 2, "verdict": "supports", "why": "ignored: no ids"}]}), "openrouter:fake"
        calls.append("answer")
        cite = b_global if bad_cite else b_local
        return json.dumps({"answer": f"Therefore X #{cite}."}), "openrouter:fake"
    return be


def test_reason_runs_four_stages_and_answers_from_supported_only(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    briefs = interpret.community_briefs(conn, run, b, terms, concept)
    bl, bg = briefs[0]["local"]["ord"], briefs[0]["global"]["ord"]
    calls = []
    monkeypatch.setattr(interpret, "_via_openrouter", _staged_backend(bl, bg, calls))
    monkeypatch.setattr(interpret, "_via_ollama", lambda s, u, t: (_ for _ in ()).throw(RuntimeError("down")))
    res = interpret.reason(conn, run, b, terms, concept, one_shot=False, structure="none")
    assert res["ok"], res["error"]
    assert calls == ["hypothesis", "premises", "evaluate", "answer"]
    assert res["hypothesis"] == "H1" and res["hypotheses"] == ["H0", "H1"]
    v = {p["text"]: p["verdict"] for p in res["premises"]}
    assert v == {"P0": "supports", "P1": "insufficient", "P2": "unsupported"}, v
    assert res["supported_ids"] == [bl]
    assert res["cited"] == [bl] and res["self_contradicting"] == []
    assert set(res["stages"]) == {"hypothesis", "premises", "evaluate", "answer"}   # I11


def test_reason_discards_foreign_ids_and_flags_bad_citations(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    briefs = interpret.community_briefs(conn, run, b, terms, concept)
    bl, bg = briefs[0]["local"]["ord"], briefs[0]["global"]["ord"]
    if bl == bg:
        pytest.skip("local and global medoid coincide on this walk; need two ids")
    calls = []
    monkeypatch.setattr(interpret, "_via_openrouter",
                        _staged_backend(bl, bg, calls, foreign=True, bad_cite=True))
    res = interpret.reason(conn, run, b, terms, concept, one_shot=False, structure="none")
    assert res["ok"]
    assert res["foreign"] == [10**6]                                   # I9
    assert res["premises"][0]["ids"] == [bl], "foreign id must be dropped from the premise"
    assert res["self_contradicting"] == [bg] and res["cited"] == []    # I10


def test_reason_skips_answer_when_nothing_supports(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    briefs = interpret.community_briefs(conn, run, b, terms, concept)
    bl, bg = briefs[0]["local"]["ord"], briefs[0]["global"]["ord"]
    calls = []
    def be(system, user, timeout, retries=None, images=None, max_tokens=None):
        base = _staged_backend(bl, bg, calls)
        if "For each PREMISE" in system:
            calls.append("evaluate")
            return json.dumps({"evaluations": [{"index": 0, "verdict": "insufficient", "why": ""},
                                               {"index": 1, "verdict": "contradicts", "why": ""}]}), "openrouter:fake"
        return base(system, user, timeout)
    monkeypatch.setattr(interpret, "_via_openrouter", be)
    res = interpret.reason(conn, run, b, terms, concept, one_shot=False, structure="none")
    assert res["ok"] and "answer" not in calls
    assert res["answer"] == "" and res["supported_ids"] == []


def test_reason_transport_failure_is_reported_not_raised(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    monkeypatch.setattr(interpret, "_via_openrouter", lambda s, u, t: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(interpret, "_via_ollama", lambda s, u, t: (_ for _ in ()).throw(RuntimeError("y")))
    res = interpret.reason(conn, run, b, terms, concept, one_shot=False, structure="none")
    assert res["ok"] is False and "RuntimeError" in res["error"]
    assert res["briefs"] and res["briefs_text"]                       # briefs still built


def test_briefs_carry_walk_ranked_evidence_besides_the_medoids(live, walk):
    conn, run = live
    b, terms, concept = walk
    briefs = interpret.community_briefs(conn, run, b, terms, concept)
    sampled = set(b.sampled)
    for x in briefs:
        ev = x["evidence"]
        assert len(ev) <= interpret.EVIDENCE_PER_BRIEF
        assert all(e["ord"] in sampled for e in ev)
        assert all(e["ord"] not in (x["local"]["ord"], x["global"]["ord"]) for e in ev)
        scores = [b.scores.get(e["ord"], 0.0) for e in ev]
        assert scores == sorted(scores, reverse=True)
    text = interpret.render_briefs(b, briefs)
    if any(x["evidence"] for x in briefs):
        assert "retrieved evidence (walk-ranked)" in text



# ------------------------------------------ 6.6 one-shot reason (default)


def _one_shot_backend(bl, bg, calls, *, foreign=False, bad_cite=False, none_support=False):
    def be(system, user, timeout, retries=None, images=None, max_tokens=None):
        calls.append("one_shot" if "Do all of the following in ONE reply" in system else "other")
        prem = [{"text": "P0", "ids": [bl] + ([10**6] if foreign else [])},
                {"text": "P1", "ids": [bg]}, {"text": "P2", "ids": []}]
        ev = [{"index": 0, "verdict": "insufficient" if none_support else "supports", "why": "y"},
              {"index": 1, "verdict": "insufficient", "why": "n"},
              {"index": 2, "verdict": "supports", "why": "ignored"}]
        return json.dumps({"hypotheses": ["H0", "H1"], "chosen": 1, "why": "w",
                           "premises": prem, "evaluations": ev,
                           "answer": f"So X #{bg if bad_cite else bl}."}), "openrouter:fake"
    return be


def test_one_shot_is_the_default_and_makes_exactly_one_call(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    briefs = interpret.community_briefs(conn, run, b, terms, concept)
    bl, bg = briefs[0]["local"]["ord"], briefs[0]["global"]["ord"]
    calls = []
    monkeypatch.setattr(interpret, "_via_openrouter", _one_shot_backend(bl, bg, calls))
    res = interpret.reason(conn, run, b, terms, concept, structure="none")
    assert calls == ["one_shot"], calls
    assert res["ok"] and res["hypothesis"] == "H1" and list(res["stages"]) == ["one_shot"]
    v = {p["text"]: p["verdict"] for p in res["premises"]}
    assert v == {"P0": "supports", "P1": "insufficient", "P2": "unsupported"}
    assert res["supported_ids"] == [bl] and res["cited"] == [bl] and res["self_contradicting"] == []


def test_one_shot_applies_i9_and_i10(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    briefs = interpret.community_briefs(conn, run, b, terms, concept)
    bl, bg = briefs[0]["local"]["ord"], briefs[0]["global"]["ord"]
    if bl == bg:
        pytest.skip("need two distinct medoid ids")
    monkeypatch.setattr(interpret, "_via_openrouter", _one_shot_backend(bl, bg, [], foreign=True, bad_cite=True))
    res = interpret.reason(conn, run, b, terms, concept, structure="none")
    assert res["foreign"] == [10**6] and res["premises"][0]["ids"] == [bl]
    assert res["self_contradicting"] == [bg] and res["cited"] == []


def test_one_shot_empties_the_answer_when_nothing_supports(live, walk, monkeypatch):
    conn, run = live
    b, terms, concept = walk
    briefs = interpret.community_briefs(conn, run, b, terms, concept)
    bl, bg = briefs[0]["local"]["ord"], briefs[0]["global"]["ord"]
    monkeypatch.setattr(interpret, "_via_openrouter", _one_shot_backend(bl, bg, [], none_support=True))
    res = interpret.reason(conn, run, b, terms, concept, structure="none")
    assert res["ok"] and res["supported_ids"] == [] and res["answer"] == ""


# --------------------------------------------- I12/I13 structure + combined call

def test_i12_render_structure_is_deterministic_and_names_the_pairs(live, walk):
    conn, run = live
    b, _, _ = walk
    pw = gt.pathways(conn, run, b.sampled, b.sampled[:4])
    s1 = interpret.render_structure(conn, run, b, pw)
    s2 = interpret.render_structure(conn, run, b, pw)
    assert s1 == s2 and "dwpc" in s1 and "conductance" in s1
    for p in pw["pairs"][:8]:
        assert f"[id={p['a']}]" in s1 and f"[id={p['b']}]" in s1


def test_i12_render_walk_image_is_a_png(live, walk):
    conn, run = live
    b, _, _ = walk
    pw = gt.pathways(conn, run, b.sampled, b.sampled[:4])
    png = interpret.render_walk_image(conn, run, b, pw)
    assert isinstance(png, bytes) and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 20_000, "a two-panel figure should not be trivially small"


def test_i13_combined_call_is_one_call_with_both_channels(live, walk, monkeypatch):
    """judge=True: ONE transport call; the reply's reason sections AND verdicts
    are parsed, foreign verdict ids discarded, coverage computed."""
    import json as _json
    conn, run = live
    b, _, _ = walk
    calls = []

    def fake_call(system, user, timeout, images=None, max_tokens=None):
        calls.append({"system": system, "images": images, "max_tokens": max_tokens})
        lm = sorted(b.sampled)[0]
        reply = {"hypotheses": ["H"], "chosen": 0, "why": "w",
                 "premises": [{"text": "P", "ids": [lm, sorted(b.sampled)[-1]]}],
                 "evaluations": [{"index": 0, "verdict": "supports", "why": "ok"}],
                 "answer": f"A #{lm}.",
                 "verdicts": ([{"id": o, "verdict": "neutral", "why": "n"}
                               for o in b.sampled if o != lm]
                              + [{"id": lm, "verdict": "entails", "why": "y"},
                                 {"id": 10**6, "verdict": "entails", "why": "foreign"}])}
        return _json.dumps(reply), "test-backend"

    monkeypatch.setattr(interpret, "_call", fake_call)
    terms, concept = {}, {}
    rr = interpret.reason(conn, run, b, terms, concept, judge=True, structure="text")
    assert rr["ok"], rr["error"]
    assert len(calls) == 1, "reason+judge must be ONE serialized call"
    assert "verdicts" in calls[0]["system"] and calls[0]["max_tokens"] == interpret.JUDGE_MAX_TOKENS
    assert "== STRUCTURE" in rr["structure"] and rr["structure"] in rr["briefs_text"] + rr["structure"]
    assert rr["premises"][0]["verdict"] == "supports" and rr["answer"].startswith("A #")
    assert set(rr["shown"]) == set(b.sampled)
    assert rr["coverage"] == 1.0
    assert rr["entailed"] == [sorted(b.sampled)[0]]
    assert sorted(b.sampled)[-1] in rr["premises"][0]["ids"], (
        "with judge on, ANY shown chunk is citable by a premise -- not only brief ids")
    assert 10**6 in rr["foreign"], "foreign verdict id must be discarded and reported"


def test_i12_image_rejection_falls_back_to_numbers(live, walk, monkeypatch):
    import json as _json
    conn, run = live
    b, _, _ = walk
    calls = []

    def fake_call(system, user, timeout, images=None, max_tokens=None):
        calls.append(images)
        if images:
            raise RuntimeError("provider rejected image content")
        reply = {"hypotheses": ["H"], "chosen": 0, "why": "w",
                 "premises": [{"text": "P", "ids": []}],
                 "evaluations": [], "answer": ""}
        return _json.dumps(reply), "test-backend"

    monkeypatch.setattr(interpret, "_call", fake_call)
    rr = interpret.reason(conn, run, b, {}, {}, judge=False, structure="image")
    assert rr["ok"], rr["error"]
    assert calls[0] and calls[1] is None, "must retry text-only after image rejection"
    assert "numbers" in rr["structure_note"]
