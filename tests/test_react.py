"""tests/test_react.py -- offline-first tests for the react.py ReAct loop.

Spec: .spec/specs/graph-explorer/design.md 6.23 A1-A10
Task: playbook.md T61
"""

from __future__ import annotations

import dataclasses

import pytest

import react
import sampler


# --------------------------------------------------------------------------- fakes

def mk_bundle(ords, scores=None, query="q"):
    scores = scores if scores is not None else {o: 1.0 for o in ords}
    return sampler.Bundle(
        query=query, run_id="r", anchors=[], candidates=len(ords),
        sampled=list(ords), communities=[], enumerated=False,
        scores=dict(scores), origin={o: "walk" for o in ords}, params={})


class FakeEv:
    def __init__(self, digest="D", cid_of=None, cids=None):
        self.terms = {}
        self.concept = {}
        self.pathways = {}
        self.digest = digest
        self.cid_of = cid_of or {}
        self.cids = cids or []


def fake_walk(script):
    calls = []

    def _walk(query, params):
        calls.append((query, dict(params)))
        return script[len(calls) - 1]

    _walk.calls = calls
    return _walk


def fake_judge(verdict_map, calls):
    def _judge(bundle, ev):
        calls.append(list(bundle.sampled))
        return {"ok": True,
                "verdicts": [{"ord": o, "verdict": verdict_map.get(o, "neutral"), "why": ""}
                             for o in bundle.sampled],
                "entailed": [o for o in bundle.sampled if verdict_map.get(o) == "entails"],
                "contradicts": [o for o in bundle.sampled if verdict_map.get(o) == "contradicts"]}
    return _judge


def always_insufficient(query, digest, history, missing_hint=None):
    return {"sufficient": False, "missing": [], "action": "WIDEN",
            "query_add": "", "why": "", "source": "stub"}


def always_sufficient(query, digest, history, missing_hint=None):
    return {"sufficient": True, "missing": [], "action": "WIDEN",
            "query_add": "", "why": "", "source": "stub"}


# --------------------------------------------------------------------------- stats()

def test_stats_mean_and_population_sdev():
    bnd = mk_bundle([1, 2, 3, 4], scores={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0})
    st = react.stats(bnd, None)
    assert st["mean_score"] == 2.5
    assert st["sdev_score"] == 1.118

    empty = mk_bundle([], scores={})
    st0 = react.stats(empty, None)
    assert st0["mean_score"] == 0.0
    assert st0["sdev_score"] == 0.0
    assert st0["n_chunks"] == 0


def test_stats_counts_and_neutrals_fill():
    bnd = mk_bundle([1, 2, 3, 4, 5])
    rr = {"verdicts": [{"ord": 1, "verdict": "entails"},
                        {"ord": 2, "verdict": "contradicts"},
                        {"ord": 3, "verdict": "neutral"}]}
    st = react.stats(bnd, rr)
    assert st["entails"] == 1
    assert st["contradicts"] == 1
    assert st["neutrals"] == 3


def test_stats_per_cid_mean():
    bnd = mk_bundle([1, 2, 3, 4], scores={1: 1.0, 2: 3.0, 3: 5.0, 4: 7.0})
    cid_of = {1: "a", 2: "a", 3: "b", 4: "b"}
    st = react.stats(bnd, None, cid_of)
    assert st["per_cid_mean"]["a"] == 2.0
    assert st["per_cid_mean"]["b"] == 6.0


# --------------------------------------------------------------------------- A9 derived metrics

def test_precision_and_contradiction_rate_arithmetic():
    bnd = mk_bundle([1, 2, 3, 4])
    rr = {"verdicts": [{"ord": 1, "verdict": "entails"},
                        {"ord": 2, "verdict": "entails"},
                        {"ord": 3, "verdict": "contradicts"},
                        {"ord": 4, "verdict": "neutral"}]}
    st = react.stats(bnd, rr)
    # judged = entails + contradicts = 3 (neutral excluded from the denominator)
    assert st["precision_proxy"] == round(2 / 3, 4)
    assert st["contradiction_rate"] == round(1 / 3, 4)

    # all-neutral / no verdicts at all -> judged == 0, no division by zero
    st_zero = react.stats(bnd, {"verdicts": [{"ord": o, "verdict": "neutral"} for o in bnd.sampled]})
    assert st_zero["precision_proxy"] == 0.0
    assert st_zero["contradiction_rate"] == 0.0


def test_gold_recall_normalization():
    texts = ["The band Nirvana, fronted by Kurt Cobain, defined 1990s grunge."]
    # underscore vs space, case, and a term genuinely absent from the text
    assert react.gold_recall(texts, ["kurt_cobain", "Nirvana", "radiohead"]) == round(2 / 3, 4)
    assert react.gold_recall(texts, []) == 0.0
    assert react.gold_recall([], ["nirvana"]) == 0.0
    assert react.gold_recall("KURT COBAIN", ["kurt_cobain"]) == 1.0


# --------------------------------------------------------------------------- run() loop

def test_zero_entail_forces_insufficient_without_llm():
    bnd = mk_bundle([1, 2, 3])
    walk = fake_walk([(bnd, FakeEv())] * 10)
    judge_calls = []
    judge = fake_judge({}, judge_calls)   # every verdict defaults to "neutral"

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_sufficient, max_iters=3)
    assert result["stop_reason"] == "budget"

    # max_iters=0: no proposer call happens at all
    calls = {"n": 0}
    def counting_propose(*a, **k):
        calls["n"] += 1
        return always_sufficient(*a, **k)
    walk0 = fake_walk([(bnd, FakeEv())])
    result0 = react.run(None, None, "q", walk_fn=walk0, judge_fn=fake_judge({}, []),
                         propose_fn=counting_propose, max_iters=0)
    assert result0["stop_reason"] == "budget"
    assert calls["n"] == 0


def test_budget_stops_at_three_iterations_past_base():
    bnd = mk_bundle([1, 2, 3])
    walk = fake_walk([(bnd, FakeEv())] * 10)
    judge = fake_judge({}, [])

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_insufficient, max_iters=3)
    assert len(result["iterations"]) == 4
    assert result["n_iters"] == 3
    assert result["stop_reason"] == "budget"
    assert len(walk.calls) == 4


def test_early_stop_on_sufficient():
    bnd = mk_bundle([1])
    walk = fake_walk([(bnd, FakeEv())] * 5)
    judge = fake_judge({1: "entails"}, [])

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_sufficient, max_iters=3)
    assert result["n_iters"] == 0
    assert result["stop_reason"] == "sufficient"
    assert len(walk.calls) == 1


def test_accumulation_unions_with_provenance():
    bnd0 = mk_bundle([1, 2, 3])
    bnd1 = mk_bundle([3, 4, 5])
    walk = fake_walk([(bnd0, FakeEv()), (bnd1, FakeEv())])
    judge = fake_judge({}, [])

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_insufficient, max_iters=1)
    assert result["ords"] == [1, 2, 3, 4, 5]
    assert result["found_at"] == {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}


def test_judge_never_re_judges_seen_chunks():
    bnd0 = mk_bundle([1, 2, 3])
    bnd1 = mk_bundle([3, 4, 5])
    walk = fake_walk([(bnd0, FakeEv()), (bnd1, FakeEv())])
    judge_calls = []
    # ord 3 gets a DIFFERENT verdict on the (never-made) second judge call to
    # prove the first verdict, "entails", wins forever.
    seen = {"n": 0}
    def judge(bundle, ev):
        seen["n"] += 1
        judge_calls.append(list(bundle.sampled))
        verdict_map = {3: "entails"} if seen["n"] == 1 else {3: "contradicts"}
        return {"ok": True,
                "verdicts": [{"ord": o, "verdict": verdict_map.get(o, "neutral"), "why": ""}
                             for o in bundle.sampled]}

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_insufficient, max_iters=1)
    assert judge_calls == [[1, 2, 3], [4, 5]]
    assert result["verdicts"][3]["verdict"] == "entails"


# --------------------------------------------------------------------------- apply_action()

def test_apply_action_clamps_and_is_pure():
    before = {"ef": 400, "ring_top": 7, "ring_per": 8, "k_anchor": 3, "m": 3,
              "bridge_pairs": 3, "expand_aliases": False}
    params = dict(before)
    new_params, new_query = react.apply_action(params, "WIDEN", query="q")
    assert params == before                    # input untouched
    assert new_params["ef"] == 512              # 400*2=800 clamped to 512
    assert new_params["ring_top"] == 8           # 7+2=9 clamped to 8

    p2, q2 = react.apply_action(dict(before), "REANCHOR", query="q", query_add="grunge nirvana")
    assert q2 == "q grunge nirvana"
    p3, q3 = react.apply_action(p2, "REANCHOR", query=q2, query_add="grunge nirvana")
    assert q3 == q2                              # not appended twice

    p4, _ = react.apply_action(dict(before), "DEEPEN", query="q")
    assert p4["m"] == before["m"] + 1
    assert p4["bridge_pairs"] == before["bridge_pairs"] + 2

    p5, q5 = react.apply_action(dict(before), "NONSENSE", query="q")
    assert p5 == before
    assert q5 == "q"


# --------------------------------------------------------------------------- propose()

def _mk_record(i):
    return react.IterationRecord(
        i=i, params=dict(react.BASE_PARAMS), query="q", n_chunks=0,
        mean_score=0.0, sdev_score=0.0, entails=0, contradicts=0, neutrals=0,
        per_cid_mean={}, new_chunks=0, precision_proxy=0.0, contradiction_rate=0.0)


def test_propose_fallback_ladder_on_junk_and_transport_failure():
    def raises(*a, **k):
        raise RuntimeError("boom")

    def not_json(*a, **k):
        return "not json", "backend"

    def wrong_action(*a, **k):
        return '{"sufficient": false, "action": "TELEPORT"}', "backend"

    history = [_mk_record(0)]        # len 1 -> LADDER[0] == WIDEN
    for stub in (raises, not_json, wrong_action):
        result = react.propose("q", "digest", history, call=stub)
        assert result["source"] == "fallback"
        assert result["action"] == "WIDEN"

    history2 = [_mk_record(0), _mk_record(1)]     # len 2 -> LADDER[1] == REANCHOR
    result2 = react.propose("q", "digest", history2, call=raises)
    assert result2["action"] == "REANCHOR"

    history3 = [_mk_record(0), _mk_record(1), _mk_record(2)]  # len 3 -> LADDER[2] == DEEPEN
    result3 = react.propose("q", "digest", history3, call=raises)
    assert result3["action"] == "DEEPEN"


# --------------------------------------------------------------------------- transcript

def test_transcript_is_complete_and_renderable():
    bnd = mk_bundle([1, 2, 3])
    walk = fake_walk([(bnd, FakeEv())] * 4)
    judge = fake_judge({}, [])

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_insufficient, max_iters=2)
    iters = result["iterations"]
    assert len(iters) == 3
    assert iters[0].action is None
    for r in iters[:-1]:
        assert r.stop_reason is None
    assert iters[-1].stop_reason is not None

    for r in iters:
        assert r.n_chunks == 3
        assert isinstance(r.precision_proxy, float)
        assert isinstance(r.contradiction_rate, float)

    iters[0].params["ef"] = 999999
    assert iters[1].params["ef"] != 999999   # no aliasing across records

    table = result["history_table"]
    assert table.startswith("it action")
    assert table.count("\n") >= len(iters)   # header + one row per iteration


def test_gold_terms_populate_gold_recall_evidence():
    bnd = mk_bundle([1])
    walk = fake_walk([(bnd, FakeEv(digest="nirvana kurt_cobain grunge"))] * 2)
    judge = fake_judge({1: "entails"}, [])

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_sufficient, max_iters=1,
                        gold_terms=["kurt_cobain", "radiohead"])
    rec = result["iterations"][0]
    assert rec.gold_recall_evidence == round(1 / 2, 4)


# --------------------------------------------------------------------------- live smoke

@pytest.mark.live_net
@pytest.mark.live_db
def test_react_live_smoke():
    import os
    from conftest import require_gt_conn, require_run

    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    conn = require_gt_conn()
    run_ = require_run(conn, "mixed-full-dual")
    result = react.run(conn, run_, "most famous musician of the 1990s")
    assert len(result["iterations"]) >= 1
    assert result["stop_reason"] in ("sufficient", "budget", "no-op action")
