"""tests/test_react.py -- offline-first tests for the react.py ReAct loop.

Spec: .spec/specs/graph-explorer/design.md 6.23 A1-A10
Task: playbook.md T61
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

import react
import sampler

_diag_agentic_spec = importlib.util.spec_from_file_location(
    "diag_agentic", Path(__file__).resolve().parent.parent / "tools" / "diag_agentic.py")
diag_agentic = importlib.util.module_from_spec(_diag_agentic_spec)
_diag_agentic_spec.loader.exec_module(diag_agentic)


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

    history = [_mk_record(0)]        # len 1 -> LADDER[0] == REANCHOR (A12 reorder)
    for stub in (raises, not_json, wrong_action):
        result = react.propose("q", "digest", history, call=stub)
        assert result["source"] == "fallback"
        assert result["action"] == "REANCHOR"

    history2 = [_mk_record(0), _mk_record(1)]     # len 2 -> LADDER[1] == DEEPEN
    result2 = react.propose("q", "digest", history2, call=raises)
    assert result2["action"] == "DEEPEN"

    history3 = [_mk_record(0), _mk_record(1), _mk_record(2)]  # len 3 -> LADDER[2] == WIDEN
    result3 = react.propose("q", "digest", history3, call=raises)
    assert result3["action"] == "WIDEN"


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

# --------------------------------------------------------------------------- A10 gold lane (tools/diag_agentic.py)

def test_gold_rows_have_id_prompt_and_nonempty_gold_terms():
    ids = set()
    for rid, prompt, gold_terms in diag_agentic.ROWS:
        assert rid and isinstance(rid, str)
        assert rid not in ids, f"duplicate row id {rid!r}"
        ids.add(rid)
        assert prompt and isinstance(prompt, str)
        assert gold_terms, f"{rid} has empty gold_terms"
        assert all(isinstance(t, str) and t for t in gold_terms)
    assert len(diag_agentic.ROWS) == 9


def test_score_row_pass_rule_is_full_gold_recall_evidence():
    bnd = mk_bundle([1])
    walk = fake_walk([(bnd, FakeEv(digest="kurt cobain nirvana"))] * 2)
    judge = fake_judge({1: "entails"}, [])

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_sufficient, max_iters=1,
                        gold_terms=["kurt cobain", "nirvana"])
    sc = diag_agentic.score_row(result, answer_text="kurt cobain",
                                 gold_terms=["kurt cobain", "nirvana"])
    assert sc["evid"] == 1.0
    assert sc["passed"] is True
    assert sc["ans"] == 0.5

    # a partial hit on the final iteration's accumulated evidence fails
    walk2 = fake_walk([(bnd, FakeEv(digest="kurt cobain only"))] * 2)
    result2 = react.run(None, None, "q", walk_fn=walk2, judge_fn=judge,
                         propose_fn=always_sufficient, max_iters=1,
                         gold_terms=["kurt cobain", "nirvana"])
    sc2 = diag_agentic.score_row(result2, gold_terms=["kurt cobain", "nirvana"])
    assert sc2["evid"] < 1.0
    assert sc2["passed"] is False
    assert sc2["ans"] == 0.0    # no answer_text given


def test_gate_reason_ok_for_non_superlative_with_cited_entail():
    result = {"entails": [1], "ords": [1, 2],
              "verdicts": {1: {"why": "the chunk describes the attack directly"}}}
    assert diag_agentic.gate_reason(result, "what happened at pearl harbor",
                                     "It was a surprise attack. #1") == "ok"


def test_gate_reason_zero_entails():
    result = {"entails": [], "ords": [1, 2], "verdicts": {}}
    assert diag_agentic.gate_reason(result, "what happened at pearl harbor",
                                     "No answer.") == "zero-entails"


def test_gate_reason_superlative_pin_for_g9_style_row():
    # entails exist but none carries a population-ranking superlative claim
    result = {"entails": [1], "ords": [1, 2],
              "verdicts": {1: {"why": "this hurricane caused major damage"}}}
    assert diag_agentic.gate_reason(
        result, "what was the deadliest hurricane on record",
        "The deadliest was Hurricane X. #1") == "superlative"


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
    assert result["stop_reason"] in ("sufficient", "budget", "no-op action", "no-movement")


# ---------------------------------------- A3/A5 amendments (fixed-point guard)

def test_filter_query_add_drops_prompt_echoes():
    """Live G1 receipt: the judge proposed '1990s famous musician' for a
    1990s-musician prompt -- all echoes, all dropped."""
    out = react.filter_query_add("1990s famous musician",
                                 "who is the most famous musician of the 1990's?")
    assert out == ""
    out2 = react.filter_query_add("nirvana kurt musician",
                                  "who is the most famous musician of the 1990's?")
    assert out2 == "nirvana kurt"


def test_cap_bundle_keeps_top_n_by_score_and_never_mutates():
    bnd = mk_bundle(list(range(10)))
    bnd = dataclasses.replace(bnd, scores={o: float(o) for o in range(10)})
    capped = react.cap_bundle(bnd, 3)
    assert capped.sampled == [9, 8, 7]
    assert set(capped.scores) == {9, 8, 7}
    assert len(bnd.sampled) == 10          # input untouched
    assert react.cap_bundle(bnd, 100) is bnd  # within cap -> unchanged


def _dil_record(i, action, mean, n):
    return react.IterationRecord(
        i=i, params={}, query="q", n_chunks=n, mean_score=mean, sdev_score=0.0,
        entails=0, contradicts=0, neutrals=0, per_cid_mean={}, new_chunks=0,
        precision_proxy=0.0, contradiction_rate=0.0, action=action)


# ---------------------------------------- A12 dilution detection + ladder reorder

def test_ladder_reorder_is_reanchor_deepen_widen():
    assert react.LADDER == ("REANCHOR", "DEEPEN", "WIDEN")


def test_dilution_detected_fires_on_widen_drop_with_rising_n():
    history = [_dil_record(0, None, 1.0, 10),
               _dil_record(1, "REANCHOR", 1.2, 12),
               _dil_record(2, "DEEPEN", 1.1, 15),
               _dil_record(3, "WIDEN", 0.3, 20)]
    assert react.dilution_detected(history) == "WIDEN"


def test_dilution_detected_needs_three_iterations():
    history = [_dil_record(0, None, 1.0, 10), _dil_record(1, "WIDEN", 0.01, 20)]
    assert react.dilution_detected(history) is None


def test_dilution_detected_skips_nonpositive_means():
    # a zero mean_score record must be skipped (no log2), not crash
    history = [_dil_record(0, None, 1.0, 10),
               _dil_record(1, "REANCHOR", 0.0, 12),
               _dil_record(2, "DEEPEN", 1.1, 15),
               _dil_record(3, "WIDEN", 0.3, 20)]
    assert react.dilution_detected(history) == "WIDEN"


def test_diluted_action_excluded_and_substituted_in_run():
    """A12: once WIDEN is flagged as dilution, the loop must not walk it
    again even when the proposer keeps proposing it -- the deterministic
    ladder (REANCHOR, DEEPEN, WIDEN) supplies the first non-excluded
    substitute, and the substitution is recorded on the next record."""
    bnds = [mk_bundle([1, 2], {1: 2.0, 2: 2.0}),
            mk_bundle([1, 2, 3], {1: 2.2, 2: 2.2, 3: 2.2}),
            mk_bundle([1, 2, 3, 4], {1: 2.1, 2: 2.1, 3: 2.1, 4: 2.1}),
            mk_bundle(list(range(1, 10)), {i: 0.3 for i in range(1, 10)}),
            mk_bundle(list(range(1, 12)), {i: 0.3 for i in range(1, 12)})]
    walk = fake_walk([(b, FakeEv()) for b in bnds])
    judge = fake_judge({}, [])

    def always_widen(query, digest, history, missing_hint=None):
        return {"sufficient": False, "missing": [], "action": "WIDEN",
                "query_add": "", "why": "", "source": "stub"}

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_widen, max_iters=4)
    iters = result["iterations"]
    # iteration 3's WIDEN drops mean_score with n rising -> dilution fires;
    # iteration 4 must NOT be WIDEN despite the proposer asking for it again.
    assert iters[3].action == "WIDEN"
    assert iters[4].action != "WIDEN"
    assert iters[4].excluded == "WIDEN"
    # the substitution shows up in the renderable transcript
    assert "WIDEN>" in result["history_table"]


# ---------------------------------------- A13 entails-first answer bundle

def test_entails_first_bundle_orders_entails_before_neutrals_and_caps():
    bnd = mk_bundle(list(range(6)), {0: 1.0, 1: 5.0, 2: 2.0, 3: 9.0, 4: 3.0, 5: 8.0})
    verdict_of = {1: {"verdict": "entails"}, 3: {"verdict": "neutral"},
                  5: {"verdict": "entails"}}
    capped = react.entails_first_bundle(bnd, verdict_of, n=3)
    assert capped.sampled == [5, 1, 3]          # entails by score desc, then best fill
    assert set(capped.scores) == {5, 1, 3}

    uncapped = react.entails_first_bundle(bnd, verdict_of, n=10)
    assert uncapped.sampled == [5, 1, 3, 4, 2, 0]
    assert bnd.sampled == [0, 1, 2, 3, 4, 5]    # input never mutated


def test_run_result_includes_answer_bundle():
    bnd0 = mk_bundle([1, 2])
    bnd1 = mk_bundle([2, 3])
    walk = fake_walk([(bnd0, FakeEv()), (bnd1, FakeEv())])
    judge = fake_judge({1: "entails"}, [])

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_insufficient, max_iters=1)
    ab = result["answer_bundle"]
    assert ab is not None
    assert 1 in ab.sampled          # the entail found in iteration 0 survives
    assert set(ab.sampled) <= {1, 2, 3}
    assert result["bundle"].sampled == [1, 2, 3]   # union across both walks


# ---------------------------------------- A17(a) yield column

def test_stats_yield_ratio_is_entails_over_n_chunks():
    bnd = mk_bundle([1, 2, 3, 4, 5, 6, 7, 8])
    rr = {"verdicts": [{"ord": 1, "verdict": "entails"}]}
    st = react.stats(bnd, rr)
    assert st["yield_ratio"] == round(1 / 8, 4)
    # precision_proxy is degenerate at 1.00 (zero contradicts) while yield
    # tells 1/8 apart from a fuller walk -- that is the whole point of A17(a)
    assert st["precision_proxy"] == 1.0

    rr2 = {"verdicts": [{"ord": o, "verdict": "entails"} for o in [1, 2, 3, 4, 5, 6]]}
    st2 = react.stats(bnd, rr2)
    assert st2["yield_ratio"] == round(6 / 8, 4)
    assert st2["precision_proxy"] == 1.0     # still degenerate, both zero contradicts

    empty = mk_bundle([])
    assert react.stats(empty, None)["yield_ratio"] == 0.0


def test_render_history_carries_a_yield_column():
    r = react.IterationRecord(
        i=0, params={}, query="q", n_chunks=8, mean_score=1.0, sdev_score=0.0,
        entails=1, contradicts=0, neutrals=7, per_cid_mean={}, new_chunks=8,
        precision_proxy=1.0, contradiction_rate=0.0, yield_ratio=0.125)
    table = react.render_history([r])
    assert "yield" in table.splitlines()[0]
    assert "0.12" in table.splitlines()[1] or "0.13" in table.splitlines()[1]


def test_sufficiency_system_names_yield_as_the_snr():
    assert "yield" in react.SUFFICIENCY_SYSTEM
    assert "degenerate" in react.SUFFICIENCY_SYSTEM.lower()


# ---------------------------------------- A17(b) per-token query dedup

def test_apply_action_dedups_reanchor_terms_per_token():
    before = {"ef": 64, "ring_top": 0, "ring_per": 0, "k_anchor": 3, "m": 3,
              "bridge_pairs": 0, "expand_aliases": False}
    q = "who is the most famous musician of the 1990s"
    p1, q1 = react.apply_action(dict(before), "REANCHOR", query=q,
                                 query_add="grunge nirvana")
    assert q1 == q + " grunge nirvana"
    # re-appending the SAME phrase a second time, plus one genuinely new word,
    # must keep exactly one copy of each already-present token and add only
    # the new one -- the live defect re-appended a whole already-present
    # multi-word phrase because the old check tested the WHOLE string.
    p2, q2 = react.apply_action(p1, "REANCHOR", query=q1,
                                 query_add="grunge nirvana pearl jam")
    assert q2 == q1 + " pearl jam"
    assert q2.lower().split().count("nirvana") == 1
    assert q2.lower().split().count("grunge") == 1

    # case-insensitive dedup
    p3, q3 = react.apply_action(dict(before), "REANCHOR", query="kurt cobain",
                                 query_add="Kurt Cobain Dave Grohl")
    assert q3 == "kurt cobain Dave Grohl"


# ---------------------------------------- A17(c) no-movement stop

def test_no_movement_stop_fires_on_a_flat_history():
    """A17(c): once the walk has run long enough to have two full iterations
    of history beyond the base (i>=2), a mean/per-cid/entails-flat iteration
    stops the loop rather than spending the rest of the budget for nothing."""
    bnd = mk_bundle([1, 2, 3, 4], scores={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    ev = FakeEv(cid_of={1: "a", 2: "a", 3: "b", 4: "b"})
    walk = fake_walk([(bnd, ev)] * 10)
    judge = fake_judge({1: "entails"}, [])   # entails found once, at iteration 0

    def widen_forever(query, digest, history, missing_hint=None):
        return {"sufficient": False, "missing": [], "action": "WIDEN",
                "query_add": "", "why": "", "source": "stub"}

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=widen_forever, max_iters=5)
    assert result["stop_reason"] == "no-movement"
    # must not fire at iteration 0 or 1 -- earliest possible stop is i=2
    assert result["n_iters"] == 2


def test_no_movement_stop_does_not_fire_when_entails_increase():
    bnds = [mk_bundle([1, 2]), mk_bundle([1, 2, 3]),
            mk_bundle([1, 2, 3, 4]), mk_bundle([1, 2, 3, 4, 5])]
    walk = fake_walk([(b, FakeEv()) for b in bnds])
    # every newly-seen ord entails -- entails strictly increases each iteration
    judge = fake_judge({1: "neutral", 2: "neutral", 3: "entails",
                         4: "entails", 5: "entails"}, [])

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_insufficient, max_iters=3)
    assert result["stop_reason"] != "no-movement"
    assert result["stop_reason"] == "budget"


def test_no_movement_stop_does_not_fire_while_ent_total_is_zero():
    """A17(c) carve-out: a walk that has found NOTHING yet is the ordinary
    zero-entail case (A6/A2), not stagnation -- it keeps spending budget on
    purpose rather than stopping early just because entails stayed at 0."""
    bnd = mk_bundle([1, 2, 3, 4], scores={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    ev = FakeEv(cid_of={1: "a", 2: "a", 3: "b", 4: "b"})
    walk = fake_walk([(bnd, ev)] * 10)
    judge = fake_judge({}, [])   # nothing ever entails

    result = react.run(None, None, "q", walk_fn=walk, judge_fn=judge,
                        propose_fn=always_insufficient, max_iters=3)
    assert result["stop_reason"] != "no-movement"
    assert result["stop_reason"] == "budget"


# ---------------------------------------- A17(d) honest stop semantics

def test_stop_reason_label_clarifies_budget_is_not_sufficiency():
    import walker_core
    assert walker_core.stop_reason_label("budget") != "budget"
    assert "iterations exhausted" in walker_core.stop_reason_label("budget")
    assert "sufficiency" in walker_core.stop_reason_label("budget")
    assert walker_core.stop_reason_label("sufficient") == "sufficient"
    assert walker_core.stop_reason_label("no-movement") == "no-movement"


def test_repeated_identical_walk_stops_as_fixed_point():
    """A3(c): a proposer that always echoes the prompt (empty after filter ->
    DEEPEN) still may not re-walk a seen (query, params); after one ladder
    escalation the loop stops rather than spin."""
    bnd = mk_bundle([1])
    walk = fake_walk([(bnd, FakeEv(digest="d"))] * 10)
    judge = fake_judge({}, [])

    def echo_proposer(query, digest, history, missing_hint=None):
        return {"sufficient": False, "missing": [], "action": "REANCHOR",
                "query_add": "famous musician", "why": "", "source": "model"}

    res = react.run(None, None, "who is the most famous musician?",
                    walk_fn=walk, judge_fn=judge, propose_fn=echo_proposer,
                    max_iters=5)
    assert res["stop_reason"] in ("fixed-point", "budget")
    queries = [r.query for r in res["iterations"]]
    assert len(queries) == len(set((q, tuple(sorted(r.params.items())))
                                    for q, r in zip(queries, res["iterations"]))), \
        "no (query, params) pair may repeat"


# ------------------------------------------- A19 the answer cap must not starve

def test_answer_cap_matches_cap_bundle_default():
    """A19 KNOWN-BAD: the call site hardcoded 50 while cap_bundle's documented
    default is 100 -- a second, tighter budget stacked on the renderer's real
    one. Measured on ab-document after R25: the base walk found 10 gold-bearing
    chunks and the 50-cap discarded 7 of them before the judge saw anything."""
    import inspect
    assert react.ANSWER_CAP == 100
    assert inspect.signature(react.cap_bundle).parameters["n"].default == react.ANSWER_CAP
    src = inspect.getsource(react.run)
    assert "entails_first_bundle(union_bundle, verdict_of, ANSWER_CAP)" in src, \
        "the answer bundle must use the named constant, not a literal"


def test_entails_first_cannot_rescue_a_cut_tail():
    """Why raising the cap was the fix and reordering was not: on the FIRST pass
    the judge has entailed nothing, so entails_first_bundle degenerates to score
    order and cuts exactly the tail the walk just earned."""
    b = mk_bundle(range(10), scores={o: float(10 - o) for o in range(10)})
    got = react.entails_first_bundle(b, {}, 5)          # nothing entailed yet
    assert list(got.sampled) == [0, 1, 2, 3, 4]         # pure score order
    assert 9 not in got.sampled                          # the tail is gone
    got2 = react.entails_first_bundle(b, {9: {"verdict": "entails"}}, 5)
    assert 9 in got2.sampled                             # only AFTER a verdict
