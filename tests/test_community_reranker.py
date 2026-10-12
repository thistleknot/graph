"""Pins src/community_reranker.py: score alignment through batching, the solo-request default, and the payload shape. The service is faked: no GPU, no network.

Guard: a score is only a pair score when the request holds one summary (measured 2026-10-12, see community_reranker.configured_batch_size).
Spec: handoff.md (2026-10-12). No other governing spec.

Run:  pytest tests/test_community_reranker.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import community_reranker as cr


def page(*cids):
    return {c: {"title": "t%d" % c, "summary": "summary of %d" % c, "size": 3} for c in cids}


class FakeService:
    """Answers each NOUL question r<i> with a score from the community it names, so a misaligned batch shows."""
    def __init__(self, score_of):
        self.score_of, self.requests = score_of, []

    def __call__(self, body):
        req = json.loads(body.decode("utf-8"))
        self.requests.append(req)
        labels = {c["id"] for c in req["state"]["community_summaries"]}
        answers = {}
        for key, q in req["questions"].items():
            label = q["instructions"].split()[2]                                  # "Is summary c12 relevant ..."
            assert label in labels                                                # a question only names a summary of its own request
            answers[key] = {"type": "noul", "noul": self.score_of(int(label[1:]))}
        return {"answers": answers, "usage": {"input_tokens": 1}}, 0.01


@pytest.fixture
def service(monkeypatch):
    svc = FakeService(lambda c: (c % 10) / 10)
    monkeypatch.setattr(cr, "_post", svc)
    monkeypatch.delenv("COMMUNITY_JEV_BATCH_SIZE", raising=False)
    return svc


def test_the_default_is_one_summary_per_request_so_a_score_is_a_pair_score(service):
    assert cr.configured_batch_size() == 1 and cr.MIN_BATCH_SIZE == 1
    scores, stats = cr.score_communities("q", [page(12, 13, 14), page(15, 16)])
    assert len(service.requests) == 5 and all(len(r["state"]["community_summaries"]) == 1 for r in service.requests)
    assert [s["community"] for s in scores] == [12, 13, 14, 15, 16] and [s["score"] for s in scores] == [0.2, 0.3, 0.4, 0.5, 0.6]
    assert [b["candidates"] for b in stats] == [1, 1, 1, 1, 1]


def test_a_larger_batch_is_still_possible_up_to_128_and_every_score_stays_with_its_community(service, monkeypatch):
    monkeypatch.setenv("COMMUNITY_JEV_BATCH_SIZE", "4")
    scores, stats = cr.score_communities("q", [page(*range(20, 30))])
    assert [b["candidates"] for b in stats] == [4, 4, 2] and [s["community"] for s in scores] == list(range(20, 30))
    assert [s["score"] for s in scores] == [(c % 10) / 10 for c in range(20, 30)]            # score i belongs to community i through the batching
    monkeypatch.setenv("COMMUNITY_JEV_BATCH_SIZE", "129")
    with pytest.raises(ValueError):
        cr.configured_batch_size()


def test_the_payload_names_each_summary_by_its_own_label_and_carries_the_query_once(service):
    cr.score_communities("what is x?", [page(7)])
    req = service.requests[0]
    assert req["state"]["query"] == "what is x?" and req["state"]["community_summaries"] == [{"id": "c7", "title": "t7", "summary": "summary of 7"}]
    assert req["questions"] == {"r0": {"type": "noul", "instructions": "Is summary c7 relevant to answering the query?"}}


def test_a_response_without_a_noul_score_stops_the_run_rather_than_inventing_one(monkeypatch):
    monkeypatch.setattr(cr, "_post", lambda body: ({"answers": {"r0": {"type": "choice"}}}, 0.0))
    monkeypatch.delenv("COMMUNITY_JEV_BATCH_SIZE", raising=False)
    with pytest.raises(RuntimeError, match="omitted NOUL"):
        cr.score_communities("q", [page(1)])


def test_the_threshold_is_a_probability(monkeypatch):
    monkeypatch.setenv("COMMUNITY_JEV_THRESHOLD", "0.65")
    assert cr.relevance_threshold() == 0.65
    monkeypatch.setenv("COMMUNITY_JEV_THRESHOLD", "1.5")
    with pytest.raises(ValueError):
        cr.relevance_threshold()
