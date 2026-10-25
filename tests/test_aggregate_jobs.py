"""Verify durable global aggregate job state and recovery contracts.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| CALL | Verify stable keys reuse committed provider responses and reject uncertainty. | [mandatory] | T1-T2 |
| RECOVER | Reopen a database and classify interrupted jobs/calls explicitly. | [mandatory] | T3 |
| RUN | Verify bounded asynchronous completion and immutable manifests. | [mandatory] | T4 |

T1 A successful response is returned from SQLite without a second callback invocation.
T2 Failed or uncertain keys cannot be invisibly retried.
T3 Reopening marks abandoned in-flight calls uncertain and running jobs incomplete.
T4 Job result and completion state survive readback; budget extension is rejected.

Preconditions: pytest and stdlib SQLite are available; tests use isolated temporary databases.
Failure modes: persistence, status-transition, or idempotency regressions fail these tests.
"""
from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from aggregate_jobs import AggregateJobs
import global_aggregate as ga


def test_committed_call_reuses_response_and_uncertain_call_blocks(tmp_path):
    """T1/T2: logical call keys never repeat completed or ambiguous provider work."""
    path = tmp_path / "jobs.sqlite"
    jobs = AggregateJobs(path)
    job = jobs.create({"predicate": "x"}, {"build": 4, "budget": 2})
    count = 0

    def callback():
        nonlocal count
        count += 1
        return {"label": "yes"}

    assert jobs.call(job, "paper:a", callback) == {"label": "yes"}
    assert jobs.call(job, "paper:a", callback) == {"label": "yes"}
    assert count == 1
    with pytest.raises(RuntimeError, match="provider timeout"):
        jobs.call(job, "paper:b", lambda: (_ for _ in ()).throw(RuntimeError("provider timeout")))
    with pytest.raises(RuntimeError, match="explicit reconciliation"):
        jobs.call(job, "paper:b", callback)
    snapshot = jobs.get(job)
    assert snapshot.calls_attempted == 2
    assert snapshot.calls_completed == 1
    assert snapshot.calls_uncertain == 1
    jobs.close()


def test_reopen_marks_interrupted_call_uncertain_and_job_incomplete(tmp_path):
    """T3: a process restart exposes uncertainty rather than assuming completion."""
    path = tmp_path / "jobs.sqlite"
    jobs = AggregateJobs(path)
    job = jobs.create({}, {"seed": 13})
    with sqlite3.connect(path) as db:
        db.execute("UPDATE aggregate_job SET status='running' WHERE job_id=?", (job,))
        db.execute("INSERT INTO aggregate_job_call VALUES (?, 'paper:x', 'in_flight', NULL, NULL, ?, NULL)",
                   (job, time.time()))
    jobs.close()

    reopened = AggregateJobs(path)
    snapshot = reopened.get(job)
    assert snapshot.status == "incomplete"
    assert snapshot.calls_attempted == 1
    assert snapshot.calls_uncertain == 1
    with pytest.raises(RuntimeError, match="explicit reconciliation"):
        reopened.call(job, "paper:x", lambda: {"label": "no"})
    reopened.close()


def test_runner_persists_result_and_rejects_budget_changes(tmp_path):
    """T4: bounded runner transitions and manifest immutability survive readback."""
    jobs = AggregateJobs(tmp_path / "jobs.sqlite", max_workers=1)
    job = jobs.create({"question": "count"}, {"budget": 20, "seed": 1})
    started = threading.Event()
    release = threading.Event()

    def runner(_job_id):
        started.set()
        release.wait(timeout=3)
        return {"count": 7}

    assert jobs.start(job, runner)
    assert started.wait(timeout=1)
    assert jobs.get(job).status == "running"
    release.set()
    deadline = time.monotonic() + 2
    while jobs.get(job).status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    snapshot = jobs.get(job)
    assert snapshot.status == "completed"
    assert snapshot.result == {"count": 7}
    assert snapshot.manifest["budget"] == 20
    with pytest.raises(ValueError, match="immutable"):
        jobs.resume(job, runner, additional_calls=1)
    jobs.close()


def test_semantic_tally_uses_job_call_ledger_after_label_cache_miss(tmp_path):
    """The durable logical-call ledger restores paid labels when the tally cache is absent."""
    jobs = AggregateJobs(tmp_path / "jobs.sqlite")
    job = jobs.create({"predicate": "target"}, {"source_build": 8, "budget": 2})
    population = {"a": 1, "b": 1}
    config = ga.SemanticTallyConfig(budget=2, pilot_budget=2, seed=10,
                                    cache_path=tmp_path / "labels.sqlite")
    loader = lambda ids, _cap: {paper_id: {"text": paper_id, "truncated": False} for paper_id in ids}
    provider_calls = []

    def executor(key, callback):
        return jobs.call(job, key, callback)

    first = ga.semantic_tally(population, "target", loader,
                              lambda *_: provider_calls.append("first") or {"label": "yes"},
                              config, 8, call_executor=executor)
    assert first.calls == 2
    assert provider_calls == ["first", "first"]

    config = ga.SemanticTallyConfig(budget=2, pilot_budget=2, seed=10,
                                    cache_path=tmp_path / "empty-label-cache.sqlite")
    second = ga.semantic_tally(population, "target", loader,
                               lambda *_: provider_calls.append("unexpected") or {"label": "no"},
                               config, 8, call_executor=executor)
    assert second.calls == 2
    assert second.estimate == first.estimate
    assert provider_calls == ["first", "first"]
    jobs.close()
