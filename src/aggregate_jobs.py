"""Durable execution state for bounded global aggregate jobs.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| STORE | Persist immutable manifests, progress, results, and call ledger in the tally SQLite file. | [mandatory] | J1-J4 |
| RESERVE | Mark a logical provider call in-flight before invoking its callback. | [mandatory] | J3-J5 |
| COMMIT | Store each successful provider response immediately under its stable call key. | [mandatory] | J3-J5 |
| RECOVER | Surface uncertain in-flight calls and make only safely resumable jobs incomplete. | [mandatory] | J6 |
| SERVE | Run accepted jobs in a bounded process-local executor. | [mandatory] | J7 |

EARS guards
J1 A job manifest is immutable after creation; it pins source, model, prompt, seed, draw order, and budgets.
J2 A logical call key is unique within a job and a completed call is returned without invoking its callback.
J3 A call reservation is committed before callback execution; a response is committed before returning.
J4 Provider errors and process loss never produce a successful response record.
J5 Call attempts and completed calls are separately counted and visible in snapshots.
J6 Startup converts abandoned running jobs to incomplete and in-flight calls to uncertain; uncertain calls are never silently retried.
J7 The executor bounds concurrent jobs; provider concurrency is bounded separately by the caller's shared semaphore.

CLOSED
- PostgreSQL job rows are rejected because this workflow already checkpoints classification in a local SQLite database.
- Replaying an unconfirmed provider request is rejected because the provider may have completed and billed it.

Preconditions: the SQLite parent directory is writable; manifests and snapshots are JSON serializable.
Failure modes: SQLite errors, duplicate job IDs, malformed stored JSON, uncertain-call reuse, or callback errors are explicit.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Mapping

JOB_DDL = (
    "CREATE TABLE IF NOT EXISTS aggregate_job (job_id TEXT PRIMARY KEY, status TEXT NOT NULL, "
    "manifest_json TEXT NOT NULL, progress_json TEXT NOT NULL, result_json TEXT, error TEXT, "
    "created_at REAL NOT NULL, updated_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS aggregate_job_call (job_id TEXT NOT NULL, call_key TEXT NOT NULL, "
    "status TEXT NOT NULL, response_json TEXT, error TEXT, started_at REAL NOT NULL, completed_at REAL, "
    "PRIMARY KEY (job_id, call_key), FOREIGN KEY(job_id) REFERENCES aggregate_job(job_id))",
)


@dataclass(frozen=True)
class JobSnapshot:
    """Public persisted state for one aggregate job.

    Require: `manifest`, `progress`, and optional `result` were decoded from its SQLite row.
    Guarantee: callers receive status and logical/actual call counts without private connection state.
    Maintain: fields describe one committed snapshot.
    Assert: status is one of queued, running, incomplete, completed, or failed.
    """

    job_id: str
    status: str
    manifest: Mapping[str, Any]
    progress: Mapping[str, Any]
    result: Mapping[str, Any] | None
    error: str | None
    calls_attempted: int
    calls_completed: int
    calls_uncertain: int
    calls_in_flight: int


class AggregateJobs:
    """Persist and schedule resumable aggregate work.

    Require: `path` names the shared tally SQLite database and worker count is positive.
    Guarantee: manifests/results/call responses survive process restart; jobs run in a bounded pool.
    Maintain: completed call keys are never charged or executed again; uncertain calls stay visible.
    Assert: every submitted job has a stable ID and serializable manifest.
    """

    def __init__(self, path: str | Path, max_workers: int = 2):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aggregate-job")
        self._futures: dict[str, Future] = {}
        with closing(self._connect()) as db, db:
            for statement in JOB_DDL:
                db.execute(statement)
            db.execute("UPDATE aggregate_job_call SET status='uncertain', error='process stopped before response commit' "
                       "WHERE status='in_flight'")
            db.execute("UPDATE aggregate_job SET status='incomplete', error='process stopped during execution', "
                       "updated_at=? WHERE status='running'", (time.time(),))

    def _connect(self) -> sqlite3.Connection:
        """Open a foreign-key checked SQLite connection for one short transaction.

        Returns:
            A new SQLite connection to the configured job database.
        Require: The parent directory is writable.
        Guarantee: Foreign-key enforcement is active for the connection.
        Maintain: Callers own and close the returned connection.
        Assert: The database path remains the repository's shared file.
        """
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def create(self, request: Mapping[str, Any], manifest: Mapping[str, Any], job_id: str | None = None) -> str:
        """Create a queued job with immutable request and execution manifest.

        Require: optional job_id is unique and both mappings are JSON serializable.
        Guarantee: durable queued row exists before the ID is returned.
        Maintain: request is included in the immutable manifest for recovery.
        Assert: duplicate IDs raise IntegrityError without overwriting prior work.

        Args:
            request: Immutable caller inputs.
            manifest: Pinned execution and sampling metadata.
            job_id: Optional caller-supplied stable identifier.
        Returns:
            Created job identifier.
        Raises:
            sqlite3.IntegrityError: If the identifier already exists.
        """
        job_id = job_id or uuid.uuid4().hex
        payload = dict(manifest)
        payload["request"] = dict(request)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        now = time.time()
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO aggregate_job VALUES (?, 'queued', ?, '{}', NULL, NULL, ?, ?)",
                       (job_id, encoded, now, now))
        return job_id

    def get(self, job_id: str) -> JobSnapshot | None:
        """Read committed job state and call counts.

        Require: job_id is a string.
        Guarantee: absent IDs return None; present IDs return one consistent persisted snapshot.
        Maintain: call counters are derived from the durable call ledger.
        Assert: stored JSON decodes to mappings.

        Args:
            job_id: Durable job identifier.
        Returns:
            JobSnapshot when present, otherwise None.
        """
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT status, manifest_json, progress_json, result_json, error FROM aggregate_job "
                             "WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            counts = dict(db.execute("SELECT status, count(*) FROM aggregate_job_call WHERE job_id=? GROUP BY status",
                                     (job_id,)).fetchall())
        result = json.loads(row[3]) if row[3] else None
        return JobSnapshot(job_id, row[0], json.loads(row[1]), json.loads(row[2]), result, row[4],
                           sum(counts.values()), counts.get("completed", 0),
                           counts.get("uncertain", 0), counts.get("in_flight", 0))

    def call(self, job_id: str, key: str, callback: Callable[[], Any]) -> Any:
        """Reserve, execute, and commit one idempotently keyed provider call.

        Require: job exists and callback performs exactly one logical provider call.
        Guarantee: successful responses are durable before return; errors remain explicit.
        Maintain: a completed key reuses its response; uncertain keys raise without callback execution.
        Assert: reservation is committed before callback begins.

        Args:
            job_id: Owning job identifier.
            key: Stable logical call identity.
            callback: One provider operation returning JSON-serializable output.
        Returns:
            Durable response value.
        Raises:
            RuntimeError: If the key is uncertain or callback fails.
        """
        now = time.time()
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT status, response_json FROM aggregate_job_call WHERE job_id=? AND call_key=?",
                             (job_id, key)).fetchone()
            if row and row[0] == "completed":
                return json.loads(row[1])
            if row:
                raise RuntimeError(f"aggregate call {key} is {row[0]}; explicit reconciliation required")
            db.execute("INSERT INTO aggregate_job_call VALUES (?, ?, 'in_flight', NULL, NULL, ?, NULL)",
                       (job_id, key, now))
            db.execute("UPDATE aggregate_job SET updated_at=? WHERE job_id=?", (now, job_id))
        try:
            response = callback()
            encoded = json.dumps(response, sort_keys=True, separators=(",", ":"))
        except BaseException as exc:
            with closing(self._connect()) as db, db:
                db.execute("UPDATE aggregate_job_call SET status='uncertain', error=?, completed_at=? "
                           "WHERE job_id=? AND call_key=? AND status='in_flight'",
                           (f"{type(exc).__name__}: {exc}", time.time(), job_id, key))
                db.execute("UPDATE aggregate_job SET updated_at=? WHERE job_id=?", (time.time(), job_id))
            raise
        with closing(self._connect()) as db, db:
            db.execute("UPDATE aggregate_job_call SET status='completed', response_json=?, completed_at=? "
                       "WHERE job_id=? AND call_key=? AND status='in_flight'", (encoded, time.time(), job_id, key))
        return response

    def start(self, job_id: str, runner: Callable[[str], Mapping[str, Any]]) -> bool:
        """Submit one queued job to the bounded executor.

        Require: runner accepts job_id and returns a JSON-safe result mapping.
        Guarantee: a successfully claimed job executes asynchronously at most once per process.
        Maintain: runner errors set failed state; successful results set completed state.
        Assert: only queued/incomplete jobs can be claimed.

        Args:
            job_id: Durable job identifier.
            runner: Job execution callback.
        Returns:
            True when the runner was accepted.
        """
        with self._lock:
            if job_id in self._futures and not self._futures[job_id].done():
                return False
            with closing(self._connect()) as db, db:
                changed = db.execute("UPDATE aggregate_job SET status='running', error=NULL, updated_at=? "
                                     "WHERE job_id=? AND status IN ('queued','incomplete')",
                                     (time.time(), job_id)).rowcount
            if not changed:
                return False
            self._futures[job_id] = self._executor.submit(self._run, job_id, runner)
            return True

    def _run(self, job_id: str, runner: Callable[[str], Mapping[str, Any]]) -> None:
        """Execute runner and commit its terminal state.

        Args:
            job_id: Claimed job identifier.
            runner: Job execution callback.
        Require: The job status is running.
        Guarantee: Runner output or failure is persisted as terminal state.
        Maintain: Provider failures remain failures, not fabricated results.
        Assert: Stored successful output is JSON serializable.
        """
        try:
            result_json = json.dumps(runner(job_id), sort_keys=True, separators=(",", ":"))
        except BaseException as exc:
            with closing(self._connect()) as db, db:
                db.execute("UPDATE aggregate_job SET status='failed', error=?, updated_at=? WHERE job_id=?",
                           (f"{type(exc).__name__}: {exc}", time.time(), job_id))
            return
        with closing(self._connect()) as db, db:
            db.execute("UPDATE aggregate_job SET status='completed', result_json=?, progress_json=?, error=NULL, "
                       "updated_at=? WHERE job_id=?", (result_json, result_json, time.time(), job_id))

    def resume(self, job_id: str, runner: Callable[[str], Mapping[str, Any]], additional_calls: int = 0) -> bool:
        """Resume an incomplete/failed job without changing its original budgets.

        Require: additional_calls is zero; increasing budgets requires creating a new audited manifest.
        Guarantee: persisted completed calls remain reusable and uncertain calls block on their own keys.
        Maintain: the original manifest is unchanged.
        Assert: only incomplete/failed jobs may resume.

        Args:
            job_id: Durable job identifier.
            runner: Job execution callback.
            additional_calls: Must remain zero for an immutable job.
        Returns:
            True when resume work was accepted.
        Raises:
            ValueError: If a caller attempts to increase the total budget.
        """
        if additional_calls != 0:
            raise ValueError("job budget is immutable; create a new job to authorize additional calls")
        with closing(self._connect()) as db, db:
            db.execute("UPDATE aggregate_job SET status='incomplete', error=NULL, updated_at=? "
                       "WHERE job_id=? AND status='failed'", (time.time(), job_id))
        return self.start(job_id, runner)

    def close(self, wait: bool = True) -> None:
        """Stop accepting work and optionally wait for active runners to finish.

        Args:
            wait: Whether to wait for accepted jobs.
        Require: Caller owns the repository lifecycle.
        Guarantee: The executor no longer accepts new work.
        Maintain: Persisted jobs remain available after process restart.
        Assert: No database connection is held by this method.
        """
        self._executor.shutdown(wait=wait, cancel_futures=False)
