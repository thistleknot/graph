"""tests/conftest.py -- shared sys.path setup and live-Postgres/neo4j skip helpers.

Every test file used to carry its own `sys.path.insert(0, str(Path(__file__)...))`
line and its own try/except-pytest.skip boilerplate around a Postgres or neo4j
probe. This is the one place both live now.

Spec: .spec/specs/graph-explorer/design.md §6.21(d)
Task: playbook.md T33
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
_TESTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_REPO_ROOT, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import psycopg
import pytest

import graph_tools as gt


def require_dsn_db(dsn):
    """Preflight-checks a Postgres DSN is reachable; skips the test cleanly if not."""
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no database: {exc}")


def require_gt_conn():
    """Connects via graph_tools.connect(); skips the test cleanly if Postgres is down."""
    try:
        return gt.connect()
    except psycopg.OperationalError as e:          # pragma: no cover
        pytest.skip(f"no database: {e}")


def require_run(conn, label):
    """Fetches a run by label; skips the test cleanly if it doesn't exist."""
    try:
        return gt.get_run(conn, label)
    except Exception as e:                         # pragma: no cover
        pytest.skip(f"no live run {label!r}: {e}")


# Fixture names, across the suite, that only ever gate live Postgres or live
# network access (verified unique by grep -- no non-DB fixture reuses them).
# `-m` selection happens at collection time, before any fixture runs, so a
# marker added inside the fixture itself is too late; this hook tags the test
# item once, at collection, for every test whose fixture closure touches one.
_LIVE_DB_FIXTURES = {"db", "conn", "run", "live", "app"}
_LIVE_NET_FIXTURES = {"neo4j_up"}


def pytest_collection_modifyitems(items):
    for item in items:
        names = set(getattr(item, "fixturenames", ()))
        if names & _LIVE_DB_FIXTURES:
            item.add_marker(pytest.mark.live_db)
        if names & _LIVE_NET_FIXTURES:
            item.add_marker(pytest.mark.live_net)
