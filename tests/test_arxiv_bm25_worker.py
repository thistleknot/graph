"""Pin stale-build detection for the automatic BM25 companion index."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import arxiv_bm25_worker as worker


class FakeConnection:
    def __init__(self, graph=None, bm25=None):
        self.graph = graph
        self.bm25 = bm25

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def fetchone(self):
        return self.row

    def execute(self, query, params=None):
        if "sect_build" in query:
            self.row = self.graph
        elif query.startswith("SELECT (SELECT count(*)"):
            count = self.bm25[1]
            dim = self.bm25[2].get("bm25", {}).get("dim", 2)
            self.row = (count, count, dim)
        else:
            self.row = self.bm25
        return self


def test_index_is_needed_when_no_bm25_build_exists():
    assert worker.index_needed(FakeConnection(graph=(12, {})), FakeConnection())[0] is True


def test_index_is_current_only_when_bm25_records_the_same_graph_build():
    current = FakeConnection(graph=(12, {}), bm25=(4, 100, {"source_graph_build": 12, "bm25": {"dim": 2}}))
    stale = FakeConnection(graph=(13, {}), bm25=(4, 100, {"source_graph_build": 12, "bm25": {"dim": 2}}))
    assert worker.index_needed(current, current) == (False, 12)
    assert worker.index_needed(stale, stale) == (True, 13)


def test_worker_refuses_to_claim_current_without_a_live_graph_build():
    with pytest.raises(RuntimeError, match="no live section graph build"):
        worker.index_needed(FakeConnection(), FakeConnection())
