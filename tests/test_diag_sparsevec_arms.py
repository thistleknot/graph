"""Pins the EF CAP rule in the 2x2 comparison tool.

pgvector's HNSW scan returns at most ef_search rows (verified 2026-09-27: ef=40 with
LIMIT 100/50/40 all returned 40). So a metric at k > ef_search is computed over a
truncated list and is wrong -- which is exactly what happened to every rec@50 reported
at ef=40 the day before. `ks_for` is the guard; if it regressed to returning all ks,
rec@50 at ef=40 would silently come back.

No database.  Run:  pytest tests/test_diag_sparsevec_arms.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from diag_sparsevec_arms import KS_FULL, ks_for  # noqa: E402


def test_ef_40_drops_k50():
    assert ks_for(40) == (1, 5, 10)


def test_ef_400_keeps_every_k():
    assert ks_for(400) == KS_FULL == (1, 5, 10, 50)


def test_ef_equal_to_k_is_still_valid():
    """min(k, ef) rows is a complete top-k when ef == k."""
    assert 50 in ks_for(50)
    assert 50 not in ks_for(49)
