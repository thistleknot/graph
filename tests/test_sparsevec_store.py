"""Pins sparsevec_store's pure parts: the wire format and the nonzero cap.

No database. The literal format is the contract with pgvector -- 1-based indices,
ascending, '/dim' suffix -- and a wrong index base would silently shift every
weight onto the neighbouring piece, which no recall number would reveal as a
format bug. The cap test pins V2: keep the HEAVIEST 1000, not the first 1000.

Run:  pytest tests/test_sparsevec_store.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sparsevec_store as ss


def test_literal_is_one_based_ascending_with_dim_suffix():
    assert ss._literal({4: 0.5, 0: 2.0, 2: 1.25}, 10) == "{1:2,3:1.25,5:0.5}/10"


def test_literal_accepts_pairs_and_rounds_to_six_significant():
    assert ss._literal([(1, 0.123456789)], 3) == "{2:0.123457}/3"


def test_literal_empty_is_valid_zero_vector():
    assert ss._literal({}, 7) == "{}/7"


def test_table_name_is_sanitised_and_prefixed():
    assert ss._table("Sub-100k v2") == "lex_chunk_sub_100k_v2"
    assert ss._table("arxiv") == "lex_chunk_arxiv"


def test_cap_keeps_heaviest_not_first(monkeypatch):
    """V2 via the same sort write_chunks uses: heaviest survive, by weight."""
    monkeypatch.setattr(ss, "SPARSEVEC_MAX_NNZ", 3)
    items = [(0, 0.1), (1, 9.0), (2, 0.2), (3, 5.0), (4, 7.0)]
    items.sort(key=lambda kv: -kv[1])
    kept = items[:ss.SPARSEVEC_MAX_NNZ]
    assert sorted(i for i, _ in kept) == [1, 3, 4]
    assert ss._literal(kept, 5) == "{2:9,4:5,5:7}/5"
