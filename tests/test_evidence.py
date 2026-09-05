"""tests/test_evidence.py -- pins evidence.py's pure helpers offline and
evidence.assemble end to end against a live run.

Spec: .spec/specs/graph-explorer/design.md 6.21(a)
Task: playbook.md T35

Run:  pytest tests/test_evidence.py -v      (live_db tests need docker compose up -d)
"""
from __future__ import annotations

import dataclasses

import pytest

import evidence
import graph_tools as gt
import interpret
import sampler
from conftest import require_gt_conn, require_run

# ------------------------------------------------------------ top_quartile

def test_top_quartile_empty_cross():
    q3, strong = evidence.top_quartile([])
    assert q3 == 0.0
    assert strong == []


def test_top_quartile_single_element():
    cross = [(1, "carrier", 2.0)]
    q3, strong = evidence.top_quartile(cross)
    assert q3 == 2.0
    assert strong == [(1, "carrier", 2.0)]


def test_top_quartile_uniform_weights_all_strong():
    cross = [(o, "t", 1.0) for o in range(4)]
    q3, strong = evidence.top_quartile(cross)
    assert q3 == 1.0
    assert len(strong) == 4          # every binding ties the threshold


def test_top_quartile_skewed_weights_only_tail():
    """8 elements, 5 low + 3 high: index int(0.75*7)=5 lands in the high
    block, so q3 excludes the low majority -- only the tail is 'strong'."""
    cross = ([(o, "t", 0.1) for o in range(5)]
             + [(5, "t", 9.0), (6, "t", 9.0), (7, "t", 9.0)])
    q3, strong = evidence.top_quartile(cross)
    assert q3 == 9.0
    assert strong == [(5, "t", 9.0), (6, "t", 9.0), (7, "t", 9.0)]


def test_top_quartile_pins_index_arithmetic_on_nine_elements():
    """ws_[int(0.75 * (len(ws_) - 1))] on a 9-element list -- hand-computed:
    weights 0..8, sorted is itself, index = int(0.75 * 8) = 6 -> value 6.0."""
    cross = [(i, "t", float(i)) for i in range(9)]
    q3, strong = evidence.top_quartile(cross)
    assert q3 == 6.0
    assert sorted(b[2] for b in strong) == [6.0, 7.0, 8.0]


# ----------------------------------------------------------- src_counts_of

def test_src_counts_of_two_communities():
    kept = [1, 2, 3, 4]
    cid_of = {1: 0, 2: 0, 3: 1, 4: 1}
    src_of = {1: "wiki", 2: "wiki", 3: "brown", 4: "quotes"}
    out = evidence.src_counts_of(kept, cid_of, src_of)
    assert out == {0: {"wiki": 2}, 1: {"brown": 1, "quotes": 1}}


def test_src_counts_of_missing_cid_keys_under_none():
    kept = [1]
    cid_of = {}                       # 1 has no cid entry
    src_of = {1: "wiki"}
    out = evidence.src_counts_of(kept, cid_of, src_of)
    assert out == {None: {"wiki": 1}}


def test_src_counts_of_missing_source_keys_under_unlabelled():
    kept = [1]
    cid_of = {1: 0}
    src_of = {1: None}
    out = evidence.src_counts_of(kept, cid_of, src_of)
    assert out == {0: {"unlabelled": 1}}


# ------------------------------------------------------------- resolver_of

def test_resolver_of_normal_case():
    sal = {1: {"top": ["carrier", "navy"]}}
    src_of = {1: "wiki"}
    out = evidence.resolver_of([1], src_of, sal)
    assert out == {1: "wiki:carrier"}


def test_resolver_of_empty_top_renders_question_mark():
    sal = {1: {"top": []}}
    src_of = {1: "wiki"}
    out = evidence.resolver_of([1], src_of, sal)
    assert out == {1: "wiki:?"}


def test_resolver_of_ord_absent_from_sal():
    src_of = {1: "brown"}
    out = evidence.resolver_of([1], src_of, {})
    assert out == {1: "brown:?"}


# --------------------------------------------------------- digest shaping

def test_digest_shaping_is_byte_stable_and_carries_sections():
    """Builds the render_digest argument tuple the way evidence.assemble
    does -- via top_quartile/src_counts_of/resolver_of over canned dicts --
    and asserts the output is byte-stable and contains each section header.
    Mirrors tests/test_interpret.py:780-860."""
    touched = [{"cid": 8, "hits": 74, "size": 962}]
    kw = {8: ["ship", "aircraft", "navy"]}
    kept = [1, 2]
    cid_of = {1: 8, 2: 8}
    src_of = {1: "wiki", 2: "brown"}
    sal = {1: {"top": ["carrier"]}, 2: {"top": ["army"]}}
    src_counts = evidence.src_counts_of(kept, cid_of, src_of)
    resolver = evidence.resolver_of(kept, src_of, sal)
    cross = [(1, "carrier", 2.0), (1, "midway", 1.0), (2, "navy", 0.5)]
    _, strong = evidence.top_quartile(cross)
    args = (touched, kw, src_counts, [[1, 2]], [["carrier", "torpedo", "midway"]],
            {"carrier", "midway"}, {"carrier", "navy"}, strong,
            {"pairs": [{"a": 1, "b": 2, "dwpc": 0.5, "path": [1, 9, 2]}]})
    d1 = interpret.render_digest(*args, resolver=resolver)
    assert d1 == interpret.render_digest(*args, resolver=resolver)
    assert "== DIGEST" in d1
    assert "c8|74/962|" in d1
    assert "1=wiki:carrier" in d1


# -------------------------------------------------- live: evidence.assemble

Q_A = "how did aircraft carriers decide the battle of midway"
Q_B = "a quote about courage"


@pytest.fixture(scope="module")
def conn():
    c = require_gt_conn()
    yield c
    c.close()


@pytest.fixture(scope="module")
def run(conn):
    return require_run(conn, "mixed-full-dual")


def test_assemble_is_deterministic(conn, run):
    embed = evidence.load_embed(__import__("config").MODEL_DIR)
    for q, knobs in ((Q_A, {}), (Q_B, {"ef": 24, "T": 0.0, "bridge_pairs": 0})):
        bnd, _ = sampler.ef_evidence(conn, run, q, seed=0, **knobs)
        assert bnd.sampled, f"prompt {q!r} matched nothing"
        ev1 = evidence.assemble(conn, run, bnd, embed=embed)
        ev2 = evidence.assemble(conn, run, bnd, embed=embed)
        assert ev1.digest == ev2.digest
        assert ev1.touched == ev2.touched
        assert ev1.cids == ev2.cids


def test_assemble_small_walk_has_no_dendrite(conn, run):
    bnd, _ = sampler.ef_evidence(conn, run, Q_A, seed=0)
    assert bnd.sampled
    small = dataclasses.replace(bnd, sampled=bnd.sampled[:3])
    ev = evidence.assemble(conn, run, small, embed=None)
    assert ev.dendrite is None
    assert ev.digest is None
    # the walk_state half must survive the dendrite half returning nothing
    assert ev.touched
    assert ev.cid_of
    assert ev.pathways is not None


def test_assemble_field_contract(conn, run):
    bnd, _ = sampler.ef_evidence(conn, run, Q_A, seed=0)
    assert bnd.sampled
    ev = evidence.assemble(conn, run, bnd, embed=None)
    for f in dataclasses.fields(evidence.Evidence):
        assert hasattr(ev, f.name)
    assert set(ev.in_cid) == set(ev.cids)
    assert set(ev.cid_of) == set(bnd.sampled)
