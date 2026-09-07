"""tests/test_evidence.py -- pins evidence.py's pure helpers offline and
evidence.assemble end to end against a live run.

Spec: .spec/specs/graph-explorer/design.md 6.21(a)
Task: playbook.md T35

Run:  pytest tests/test_evidence.py -v      (live_db tests need docker compose up -d)
"""
from __future__ import annotations

import dataclasses
import urllib.error

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


# ---- mirror writes (6.21(c)) ----

def _base_evidence(**overrides):
    base = dict(
        touched=[], cids=[], concept={}, terms={}, cid_of={}, in_cid={},
        xedges={}, medoids={}, salient={}, term_members={}, pa_anchors=[],
        pathways={"pairs": []}, dendrite=None, src_counts={}, kw={}, cset=set(),
        uset=set(), strong=[], resolver={}, metrics={}, digest="D",
    )
    base.update(overrides)
    return evidence.Evidence(**base)


class _StubXn:
    def __init__(self, raise_on=None, exc=None):
        self.calls = []
        self.raise_on = raise_on
        self.exc = exc

    def write_walk(self, bundle, pathways, *, prompt=None):
        self.calls.append(("write_walk", bundle, pathways, prompt))
        if self.raise_on == "write_walk":
            raise self.exc

    def write_digest(self, bundle, digest, touched, *, prompt=None):
        self.calls.append(("write_digest", bundle, digest, touched, prompt))
        if self.raise_on == "write_digest":
            raise self.exc


def test_digest_payload_none_in_none_out():
    assert evidence.digest_payload(None) is None


def test_digest_payload_returns_exactly_the_three_keys():
    ds = {"chunks": {"c": 1}, "sal": {"s": 2}, "kept": [1, 2], "cross": "extra"}
    out = evidence.digest_payload(ds)
    assert out == {"chunks": ds["chunks"], "sal": ds["sal"], "kept": ds["kept"]}
    assert out["chunks"] is ds["chunks"]
    assert "cross" not in out


def test_community_payload_shapes_six_keys_with_and_without_metrics():
    touched = [{"cid": 1, "size": 20, "hits": 5, "density": 0.03, "conductance": 0.8},
               {"cid": 2, "size": 9, "hits": 2}]
    kw = {1: ["jury", "trial"]}
    out = evidence.community_payload(touched, kw)
    assert len(out) == 2
    assert set(out[0]) == {"cid", "keywords", "size", "hits", "density", "conductance"}
    assert out[0] == {"cid": 1, "keywords": ["jury", "trial"], "size": 20,
                      "hits": 5, "density": 0.03, "conductance": 0.8}
    assert out[1] == {"cid": 2, "keywords": [], "size": 9, "hits": 2,
                      "density": None, "conductance": None}


def test_mirror_walk_success_calls_write_walk_before_write_digest():
    ev = _base_evidence(pathways={"pairs": []}, touched=[], kw={})
    ds = {"chunks": {}, "sal": {}, "kept": []}
    xn = _StubXn()
    err = evidence.mirror_walk("bundle", ev, ds, prompt="q", xn=xn)
    assert err is None
    names = [c[0] for c in xn.calls]
    assert names == ["write_walk", "write_digest"]
    assert xn.calls[0][3] == "q" and xn.calls[1][4] == "q"


def test_mirror_walk_with_ds_none_skips_write_digest():
    ev = _base_evidence()
    xn = _StubXn()
    err = evidence.mirror_walk("bundle", ev, None, prompt="q", xn=xn)
    assert err is None
    names = [c[0] for c in xn.calls]
    assert names == ["write_walk"]


@pytest.mark.parametrize("exc", [
    urllib.error.URLError("refused"),
    RuntimeError("neo4j tx error, deadlock"),
    OSError("timed out"),
])
def test_mirror_walk_catches_transport_and_server_errors(exc):
    ev = _base_evidence()
    xn = _StubXn(raise_on="write_walk", exc=exc)
    err = evidence.mirror_walk("bundle", ev, None, prompt="q", xn=xn)
    assert err == str(exc)


def test_mirror_walk_does_not_swallow_programming_errors():
    """The narrowing gate (6.21(c)): a TypeError in the payload path is a bug,
    not an unreachable server, and must surface as a traceback."""
    ev = _base_evidence()
    xn = _StubXn(raise_on="write_walk", exc=TypeError("payload bug"))
    with pytest.raises(TypeError):
        evidence.mirror_walk("bundle", ev, None, prompt="q", xn=xn)


# ---- analysis-view queries (T39, design 6.22 P5/P7) ----

def test_subgraph_edge_weights_collapses_direction_with_max(monkeypatch):
    rows = [
        {"src": 1, "dst": 1, "strength": 9.0, "provenance": "x"},  # self-loop, skipped
        {"src": 1, "dst": 2, "strength": 0.3, "provenance": "x"},
        {"src": 2, "dst": 1, "strength": 0.8, "provenance": "x"},  # asymmetric dup
    ]
    monkeypatch.setattr(evidence.gt, "subgraph_edges", lambda conn, run, ords: rows)
    out = evidence.subgraph_edge_weights(None, None, [1, 2])
    assert out == {(1, 2): 0.8}


@pytest.mark.live_db
def test_analysis_inputs_smoke():
    conn = require_gt_conn()
    try:
        run = require_run(conn, "mixed-full-dual")
        bnd, _ = sampler.ef_evidence(conn, run, Q_A, seed=0)
        assert bnd.sampled
        out = evidence.analysis_inputs(conn, run, bnd.sampled)
        assert set(out) == {"edges", "ents", "rels"}
        ents = out["ents"]
        walked_total = sum(sum(m.values()) for m in ents["mentions"].values())
        assert ents["corpus_total"] >= walked_total
        for eid in ents["names"]:
            assert any(eid in m for m in ents["mentions"].values())
        mentioned = {eid for m in ents["mentions"].values() for eid in m}
        for r in out["rels"]:
            assert r["src"] in mentioned
            assert r["dst"] in mentioned
    finally:
        conn.close()


# ---- class naming (T79, design 6.24 E15 amendment) ----

def test_pick_class_label_picks_the_planted_song_over_the_ubiquitous_later():
    # live receipt (spec 6.24 E15 amendment): mass order is later 23360 >
    # song 16796 > became 15965 > album 15124, so raw mass elects "later".
    # "later" is planted near-ubiquitous (high df) so distinctiveness demotes
    # it; "song" is comparatively rare (low df) so it survives the demotion.
    chunks = 40000
    candidates = [
        ("later", 23360, 38000),   # in nearly every chunk -> ln(~1.05) ~ 0
        ("song", 16796, 4000),     # far rarer -> ln(10) ~ 2.3
        ("became", 15965, 30000),
        ("album", 15124, 6000),
    ]
    assert evidence.pick_class_label(candidates, chunks) == "song"


def test_pick_class_label_breaks_ties_by_mass_then_name():
    # identical score (same mass, same df -> same distinctiveness): mass tie
    # too, so alphabetically-first name wins deterministically.
    chunks = 100
    candidates = [("zeta", 10, 5), ("alpha", 10, 5)]
    assert evidence.pick_class_label(candidates, chunks) == "alpha"

    # unequal mass, same df: higher mass wins even though score ties are not
    # in play here (score scales with mass directly).
    candidates2 = [("low", 5, 5), ("high", 10, 5)]
    assert evidence.pick_class_label(candidates2, chunks) == "high"


def test_class_label_score_demotes_high_df_toward_zero():
    # a term touching every chunk (df == chunks) scores near zero regardless
    # of mass -- the PPMI-as-demotion-filter law (memory: PPMI is a demotion
    # filter, not a term weight).
    ubiquitous = evidence.class_label_score(mass=50000, df=10000, chunks=10000)
    rare = evidence.class_label_score(mass=50000, df=10, chunks=10000)
    assert ubiquitous == pytest.approx(0.0, abs=1e-9)
    assert rare > ubiquitous


@pytest.mark.live_db
def test_class_labels_covers_every_class_and_changes_at_least_one_label():
    conn = require_gt_conn()
    try:
        run = require_run(conn, "mixed-full-dual")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT class_id FROM entities"
                " WHERE run_id = %s AND class_id IS NOT NULL",
                (run.run_id,))
            class_ids = {r["class_id"] for r in cur.fetchall()}
            cur.execute(
                "SELECT entity_id, name FROM entities"
                " WHERE run_id = %s AND entity_id = ANY(%s::int[])",
                (run.run_id, sorted(class_ids)))
            min_id_name = {r["entity_id"]: r["name"] for r in cur.fetchall()}

        labels = evidence.class_labels(conn, run)

        # every class_id the run knows about gets a label
        assert class_ids
        assert set(labels) == class_ids
        for cid in class_ids:
            assert labels[cid]

        # the rule changed something: at least one class's argmax label
        # differs from the old min(members) representative's own name
        assert any(labels[cid] != min_id_name[cid] for cid in class_ids)
    finally:
        conn.close()


@pytest.mark.live_db
def test_walk_entities_class_names_use_class_labels_not_raw_min_id_name():
    conn = require_gt_conn()
    try:
        run = require_run(conn, "mixed-full-dual")
        bnd, _ = sampler.ef_evidence(conn, run, Q_A, seed=0)
        assert bnd.sampled
        ents = evidence.walk_entities(conn, run, bnd.sampled)
        labels = evidence.class_labels(conn, run)
        for cid, name in ents["class_names"].items():
            assert name == labels.get(cid, name)
    finally:
        conn.close()


@pytest.mark.live_db
def test_class_reference_label_matches_class_labels():
    conn = require_gt_conn()
    try:
        run = require_run(conn, "mixed-full-dual")
        labels = evidence.class_labels(conn, run)
        ref = evidence.class_reference(conn, run)
        for c in ref["entity_classes"]:
            assert c["label"] == labels.get(c["class_id"], c["label"])
    finally:
        conn.close()
