"""tests/test_walker_core.py -- the walker's UI-free logic, without Streamlit.

Spec: .spec/specs/graph-explorer/design.md 6.21(c)
Task: playbook.md T36
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

import walker_core
from evidence import Evidence


def _ev(**overrides):
    base = dict(
        touched=[], cids=[], concept={}, terms={}, cid_of={}, in_cid={},
        xedges={}, medoids={}, salient={}, term_members={}, pa_anchors=[],
        pathways={"pairs": []}, dendrite=None, src_counts={}, kw={}, cset=set(),
        uset=set(), strong=[], resolver={}, metrics={}, digest="D",
    )
    base.update(overrides)
    return Evidence(**base)


# --------------------------------------------------------------- import proof

def test_import_leaves_no_connection():
    """The load-bearing proof: a FRESH interpreter imports walker_core with
    the database down and exits 0 with empty stderr."""
    result = subprocess.run(
        [sys.executable, "-c", "import walker_core"],
        cwd=".", capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


# --------------------------------------------------------------------- clip

def test_clip_short_text_returned_identical():
    assert walker_core.clip("hello world", 100) == "hello world"


def test_clip_long_text_cuts_at_last_space():
    text = "the quick brown fox jumps over the lazy dog"
    out = walker_core.clip(text, 15)
    assert out.endswith(" …")
    assert out[:-2] == text[:15].rsplit(None, 1)[0]
    assert out[:-2] == "the quick"      # cut before "brow", not inside it


def test_clip_single_long_token_hard_cut_no_crash():
    text = "a" * 50
    out = walker_core.clip(text, 10)
    assert out == "a" * 10 + " …"


def test_clip_none_and_empty_return_empty_string():
    assert walker_core.clip(None, 10) == ""
    assert walker_core.clip("", 10) == ""


# --------------------------------------------------------------- load_labels

def test_load_labels_missing_path_returns_empty(tmp_path):
    assert walker_core.load_labels(tmp_path / "nope.json", "run-1") == {}


def test_load_labels_unparseable_json_returns_empty(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text("{not json", encoding="utf-8")
    assert walker_core.load_labels(f, "run-1") == {}


def test_load_labels_foreign_run_id_returns_stale_marker(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"run_id": "other-run",
                             "communities": [{"cid": 7, "label": "Elections"}]}),
                 encoding="utf-8")
    out = walker_core.load_labels(f, "this-run")
    assert out == {"__stale_run__": "other-run"}
    assert 7 not in out


def test_load_labels_matching_run_id_returns_labels(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"run_id": "run-1",
                             "communities": [{"cid": 7, "label": "Elections"}]}),
                 encoding="utf-8")
    out = walker_core.load_labels(f, "run-1")
    assert out == {7: {"cid": 7, "label": "Elections"}}


def test_load_labels_community_with_no_label_key_dropped(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"run_id": "run-1",
                             "communities": [{"cid": 7, "label": "Elections"},
                                             {"cid": 8}]}),
                 encoding="utf-8")
    out = walker_core.load_labels(f, "run-1")
    assert set(out) == {7}


# ---------------------------------------------------------------- src_of_map

def test_src_of_map_one_entry_per_ord_none_preserved(monkeypatch):
    monkeypatch.setattr(walker_core.gt, "node", lambda conn, run, o: {"ord": o})
    monkeypatch.setattr(walker_core.gt, "source_of",
                        lambda nd: "wiki" if nd["ord"] == 1 else None)
    out = walker_core.src_of_map(None, None, [1, 2])
    assert out == {1: "wiki", 2: None}


# -------------------------------------------------------------- partition_rows

def test_partition_rows_shapes_chains_sources_and_terms():
    ds = {
        "chunks": {"chains": [[1, 2], [3]]},
        "sal": {
            1: {"top": ["carrier", "midway"]},
            2: {"top": ["carrier"]},
            3: {"top": ["army"]},
        },
    }
    src_of = {1: "wiki", 2: "wiki", 3: None}   # 3 -> None -> "?" via .get(o) or "?"
    rows = walker_core.partition_rows(ds, src_of)
    assert [r["chain"] for r in rows] == [1, 2]
    assert rows[0]["chunks"] == 2
    assert rows[0]["sources"] == "wiki:2"
    assert "carrier(2)" in rows[0]["salient terms (carried by N members)"]
    assert "midway" in rows[0]["salient terms (carried by N members)"]
    assert "midway(2)" not in rows[0]["salient terms (carried by N members)"]
    assert rows[1]["sources"] == "?:1"          # ord 3 -> None -> "?"


def test_partition_rows_missing_ord_shows_question_mark_and_caps_at_12():
    sal = {o: {"top": [f"t{o}"]} for o in range(1, 15)}
    ds = {"chunks": {"chains": [list(range(1, 15))]}, "sal": sal}
    src_of = {}   # every ord missing -> "?"
    rows = walker_core.partition_rows(ds, src_of)
    assert rows[0]["sources"] == "?:14"
    terms = rows[0]["salient terms (carried by N members)"].split(", ")
    assert len(terms) == 12


# -------------------------------------------------------------------- assess

def test_assess_mirror_clean_returns_rr_and_none(monkeypatch):
    ev = _ev(terms={"a": 1}, concept={"b": 2}, pathways={"pairs": []}, digest="D")
    rr = {"ok": True}
    calls = {}

    def fake_reason(conn, run, bundle, terms, concept, *, embed=None, judge=None,
                    pw=None, digest=None):
        calls["args"] = (terms, concept, judge, pw, digest)
        return rr

    def fake_mirror(bundle, ev_, ds_, *, prompt):
        calls["mirror"] = True
        return None

    monkeypatch.setattr(walker_core.interpret, "reason", fake_reason)
    monkeypatch.setattr(walker_core.evidence, "mirror_walk", fake_mirror)
    out_rr, err = walker_core.assess(None, None, "bundle", ev, {"ds": 1}, "q")
    assert out_rr is rr
    assert err is None
    assert calls["args"] == (ev.terms, ev.concept, True, ev.pathways, ev.digest)
    assert calls["mirror"] is True


def test_assess_mirror_returns_error_string(monkeypatch):
    ev = _ev()
    monkeypatch.setattr(walker_core.interpret, "reason",
                        lambda *a, **k: {"ok": True})
    monkeypatch.setattr(walker_core.evidence, "mirror_walk",
                        lambda *a, **k: "connection refused")
    _, err = walker_core.assess(None, None, "bundle", ev, None, "q")
    assert err == "connection refused"


def test_assess_neo4j_mirror_env_zero_skips_mirror(monkeypatch):
    ev = _ev()
    monkeypatch.setenv("NEO4J_MIRROR", "0")
    monkeypatch.setattr(walker_core.interpret, "reason",
                        lambda *a, **k: {"ok": True})
    called = []
    monkeypatch.setattr(walker_core.evidence, "mirror_walk",
                        lambda *a, **k: called.append(1))
    _, err = walker_core.assess(None, None, "bundle", ev, None, "q")
    assert err is None
    assert called == []


# ---------- analysis view (T39, design 6.22)

# ------------------------------------------------------------- node_dwpc

def test_node_dwpc_sums_pairs_whose_best_path_contains_chunk():
    pathways = {"pairs": [
        {"a": 1, "b": 3, "dwpc": 2.0, "path": [1, 2, 3]},
        {"a": 1, "b": 4, "dwpc": 0.5, "path": [1, 4]},
    ]}
    out = walker_core.node_dwpc(pathways)
    assert out == {1: 2.5, 2: 2.0, 3: 2.0, 4: 0.5}


def test_node_dwpc_empty_and_missing_path_keys():
    assert walker_core.node_dwpc({"pairs": []}) == {}
    assert walker_core.node_dwpc({"pairs": [{"a": 1, "b": 2, "dwpc": 1.0}]}) == {}


# --------------------------------------------------------- rank_group_terms

def test_rank_group_terms_orders_by_dwpc_mass():
    members = [1, 2, 3]
    sal = {
        1: {"kept": ["carrier", "navy"]},
        2: {"kept": ["carrier"]},
        3: {"kept": ["army"]},
    }
    ndw = {1: 2.0, 2: 1.0, 3: 0.5}
    rows = walker_core.rank_group_terms(members, sal, ndw)
    terms = [r["term"] for r in rows]
    # carrier: 2.0 + 1.0 = 3.0 ; navy: 2.0 ; army: 0.5
    assert terms == ["carrier", "navy", "army"]
    assert rows[0]["dwpc"] == 3.0
    assert rows[1]["dwpc"] == 2.0
    assert rows[2]["dwpc"] == 0.5
    assert rows[0]["chunks"] == [1, 2]


def test_rank_group_terms_zero_dwpc_terms_rank_below_positive_keeping_bm25_order():
    members = [1, 2]
    sal = {
        1: {"kept": ["carrier"]},          # on-path chunk
        2: {"kept": ["zeta", "alpha"]},    # off-path chunk, zero dwpc
    }
    ndw = {1: 5.0}                          # chunk 2 absent -> 0.0 contribution
    rows = walker_core.rank_group_terms(members, sal, ndw, k=None)
    terms = [r["term"] for r in rows]
    assert terms[0] == "carrier"
    # both zero-dwpc terms rank after carrier, keeping their kept-list order
    assert terms[1:] == ["zeta", "alpha"]
    assert rows[1]["dwpc"] == 0.0 and rows[2]["dwpc"] == 0.0
    assert rows[1]["bm25"] > rows[2]["bm25"]


def test_rank_group_terms_dwpc_tie_breaks_on_bm25_then_lexicographic():
    members = [1, 2]
    # "hi" ranked first (bm25 higher) in chunk 1's kept list; "lo" ranked
    # second. Both appear only in chunk 1, so tie on dwpc, break on bm25.
    sal = {1: {"kept": ["hi", "lo"]}}
    ndw = {1: 1.0}
    rows = walker_core.rank_group_terms(members, sal, ndw, k=None)
    assert [r["term"] for r in rows] == ["hi", "lo"]
    assert rows[0]["dwpc"] == rows[1]["dwpc"] == 1.0
    assert rows[0]["bm25"] > rows[1]["bm25"]

    # identical dwpc AND identical bm25 rank -> alphabetical
    sal2 = {1: {"kept": ["zeta"]}, 2: {"kept": ["alpha"]}}
    ndw2 = {1: 1.0, 2: 1.0}
    rows2 = walker_core.rank_group_terms([1, 2], sal2, ndw2, k=None)
    assert [r["term"] for r in rows2] == ["alpha", "zeta"]


# -------------------------------------------------------- subgraph_louvain

def test_subgraph_louvain_two_cliques_split():
    ords = [1, 2, 3, 4, 5, 6]
    edges = {
        (1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0,
        (4, 5): 1.0, (4, 6): 1.0, (5, 6): 1.0,
        (3, 4): 0.01,
    }
    out = walker_core.subgraph_louvain(ords, edges)
    assert set(out) == set(ords)
    groups = {}
    for o, g in out.items():
        groups.setdefault(g, set()).add(o)
    assert len(groups) == 2
    assert {1, 2, 3} in groups.values()
    assert {4, 5, 6} in groups.values()


def test_subgraph_louvain_is_deterministic_across_calls():
    ords = [1, 2, 3, 4, 5, 6]
    edges = {
        (1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0,
        (4, 5): 1.0, (4, 6): 1.0, (5, 6): 1.0,
        (3, 4): 0.01,
    }
    out1 = walker_core.subgraph_louvain(ords, edges)
    out2 = walker_core.subgraph_louvain(ords, edges)
    assert out1 == out2


def test_subgraph_louvain_isolated_and_empty_inputs():
    out = walker_core.subgraph_louvain([1, 2, 3], {})
    assert out == {1: 0, 2: 1, 3: 2}
    assert walker_core.subgraph_louvain([], {}) == {}


# --------------------------------------------------------- group_entities

def test_group_entities_lift_prefers_locally_concentrated_entity():
    members = [1, 2]
    ents = {
        "mentions": {1: {"A": 5, "B": 5}},
        "names": {"A": "Alpha", "B": "Beta"},
        "corpus": {"A": 6, "B": 500},
        "corpus_total": 1000,
    }
    # A: group 5/10 = 0.5 share, corpus 6/1000 = 0.006 -> lift ~83.3
    # B: group 5/10 = 0.5 share, corpus 500/1000 = 0.5 -> lift 1.0
    rows = walker_core.group_entities(members, ents, floor=2)
    assert rows[0]["entity_id"] == "A"
    assert rows[0]["lift"] == pytest.approx(0.5 / 0.006)


def test_group_entities_mention_floor_drops_singletons():
    members = [1]
    ents = {
        "mentions": {1: {"A": 1, "B": 4}},
        "names": {"A": "Alpha", "B": "Beta"},
        "corpus": {"A": 1, "B": 100},
        "corpus_total": 1000,
    }
    rows = walker_core.group_entities(members, ents, floor=2)
    ids = [r["entity_id"] for r in rows]
    assert "A" not in ids
    assert "B" in ids


# -------------------------------------------------------- group_relations

def test_group_relations_requires_both_endpoints_in_group():
    members = [1, 2]
    ents = {"mentions": {1: {"A": 1, "B": 1}, 2: {}}, "names": {}, "corpus": {},
            "corpus_total": 0}
    rels = [
        {"src": "A", "dst": "B", "template": "t1", "connector": "of", "n": 5,
         "llr": 3.0, "npmi": 0.1, "example_ord": 1,
         "src_name": "A", "dst_name": "B"},
        {"src": "A", "dst": "C", "template": "t2", "connector": "of", "n": 2,
         "llr": 1.0, "npmi": 0.1, "example_ord": 1,
         "src_name": "A", "dst_name": "C"},   # C not mentioned in group
    ]
    rows = walker_core.group_relations(members, ents, rels)
    keys = [(r["template"], r["connector"]) for r in rows]
    assert ("t1", "of") in keys
    assert ("t2", "of") not in keys


# ---------------------------------------------------------- group_classes

def test_group_classes_shapes_both_panels_from_the_same_function():
    edges = {(1, 2): 1.0}
    groups_rel = walker_core.subgraph_louvain([1, 2, 3], edges)
    groups_global = {1: 0, 2: 0, 3: 1}   # a hand-written global cid_of
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    rels: list = []
    sal: dict = {}
    ndw: dict = {}

    rel_rows = walker_core.group_classes(groups_rel, ents, rels, sal, ndw)
    global_rows = walker_core.group_classes(groups_global, ents, rels, sal, ndw)

    assert {r["gid"] for r in rel_rows} == set(groups_rel.values())
    assert {r["gid"] for r in global_rows} == set(groups_global.values())
    for rows in (rel_rows, global_rows):
        sizes = [r["size"] for r in rows]
        assert sizes == sorted(sizes, reverse=True)


def test_group_classes_survives_missing_dendrite_sal():
    groups = {1: 0, 2: 0, 3: 1}
    ents = {"mentions": {1: {"A": 3}, 2: {"A": 3}, 3: {"B": 5}},
            "names": {"A": "Alpha", "B": "Beta"},
            "corpus": {"A": 6, "B": 5}, "corpus_total": 20}
    rels = [{"src": "A", "dst": "B", "template": "t", "connector": "of", "n": 1,
             "llr": 1.0, "npmi": 0.1, "example_ord": 1,
             "src_name": "A", "dst_name": "B"}]
    rows = walker_core.group_classes(groups, ents, rels, sal={}, ndw={})
    for r in rows:
        assert r["terms"] == []
    assert any(r["entities"] for r in rows)
