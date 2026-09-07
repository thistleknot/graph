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


# ---------------------------------------------------- factbook digests (T43)

def _gc(**over):
    base = dict(gid=0, members=[], size=0, terms=[], entities=[], relations=[])
    base.update(over)
    return base


def test_group_digest_header_carries_gid_size_and_source_mix():
    gc = _gc(gid=0, members=[1, 2, 3], size=3)
    src_of = {1: "wiki", 2: "wiki", 3: "brown"}
    out = walker_core.group_digest(gc, {"mentions": {}, "names": {}, "corpus": {},
                                         "corpus_total": 0}, src_of)
    assert out.splitlines()[0] == "g0 . 3 chunks . wiki:2 brown:1"


def test_group_digest_terms_show_one_decimal_in_dwpc_order():
    gc = _gc(terms=[{"term": "carrier", "dwpc": 3.0, "bm25": 0.0, "chunks": []},
                     {"term": "navy", "dwpc": 2.0, "bm25": 0.0, "chunks": []},
                     {"term": "army", "dwpc": 0.5, "bm25": 0.0, "chunks": []}])
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    out = walker_core.group_digest(gc, ents)
    line = out.splitlines()[1]
    assert line.startswith("terms(dwpc): ")
    assert "carrier 3.0" in line
    assert "3.0" in line
    assert line.index("carrier") < line.index("navy") < line.index("army")


def test_group_digest_lift_suppressed_for_walk_exclusive_entity():
    gc = _gc(entities=[
        {"entity_id": "A", "name": "Alpha", "cnt": 8, "lift": 1.0,
         "group_share": 1.0, "corpus_share": 1.0},
        {"entity_id": "B", "name": "Beta", "cnt": 5, "lift": 3.0,
         "group_share": 0.1, "corpus_share": 0.3},
    ])
    ents = {"mentions": {}, "names": {"A": "Alpha", "B": "Beta"},
            "corpus": {"A": 8, "B": 50}, "corpus_total": 100}
    out = walker_core.group_digest(gc, ents)
    line = out.splitlines()[2]
    assert "Alpha 8" in line
    assert "Alpha 8 x" not in line
    assert "Beta 5 x" in line


def test_group_digest_entities_ranked_by_mention_count_not_lift():
    entities = [
        {"entity_id": "B", "name": "Beta", "cnt": 2, "lift": 9.0,
         "group_share": 0.1, "corpus_share": 0.01},
        {"entity_id": "A", "name": "Alpha", "cnt": 9, "lift": 1.1,
         "group_share": 0.9, "corpus_share": 0.8},
    ]
    gc = _gc(entities=entities)
    ents = {"mentions": {}, "names": {"A": "Alpha", "B": "Beta"},
            "corpus": {"A": 9, "B": 2}, "corpus_total": 20}
    out = walker_core.group_digest(gc, ents)
    line = out.splitlines()[2]
    assert line.index("Alpha") < line.index("Beta")
    assert entities[0]["entity_id"] == "B"   # gc["entities"] left unmutated


def test_group_digest_relations_cap_and_corpus_n_label():
    relations = [{"template": f"t{i}", "connector": "of", "n": i, "pairs": 1,
                  "top": [(f"a{i}", f"b{i}", 1.0)]} for i in range(10)]
    gc = _gc(relations=relations)
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    out = walker_core.group_digest(gc, ents, width=300, max_pairs=8)
    line = out.splitlines()[3]
    assert line.count("corpus_n=") == 8
    assert line.endswith("+ 2 more")


def test_group_digest_truncates_long_tail_with_count():
    terms = [{"term": f"term{i}", "dwpc": float(i), "bm25": 0.0, "chunks": []}
              for i in range(60)]
    gc = _gc(terms=terms)
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    out = walker_core.group_digest(gc, ents, width=110)
    for line in out.splitlines():
        assert len(line) <= 110
    terms_line = out.splitlines()[1]
    assert "+ " in terms_line and terms_line.endswith("more")
    sep = walker_core._SEP
    body = terms_line[len("terms(dwpc): "):]
    parts = body.split(sep)
    n_more = int(parts[-1].split()[1])
    rendered_count = len(parts) - 1   # all parts but the "+ N more" tail
    assert n_more + rendered_count == 60


def test_group_digest_marker_lands_in_header():
    gc = _gc(gid=6)
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    out = walker_core.group_digest(gc, ents, marker="= c6 (global)")
    assert out.splitlines()[0].endswith("= c6 (global)")


def test_group_digest_empty_group_renders_dashes():
    gc = _gc()
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    out = walker_core.group_digest(gc, ents)
    lines = out.splitlines()
    assert lines[1] == "terms(dwpc): -"
    assert lines[2] == "entities(mentions): -"
    assert lines[3] == "relations: -"


# ---- classes (T72, design 6.24 E15-E19)

def test_group_counts_same_fn_serves_all_three_groupings():
    # 6 ords, 4 entities; A/B/C/D mentioned across ords 1..6
    ents = {"mentions": {
        1: {"A": 2}, 2: {"A": 1, "B": 3}, 3: {"B": 1}, 4: {"C": 4},
        5: {"C": 1, "D": 2}, 6: {"D": 1},
    }}
    rels = [
        {"src": "A", "dst": "B", "template": "t1"},
        {"src": "A", "dst": "B", "template": "t2"},
        {"src": "C", "dst": "D", "template": "t3"},
        {"src": "B", "dst": "Z", "template": "t4"},   # dst outside the walk
    ]
    # global cids: {1,2,3} -> cid 0, {4,5,6} -> cid 1
    global_groups = {0: [1, 2, 3], 1: [4, 5, 6]}
    # relative louvain output, same member sets, different gids
    rel_groups = {5: [1, 2, 3], 9: [4, 5, 6]}
    # dendrite chains, same member sets, 1-based chain numbers
    chains = {1: [1, 2, 3], 2: [4, 5, 6]}

    g_counts = walker_core.group_counts(global_groups, ents, rels)
    r_counts = walker_core.group_counts(rel_groups, ents, rels)
    c_counts = walker_core.group_counts(chains, ents, rels)

    assert set(g_counts) == {0, 1}
    assert set(r_counts) == {5, 9}
    assert set(c_counts) == {1, 2}

    # commensurability: identical member sets -> identical numbers regardless
    # of which grouping/gid produced them.
    assert g_counts[0] == r_counts[5] == c_counts[1]
    assert g_counts[1] == r_counts[9] == c_counts[2]
    assert g_counts[0] == {"chunks": 3, "entities": 2, "relations": 2}
    assert g_counts[1] == {"chunks": 3, "entities": 2, "relations": 1}


def test_group_counts_relations_need_both_endpoints_and_count_rows():
    ents = {"mentions": {1: {"A": 1, "B": 1}}}
    rels = [
        {"src": "A", "dst": "B", "template": "t1"},
        {"src": "A", "dst": "B", "template": "t2"},   # same pair, 2nd template
        {"src": "A", "dst": "Z", "template": "t3"},   # dst unmentioned
    ]
    out = walker_core.group_counts({0: [1]}, ents, rels)
    assert out[0]["relations"] == 2


def test_group_counts_empty_group_is_three_zeros():
    out = walker_core.group_counts({0: []}, {"mentions": {}}, [])
    assert out[0] == {"chunks": 0, "entities": 0, "relations": 0}


def test_group_classes_rows_carry_counts_from_group_counts():
    groups = {1: 0, 2: 0, 3: 1}
    ents = {"mentions": {1: {"A": 3}, 2: {"A": 3}, 3: {"B": 5}},
            "names": {"A": "Alpha", "B": "Beta"},
            "corpus": {"A": 6, "B": 5}, "corpus_total": 20}
    rels = [{"src": "A", "dst": "B", "template": "t", "connector": "of", "n": 1,
             "llr": 1.0, "npmi": 0.1, "example_ord": 1,
             "src_name": "A", "dst_name": "B"}]
    rows = walker_core.group_classes(groups, ents, rels, sal={}, ndw={})
    # rebuild the same by_group shape group_classes uses internally
    bg = {}
    for o, gid in groups.items():
        bg.setdefault(gid, []).append(o)
    expected = walker_core.group_counts(bg, ents, rels)
    for row in rows:
        assert row["counts"] == expected[row["gid"]]


def test_group_digest_header_carries_entity_and_relation_counts():
    gc = _gc(gid=0, members=[1, 2, 3], size=3, counts={"chunks": 3, "entities": 214, "relations": 87})
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    out = walker_core.group_digest(gc, ents)
    assert out.splitlines()[0] == "g0 . 3 chunks . 214 entities . 87 relations"


def test_group_digest_header_unchanged_without_counts():
    gc = _gc(gid=0, members=[1, 2, 3], size=3)
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    out = walker_core.group_digest(gc, ents)
    assert out.splitlines()[0] == "g0 . 3 chunks"


def test_group_digest_entity_line_annotates_non_singleton_class():
    entities = [
        {"entity_id": "A", "name": "Alpha", "cnt": 8, "lift": 1.0,
         "group_share": 1.0, "corpus_share": 1.0},
        {"entity_id": "B", "name": "Beta", "cnt": 5, "lift": 3.0,
         "group_share": 0.1, "corpus_share": 0.3},
    ]
    gc = _gc(entities=entities)
    ents = {"mentions": {}, "names": {"A": "Alpha", "B": "Beta"},
            "corpus": {"A": 8, "B": 50}, "corpus_total": 100,
            "class_of": {"A": "A", "B": 99}, "class_names": {99: "UnitedStates"}}
    out = walker_core.group_digest(gc, ents)
    line = out.splitlines()[2]
    assert line.count("~") == 1
    assert "Beta 5 x3.0 ~UnitedStates" in line
    assert "Alpha 8 ~" not in line   # A's class_id == entity_id -> singleton


def test_group_digest_relations_lead_with_walk_local_pairs():
    import re
    relations = [{"template": "of", "connector": "of", "n": 527581, "pairs": 3,
                  "top": [("second", "world_war", 5.0)]}]
    gc = _gc(relations=relations)
    ents = {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0}
    out = walker_core.group_digest(gc, ents, width=300)
    line = out.splitlines()[3]
    m = re.search(r"pairs=(\d+) corpus_n=(\d+)", line)
    assert m is not None
    assert m.group(1) == "3" and m.group(2) == "527581"
    assert line.index("pairs=") < line.index("corpus_n=")


def test_class_digest_header_and_member_lines():
    cref = {
        "entity_classes": [
            {"class_id": 12, "label": "united_states", "mass": 12045, "members": 34,
             "top": [("united_states", 8801), ("u_s", 2140)]},
            {"class_id": 5, "label": "france", "mass": 900, "members": 2,
             "top": [("france", 800), ("french_republic", 100)]},
        ],
        "relation_classes": [
            {"rel_class": "gen", "mass": 51204, "templates": 6,
             "top": [("of", 21003), ("of_the", 18800)]},
            {"rel_class": "loc", "mass": 300, "templates": 2,
             "top": [("in", 200), ("at", 100)]},
        ],
    }
    out = walker_core.class_digest(cref)
    lines = out.splitlines()
    assert lines[0] == "classes . 2 entity . 2 relation"
    assert lines[1].startswith("E12 united_states . 34 members . mass 12045: ")
    assert "united_states 8801" in lines[1]
    assert lines[3].startswith("R gen . 6 templates . n 51204: ")
    assert "of 21003" in lines[3]


def test_class_digest_truncates_with_more_count():
    top = [(f"member{i}", 100 - i) for i in range(40)]
    cref = {"entity_classes": [{"class_id": 1, "label": "member0", "mass": 1000,
                                "members": 40, "top": top}],
            "relation_classes": []}
    out = walker_core.class_digest(cref, width=110)
    ent_line = out.splitlines()[1]
    assert len(ent_line) <= 110
    assert "+ " in ent_line and ent_line.endswith("more")


def test_class_digest_empty_reference_renders_dashes():
    out = walker_core.class_digest({"entity_classes": [], "relation_classes": []})
    lines = out.splitlines()
    assert lines[0] == "classes . 0 entity . 0 relation"
    assert lines[1] == "-"
    assert lines[2] == "-"
    assert out != ""


def test_class_digest_card_is_byte_identical_and_escapes():
    import html as _html
    import re
    cref = {"entity_classes": [{"class_id": 1, "label": "<b>&evil</b>", "mass": 10,
                                "members": 2, "top": [("<b>&evil</b>", 5)]}],
            "relation_classes": []}
    txt = walker_core.class_digest(cref)
    card = walker_core.digest_card(txt, "#4C78A8")
    assert "<script>" not in card
    body = re.search(r"<pre[^>]*>(.*)</pre>", card, re.S).group(1)
    assert _html.unescape(body.replace("<br>", "\n")) == txt


def test_partition_rows_counts_columns_optional():
    ds = {"chunks": {"chains": [[1, 2]]}, "sal": {1: {"top": ["a"]}, 2: {"top": []}}}
    src_of = {1: "wiki", 2: "wiki"}
    rows_plain = walker_core.partition_rows(ds, src_of)
    assert set(rows_plain[0]) == {"chain", "chunks", "sources",
                                  "salient terms (carried by N members)"}
    counts = {1: {"chunks": 2, "entities": 5, "relations": 3}}
    rows_counted = walker_core.partition_rows(ds, src_of, counts=counts)
    assert rows_counted[0]["entities"] == 5
    assert rows_counted[0]["relations"] == 3


def test_dedup_groups_equal_member_sets_render_one_row():
    rel_rows = [_gc(gid=0, members=[1, 2, 3], size=3)]
    glob_rows = [_gc(gid=6, members=[1, 2, 3], size=3)]
    out = walker_core.dedup_groups(rel_rows, glob_rows)
    assert len(out) == 1
    assert out[0]["kind"] == "merged"
    assert out[0]["marker"] == "= c6 (global)"
    assert not any(e["prefix"] == "c" for e in out)


def test_dedup_groups_split_decomposition_names_counts_and_split_cid():
    c0_members = list(range(40))
    g1_members = c0_members[:28] + list(range(100, 108))
    rel_rows = [_gc(gid=1, members=g1_members, size=len(g1_members))]
    glob_rows = [_gc(gid=0, members=c0_members, size=len(c0_members)),
                 _gc(gid=12, members=list(range(100, 108)), size=8)]
    out = walker_core.dedup_groups(rel_rows, glob_rows)
    row = next(e for e in out if e["kind"] in ("local", "merged") and e["gid"] == 1)
    assert row["marker"] == "= c0:28 + c12:8 (splits c0)"


def test_dedup_groups_merge_decomposition_omits_counts():
    c3_members = list(range(20))
    c7_members = list(range(20, 35))
    rel_rows = [_gc(gid=2, members=c3_members + c7_members,
                    size=len(c3_members) + len(c7_members))]
    glob_rows = [_gc(gid=3, members=c3_members, size=len(c3_members)),
                 _gc(gid=7, members=c7_members, size=len(c7_members))]
    out = walker_core.dedup_groups(rel_rows, glob_rows)
    row = next(e for e in out if e["gid"] == 2)
    assert row["marker"] == "= c3+c7 (merges)"


def test_dedup_groups_unconsumed_global_groups_render_with_c_prefix():
    rel_rows = [_gc(gid=0, members=[1, 2], size=2)]
    glob_rows = [_gc(gid=0, members=[1, 2], size=2),
                 _gc(gid=9, members=[500, 501], size=2)]
    out = walker_core.dedup_groups(rel_rows, glob_rows)
    unconsumed = [e for e in out if e["prefix"] == "c" and e["gid"] == 9]
    assert len(unconsumed) == 1
    assert unconsumed[0]["kind"] == "global"
    assert unconsumed[0]["marker"] is None


def test_dedup_groups_output_is_largest_first_and_deterministic():
    rel_rows = [_gc(gid=0, members=list(range(5)), size=5),
                _gc(gid=1, members=list(range(5, 8)), size=3)]
    glob_rows = [_gc(gid=0, members=list(range(20, 30)), size=10)]
    out1 = walker_core.dedup_groups(rel_rows, glob_rows)
    out2 = walker_core.dedup_groups(rel_rows, glob_rows)
    sizes = [e["size"] for e in out1]
    assert sizes == sorted(sizes, reverse=True)
    assert out1 == out2


def test_chain_communities_counts_descending_with_unknown_bucket():
    chains = [[1, 2, 3, 4], [5]]
    cid_of = {1: 6, 2: 6, 3: 0}
    out = walker_core.chain_communities(chains, cid_of)
    assert out == ["c6:2 c0:1 c?:1", "c?:1"]


def test_chain_communities_empty_chain_and_truncation():
    assert walker_core.chain_communities([[]], {}) == ["-"]
    chain = list(range(200))
    cid_of = {o: o for o in chain}
    out = walker_core.chain_communities([chain], cid_of, width=110)
    assert len(out[0]) <= 110
    assert "+ " in out[0] and out[0].endswith("more")


def test_digest_card_text_is_byte_identical_after_unescaping():
    import re
    import html as _html
    gc = _gc(gid=0, members=[1, 2, 3], size=3)
    src_of = {1: "wiki", 2: "wiki", 3: "brown"}
    txt = walker_core.group_digest(gc, {"mentions": {}, "names": {}, "corpus": {},
                                         "corpus_total": 0}, src_of)
    card = walker_core.digest_card(txt, "#4C78A8")
    body = re.search(r"<pre[^>]*>(.*)</pre>", card, re.S).group(1)
    # <br> is the rendered line break (streamlit re-flows raw \n even in <pre>);
    # folding it back must restore the digest byte for byte (P11(g)).
    assert _html.unescape(body.replace("<br>", "\n")) == txt
    assert "\n" in txt and "<br>" in body  # multi-line digests really carry breaks


def test_digest_card_escapes_html_in_the_body():
    import html as _html
    txt = '<script>alert("x")</script> & <b>'
    card = walker_core.digest_card(txt, "#4C78A8")
    assert "<script>" not in card
    assert "&lt;script&gt;" in card
    assert "&amp;" in card
    import re
    body = re.search(r"<pre[^>]*>(.*)</pre>", card, re.S).group(1)
    assert _html.unescape(body.replace("<br>", "\n")) == txt


def test_digest_card_carries_the_hue_as_border_and_tint():
    card = walker_core.digest_card("g0 . 1 chunks", "#4C78A8")
    assert "border-left:4px solid #4C78A8" in card
    assert walker_core.rgba("#4C78A8", 0.10) in card


def test_card_color_local_group_takes_dominant_cid_hue():
    assert walker_core.card_color(
        {"kind": "merged", "gid": 0, "cids": [(6, 36)]}) == walker_core.cid_color(6)
    assert walker_core.card_color(
        {"kind": "local", "gid": 1, "cids": [(0, 28), (12, 8)]}) == walker_core.cid_color(0)
    assert walker_core.card_color(
        {"kind": "global", "gid": 6, "cids": []}) == walker_core.cid_color(6)


def test_card_color_local_group_with_no_overlap_falls_back_to_gid():
    assert walker_core.card_color(
        {"kind": "local", "gid": 3, "cids": []}) == walker_core.cid_color(3)


def test_rgba_and_cid_color_moved_into_core():
    assert walker_core.cid_color(None) == "#DDDDDD"
    assert walker_core.rgba("#4C78A8", 0.1) == "rgba(76,120,168,0.1)"
    assert walker_core.cid_color(0) == walker_core.cid_color(len(walker_core.PALETTE))


# ---------- P15 dark dashboard skin (T50) ----------


def test_streamlit_config_sets_the_dark_theme():
    """File-content pin, deliberately not an AppTest -- AppTest never applies
    [theme] (it renders an element tree, not a DOM)."""
    try:
        import tomllib
    except ModuleNotFoundError:                            # py < 3.11
        import tomli as tomllib
    from pathlib import Path

    cfg = tomllib.loads(
        Path(__file__).resolve().parent.parent.joinpath(
            ".streamlit", "config.toml").read_text(encoding="utf-8"))
    theme = cfg["theme"]
    assert theme["base"] == "dark"
    assert theme["primaryColor"] == "#7c5cff"
    assert theme["backgroundColor"] == "#0f1117"
    assert theme["secondaryBackgroundColor"] != theme["backgroundColor"]


def test_pill_escapes_and_carries_the_hue():
    solid = walker_core.pill("<b>x</b> & y", walker_core.GOOD)
    assert "&lt;b&gt;" in solid
    assert "<b>x" not in solid
    assert solid.count(walker_core.GOOD) == 2   # border + color
    assert walker_core.rgba(walker_core.GOOD, 0.18) in solid

    outline = walker_core.pill("x", walker_core.GOOD, outline=True)
    assert "background:transparent" in outline
    assert "rgba(" not in outline


def test_pill_has_no_newline():
    assert "\n" not in walker_core.pill("a", walker_core.PRIMARY)


def test_stat_card_escapes_every_field_and_shows_the_value():
    card = walker_core.stat_card("<i>", "l&", "<v>", "c<a>", walker_core.PRIMARY)
    assert "<i>" not in card and "&lt;i&gt;" in card
    assert "l&amp;" in card
    assert "&lt;v&gt;" in card
    assert "c&lt;a&gt;" in card
    assert f"background:{walker_core.PRIMARY}" in card
    assert "\n" not in card


def test_stat_card_accepts_non_string_value():
    card = walker_core.stat_card("x", "l", 42, "", walker_core.PRIMARY)
    assert ">42<" in card


def test_evidence_row_escapes_head_and_snippet_but_trusts_the_badge():
    badge = walker_core.pill("entails", walker_core.GOOD)
    row = walker_core.evidence_row("<i>head</i>", "line one\nline two", badge,
                                    walker_core.PRIMARY)
    assert "<span" in row      # the pill's own span survives
    assert "&lt;i&gt;" in row and "<i>head" not in row
    assert "line one<br>line two" in row


def test_hero_answer_pills_the_citations_and_escapes_the_body():
    card = walker_core.hero_answer("a<b", [11, 12], "gpt · 2 briefs")
    assert "#11" in card and "#12" in card
    assert "<span" in card
    assert "a&lt;b" in card
    assert "opacity:.6" in card


def test_answer_gate_refuses_to_answer_over_zero_entails():
    claim = "Kurt Cobain was the most famous musician of the 1990s."
    gated, text = walker_core.answer_gate(claim, 0, n_iters=3, n_chunks=88)
    assert gated is True
    assert "does not answer this" in text
    assert "88" in text
    assert "3 agentic iterations" in text
    assert claim not in text


def test_answer_gate_passes_a_supported_answer_through_untouched():
    assert walker_core.answer_gate("A.", 4, n_iters=0, n_chunks=20) == (False, "A.")


def test_answer_gate_points_at_what_the_loop_found():
    gated, text = walker_core.answer_gate("x", 0, n_iters=2, n_chunks=40, found_entails=3)
    assert gated is True
    assert "Agentic retrieval" in text
    assert "3 entailing" in text


def test_loop_answer_caption_pluralizes_iterations():
    assert walker_core.loop_answer_caption(0) == "answered after 0 agentic iterations"
    assert walker_core.loop_answer_caption(1) == "answered after 1 agentic iteration"
    assert walker_core.loop_answer_caption(3) == "answered after 3 agentic iterations"


def test_digest_card_badge_renders_and_body_stays_byte_identical():
    import html as _html
    import re
    gc = _gc(gid=0, members=[1, 2, 3], size=3)
    src_of = {1: "wiki", 2: "wiki", 3: "brown"}
    txt = walker_core.group_digest(gc, {"mentions": {}, "names": {}, "corpus": {},
                                         "corpus_total": 0}, src_of)
    card = walker_core.digest_card(txt, "#4C78A8", badge="= c6 (global)")
    assert "= c6 (global)" in card
    body = re.search(r"<pre[^>]*>(.*)</pre>", card, re.S).group(1)
    assert _html.unescape(body.replace("<br>", "\n")) == txt


def test_digest_card_outline_badge_uses_the_warn_hue():
    card = walker_core.digest_card("g0 . 1 chunks", "#4C78A8", badge="splits c1, c2",
                                    badge_outline=True)
    assert walker_core.WARN in card


def test_chrome_palette_constants_are_hex():
    import re
    for name in ("PRIMARY", "GOOD", "BAD", "WARN", "MUTED", "BG_CARD", "BG_ROW", "BORDER"):
        c = getattr(walker_core, name)
        assert re.match(r"^#[0-9a-fA-F]{6}$", c), f"{name}={c!r} is not hex"
        walker_core.rgba(c, 0.1)   # must parse without raising


# ---------- P16 3D scene builder (T53) ----------

def _p3(**kw):
    base = dict(
        ords=[1, 2, 3],
        cid_of={1: 1, 2: 2, 3: None},
        src_of={1: "brown", 2: "wiki", 3: None},
        scores={1: 0.1, 2: 0.9, 3: 0.5},
        bodies={1: "hello world", 2: "another body", 3: ""},
    )
    base.update(kw)
    return base


def test_walk3d_payload_node_per_ord_with_palette_colour_and_no_label():
    p = walker_core.walk3d_payload(**_p3())
    ids = [n["id"] for n in p["nodes"]]
    assert ids == [1, 2, 3]
    assert p["nodes"][0]["color"] == walker_core.cid_color(1)
    assert p["nodes"][1]["color"] == walker_core.cid_color(2)
    assert p["nodes"][2]["color"] == "#DDDDDD"
    for n in p["nodes"]:
        assert "name" not in n and "label" not in n


def test_walk3d_payload_size_tracks_walk_score_and_flat_scores_are_uniform():
    p = walker_core.walk3d_payload(**_p3(ords=[1, 2], scores={1: 0.1, 2: 0.9},
                                          bodies={1: "a", 2: "b"},
                                          cid_of={1: 1, 2: 2}, src_of={1: "brown", 2: "wiki"}))
    sizes = {n["id"]: n["size"] for n in p["nodes"]}
    assert sizes[2] > sizes[1]

    p2 = walker_core.walk3d_payload(**_p3(ords=[1, 2], scores={1: 0.5, 2: 0.5},
                                           bodies={1: "a", 2: "b"},
                                           cid_of={1: 1, 2: 2}, src_of={1: "brown", 2: "wiki"}))
    assert all(n["size"] == 3.0 for n in p2["nodes"])


def test_walk3d_payload_tip_carries_ord_source_cid_score_and_clipped_body():
    body = "word " * 100   # 500 chars, well past tip_chars=200
    p = walker_core.walk3d_payload(**_p3(
        ords=[7], cid_of={7: 2}, src_of={7: "brown"}, scores={7: 0.5},
        bodies={7: body}))
    tip = p["nodes"][0]["tip"]
    body_segment = tip.split("<br>")[-1]
    assert len(body_segment) <= 210
    assert body_segment.endswith("…")
    assert "#7" in tip
    assert "brown" in tip
    assert "c2" in tip
    assert "0.500" in tip


def test_walk3d_payload_tip_escapes_html_in_body_and_source():
    p = walker_core.walk3d_payload(**_p3(
        ords=[1], cid_of={1: 1}, src_of={1: "brown"}, scores={1: 0.1},
        bodies={1: '<script>alert("x")</script> & co'}))
    tip = p["nodes"][0]["tip"]
    assert "<script>" not in tip
    assert "&lt;script&gt;" in tip
    assert "&amp;" in tip


def test_walk3d_payload_links_dedup_undirected_and_drop_offgraph_endpoints():
    edges = [
        {"src": 1, "dst": 2, "strength": 0.5},
        {"src": 2, "dst": 1, "strength": 0.5},   # reversed duplicate
        {"src": 1, "dst": 1, "strength": 1.0},   # self-loop
        {"src": 1, "dst": 99, "strength": 1.0},  # off-graph endpoint
        {"src": 2, "dst": 3, "strength": 0.25},
    ]
    p = walker_core.walk3d_payload(**_p3(edges=edges))
    assert p["links"] == [
        {"source": 1, "target": 2, "w": 0.5},
        {"source": 2, "target": 3, "w": 0.25},
    ]


def test_walk3d_payload_sprites_use_community_keywords_then_salient_fallback():
    p = walker_core.walk3d_payload(**_p3(
        ords=[1, 2, 3], cid_of={1: 1, 2: 2, 3: 2}, src_of={1: "brown", 2: "wiki", 3: "wiki"},
        scores={1: 0.1, 2: 0.5, 3: 0.9},
        bodies={1: "a", 2: "b", 3: "c"},
        kw={1: ["alpha", "beta", "gamma", "delta"]},
        salient={2: {"top": ["x", "y"]}, 3: {"top": ["y", "z"]}}))
    sprites = {s["cid"]: s for s in p["sprites"]}
    assert sprites[1]["text"] == "c1: alpha beta gamma"
    assert sprites[2]["text"] == "c2: x y z"
    assert sprites[2]["members"] == [2, 3]

    p2 = walker_core.walk3d_payload(**_p3(
        ords=[1], cid_of={1: 5}, src_of={1: "brown"}, scores={1: 0.1}, bodies={1: "a"}))
    assert p2["sprites"][0]["text"] == "c5"


def test_walk3d_payload_paths_from_pathways_pairs_filtered_and_ordered():
    pathways = {"pairs": [
        {"dwpc": 1.0, "path": [1, 2, 3]},          # fully in-graph
        {"dwpc": 0.8, "path": [1, 99, 2]},         # partly out, 2 survivors
        {"dwpc": 0.5, "path": [99, 3]},            # reduced to 1 survivor -> dropped
    ]}
    p = walker_core.walk3d_payload(**_p3(pathways=pathways))
    assert p["paths"] == [[1, 2, 3], [1, 2]]

    p_empty = walker_core.walk3d_payload(**_p3(pathways={"pairs": []}))
    assert p_empty["paths"] == []
    p_none = walker_core.walk3d_payload(**_p3(pathways=None))
    assert p_none["paths"] == []


def test_walk3d_payload_umap_present_only_when_every_node_has_coords():
    umap = {1: (0.0, 0.0, 0.0), 2: (10.0, 0.0, 0.0), 3: (0.0, 10.0, 0.0)}
    p = walker_core.walk3d_payload(**_p3(umap=umap))
    assert p["has_umap"] is True
    for n in p["nodes"]:
        assert len(n["umap"]) == 3
        assert all(isinstance(v, float) for v in n["umap"])

    partial = {1: (0.0, 0.0, 0.0), 2: (10.0, 0.0, 0.0)}   # missing ord 3
    p2 = walker_core.walk3d_payload(**_p3(umap=partial))
    assert p2["has_umap"] is False
    assert all("umap" not in n for n in p2["nodes"])

    p3_ = walker_core.walk3d_payload(**_p3(umap=None))
    assert p3_["has_umap"] is False
    assert all("umap" not in n for n in p3_["nodes"])


def test_walk3d_payload_empty_walk_returns_empty_scene():
    p = walker_core.walk3d_payload(
        ords=[], cid_of={}, src_of={}, scores={}, bodies={})
    assert p == {"nodes": [], "links": [], "sprites": [], "paths": [], "has_umap": False}


def test_walk3d_html_pins_the_exact_cdn_versions_and_embeds_parseable_json():
    import re
    p = walker_core.walk3d_payload(**_p3())
    doc = walker_core.walk3d_html(p)
    assert "3d-force-graph@1.73.4" in doc
    assert "three-spritetext@1.8.2" in doc
    assert walker_core.WALK3D_FG_URL in doc
    assert walker_core.WALK3D_ST_URL in doc

    m = re.search(r'<script id="w3d-data"[^>]*>(.*?)</script>', doc, re.S)
    raw = m.group(1).replace("<\\/", "</")
    assert json.loads(raw) == p


def test_walk3d_html_escapes_script_close_and_degrades_without_the_cdn():
    p = walker_core.walk3d_payload(**_p3(
        ords=[1], cid_of={1: 1}, src_of={1: "brown"}, scores={1: 0.1},
        bodies={1: "before </script> after"}))
    doc = walker_core.walk3d_html(p)

    m = __import__("re").search(r'<script id="w3d-data"[^>]*>(.*?)</script>', doc,
                                 __import__("re").S)
    assert "</script>" not in m.group(1)
    assert "<\\/b>" in m.group(1)   # the `</` guard fires on every `</`, not just `</script>`

    assert doc.count("onerror=") == 3   # three + force-graph + spritetext
    assert "3D scene unavailable" in doc


def test_walk3d_html_hides_controls_when_no_paths_and_no_umap():
    p_bare = walker_core.walk3d_payload(**_p3())   # no pathways/umap given
    assert p_bare["paths"] == [] and p_bare["has_umap"] is False
    doc_bare = walker_core.walk3d_html(p_bare)
    path_btn = doc_bare[doc_bare.index('id="pathbtn"'):doc_bare.index('>', doc_bare.index('id="pathbtn"'))]
    umap_btn = doc_bare[doc_bare.index('id="umapbtn"'):doc_bare.index('>', doc_bare.index('id="umapbtn"'))]
    assert "display:none" in path_btn
    assert "display:none" in umap_btn

    p_full = walker_core.walk3d_payload(**_p3(
        pathways={"pairs": [{"dwpc": 1.0, "path": [1, 2]}]},
        umap={1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)}))
    assert p_full["paths"] and p_full["has_umap"] is True
    doc_full = walker_core.walk3d_html(p_full)
    path_btn2 = doc_full[doc_full.index('id="pathbtn"'):doc_full.index('>', doc_full.index('id="pathbtn"'))]
    umap_btn2 = doc_full[doc_full.index('id="umapbtn"'):doc_full.index('>', doc_full.index('id="umapbtn"'))]
    assert "display:none" not in path_btn2
    assert "display:none" not in umap_btn2


def test_walk3d_html_loads_three_before_spritetext():
    """three-spritetext's UMD reads global THREE (absent from 3d-force-graph's
    private bundle): live console receipt was 'reading LinearFilter'. The
    plain three UMD script tag must precede both other libs."""
    h = walker_core.walk3d_html({"nodes": [], "links": [], "sprites": [],
                                 "paths": [], "has_umap": False})
    assert walker_core.WALK3D_THREE_URL in h
    assert h.index(walker_core.WALK3D_THREE_URL) < h.index(walker_core.WALK3D_FG_URL)
    assert h.index(walker_core.WALK3D_THREE_URL) < h.index(walker_core.WALK3D_ST_URL)


# ---------------------------------------------------------------- A14 gate
# KNOWN-BAD FIRST: these tests are written against the live defect artifact
# (operator screenshot, 2026-09-07) and MUST fail before T76 lands.

GALLAGHER = ("Noel Gallagher is the most famous musician of the 1990's, as he "
             "reached the height of his fame during the Britpop era (#3594) with "
             "massive commercial success like 'What's the Story (Morning Glory)?' "
             "(#7666).")


def test_is_superlative_detects_the_live_prompt():
    assert walker_core.is_superlative("who is the most famous musician of the 1990's?")
    assert walker_core.is_superlative("what is the largest hurricane on record")
    assert walker_core.is_superlative("who was the first person to fly")
    assert not walker_core.is_superlative("how do tropical storms strengthen into hurricanes")
    assert not walker_core.is_superlative("what damage did the hurricane cause")


def test_superlative_prompt_with_entails_is_still_gated():
    """A14(b): 2 entails about ONE candidate never establish a maximum over a
    population. The live defect: entails=2, contradicts=2, answer crowned."""
    gated, text = walker_core.answer_gate(
        GALLAGHER, entails=2, prompt="who is the most famous musician of the 1990's?",
        entail_ords=[3594, 7666], answer_ords=[3594, 7666],
        superlative_entails=0, n_chunks=88)
    assert gated, "a superlative claim with no ranking evidence must not stand"
    assert "cannot rank" in text.lower() or "cannot crown" in text.lower()
    assert "Noel Gallagher is the most famous" not in text


def test_answer_citing_a_non_entailing_ord_is_gated():
    """A14(c): the live answer cited #7666, which Reason marked insufficient.
    That was rendered as a red note; it must gate."""
    gated, text = walker_core.answer_gate(
        "Nirvana defined the era. #6323 #9999", entails=1,
        prompt="what defined the 1990s", entail_ords=[6323],
        answer_ords=[6323, 9999], superlative_entails=0, n_chunks=88)
    assert gated, "citing a non-entailing ord must gate, not annotate"


def test_cited_ords_parses_hash_markers_in_order_deduped():
    assert walker_core.cited_ords("Nirvana defined the era. #6323 #9999 #6323") == [6323, 9999]
    assert walker_core.cited_ords("no citations here") == []


def test_needs_more_evidence_matches_answer_gate_zero_entails():
    """A15: the loop's trigger must agree with the gate on the A6 zero-entails
    case -- both say True."""
    # n_iters shapes the gated TEXT only, never the verdict -- not a param here.
    assert walker_core.needs_more_evidence("x", 0) is True
    gated, _ = walker_core.answer_gate("x", 0, n_iters=0)
    assert walker_core.needs_more_evidence("x", 0) == gated


def test_needs_more_evidence_matches_answer_gate_superlative_unranked():
    """A15: the live defect -- 3 entails, none ranking the population -- must
    trigger the loop even though entails > 0 (the old `_entails == 0` trigger
    missed exactly this case)."""
    kw = dict(prompt="who is the most famous musician of the 1990's?",
              entail_ords=[3594, 7666], answer_ords=[3594, 7666],
              superlative_entails=0)
    gated, _ = walker_core.answer_gate(GALLAGHER, entails=2, **kw)
    assert gated is True
    assert walker_core.needs_more_evidence(GALLAGHER, 2, **kw) == gated is True


def test_needs_more_evidence_matches_answer_gate_bad_citation():
    kw = dict(prompt="what defined the 1990s", entail_ords=[6323],
              answer_ords=[6323, 9999], superlative_entails=0)
    gated, _ = walker_core.answer_gate("Nirvana defined the era. #6323 #9999", 1, **kw)
    assert gated is True
    assert walker_core.needs_more_evidence(
        "Nirvana defined the era. #6323 #9999", 1, **kw) == gated is True


def test_needs_more_evidence_matches_answer_gate_clean_answer():
    """A7 do-no-harm: a clean, ungated answer must not trigger the loop."""
    gated, _ = walker_core.answer_gate("A.", 4, n_iters=0, n_chunks=20)
    assert gated is False
    assert walker_core.needs_more_evidence("A.", 4) == gated is False


def test_superlative_prompt_with_matching_superlative_evidence_passes():
    """A14(b) is not a blanket ban: WHEN a chunk itself carries the ranking
    claim, the answer may stand."""
    gated, _ = walker_core.answer_gate(
        "Nirvana was the best-selling act of the decade. #6323", entails=3,
        prompt="who was the best selling act of the 1990s", entail_ords=[6323],
        answer_ords=[6323], superlative_entails=2, n_chunks=88)
    assert not gated


# ------------------------------------------- A16 reflexive superlative guard
# KNOWN-BAD FIRST: written against the live premise chain (operator, 2026-09-07)
# "supports -- Noel Gallagher reached the height of his fame during the Britpop
# era" offered as support for "the most famous musician of the 1990's".

def test_reflexive_superlatives_are_not_population_claims():
    R = walker_core.is_reflexive_superlative
    assert R("Noel Gallagher reached the height of his fame during the Britpop era")
    assert R("it was his biggest hit")
    assert R("her peak years were the 1980s")
    assert R("their most successful album to date")
    assert R("a career high for the band")
    # population-scoped claims are NOT reflexive -- these may answer a superlative
    assert not R("the best-selling album of the decade")
    assert not R("the most famous musician of the 1990s")
    assert not R("Nirvana was the biggest band in the world that year")


def test_reflexive_evidence_does_not_exempt_the_superlative_gate():
    """A16(b): a chunk saying X peaked in his own career is not evidence that
    X led a population -- superlative_entails must not count it, so A14(b)
    still gates."""
    n = walker_core.count_population_superlatives(
        ["Noel Gallagher reached the height of his fame during the Britpop era",
         "his most successful album was released in 1995"])
    assert n == 0, "self-scoped superlatives must not count as ranking evidence"
    n2 = walker_core.count_population_superlatives(
        ["Nevermind was the best-selling album of the decade"])
    assert n2 == 1
