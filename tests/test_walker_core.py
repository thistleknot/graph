"""tests/test_walker_core.py -- the walker's UI-free logic, without Streamlit.

Spec: .spec/specs/graph-explorer/design.md 6.21(c)
Task: playbook.md T36
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

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
