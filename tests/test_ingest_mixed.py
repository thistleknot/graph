"""Exercises ingest_mixed.py's loaders, load_mixed(), and main().

Offline-only: the two network seams (`ingest_mixed._hf_rows` and
`ingest_brown.load_docs`) are monkeypatched, so the prefixing/stride/count
logic of the loaders is exercised, not stubbed away, and `datasets` never
needs to be installed.

Run:  pytest tests/test_ingest_mixed.py -q

Spec: .spec/specs/graph-explorer/design.md sec 6.14 R20 - Task: playbook.md T2
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ingest_mixed

# --------------------------------------------------------------------- _take


def test_take_strides_then_caps():
    items = [f"item{i}" for i in range(20)]
    taken = ingest_mixed._take(items, 3, 5)
    assert [i for i, _ in taken] == [0, 5, 10]


def test_take_drops_blank_before_slicing():
    items = ["a", "", "b", "   ", "c"]
    taken = ingest_mixed._take(items, 10, 1)
    assert [t for _, t in taken] == ["a", "b", "c"]


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def fake_hf(monkeypatch):
    calls = []

    def _fake(path, config, split, field):
        calls.append((path, config, split, field))
        if path == ingest_mixed.QUOTES_DATASET:
            return [f"quote {i}" for i in range(20)]
        if path == ingest_mixed.WIKI_DATASET:
            return [f"page {i} " + "body " * 30 for i in range(20)]
        raise AssertionError(f"unexpected dataset path {path}")

    monkeypatch.setattr(ingest_mixed, "_hf_rows", _fake)
    return calls


@pytest.fixture
def fake_brown(monkeypatch):
    def _fake(n, stride=1):
        return ([f"ca{i:02d}" for i in range(n)], [f"brown text {i}" for i in range(n)])

    monkeypatch.setattr(ingest_mixed.ingest_brown, "load_docs", _fake)


class FakeGraphNoSources:
    """Duck-types today's ChunkGraph.fit signature (no sources param)."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.n = 42
        self.blend_mode = "sparse-only"
        self.fit_args = None
        self.fit_kwargs = None

    def fit(self, docs, doc_ids=None):
        self.fit_args = (docs, doc_ids)
        self.fit_kwargs = {}
        return self

    def edges(self):
        return [{"src": 0, "dst": 1}]

    def communities(self, min_size=5):
        self.min_size = min_size
        return {0: [0, 1]}


class FakeGraphWithSources(FakeGraphNoSources):
    """Duck-types the post-T4 ChunkGraph.fit signature (sources param present)."""

    def fit(self, docs, doc_ids=None, sources=None):
        self.fit_args = (docs, doc_ids)
        self.fit_kwargs = {"sources": sources}
        return self


@pytest.fixture
def stubbed(monkeypatch, fake_hf, fake_brown):
    rec = {}

    def fake_chunkgraph(**kwargs):
        rec["cg"] = FakeGraphNoSources(**kwargs)
        return rec["cg"]

    def fake_save(cg, label):
        rec["save"] = (cg, label)
        return "run-id-sentinel"

    monkeypatch.setattr(ingest_mixed, "ChunkGraph", fake_chunkgraph)
    monkeypatch.setattr(ingest_mixed.pg_store, "save", fake_save)
    return rec


# ------------------------------------------------------------------ loaders


def test_load_quotes_ids_and_sources(fake_hf):
    doc_ids, docs, sources = ingest_mixed.load_quotes(3, 1)
    assert all(d.startswith("quotes/") for d in doc_ids)
    assert sources == ["quotes"] * 3
    assert len(doc_ids) == len(docs) == len(sources) == 3


def test_load_wiki_preserves_prestride_index(fake_hf):
    doc_ids, docs, sources = ingest_mixed.load_wiki(3, 2)
    assert doc_ids == ["wiki/0", "wiki/2", "wiki/4"]


def test_load_brown_ids_and_sources(fake_brown):
    doc_ids, docs, sources = ingest_mixed.load_brown(2, 10)
    assert doc_ids == ["brown/ca00", "brown/ca01"]
    assert sources == ["brown", "brown"]


def test_loaders_call_hf_rows_with_expected_dataset_identifiers(fake_hf):
    ingest_mixed.load_quotes(3, 1)
    ingest_mixed.load_wiki(3, 1)
    assert (ingest_mixed.QUOTES_DATASET, None, "train", "quote") in fake_hf
    assert (ingest_mixed.WIKI_DATASET, ingest_mixed.WIKI_CONFIG, "train", "page") in fake_hf


def test_loaders_with_n_zero_return_empty_and_skip_hf(fake_hf):
    assert ingest_mixed.load_quotes(0, 1) == ([], [], [])
    assert ingest_mixed.load_wiki(0, 1) == ([], [], [])
    assert fake_hf == []


def test_load_brown_n_zero_skips_incumbent(monkeypatch):
    def _boom(n, stride=1):
        raise AssertionError("load_docs should not be called when n_docs<=0")

    monkeypatch.setattr(ingest_mixed.ingest_brown, "load_docs", _boom)
    assert ingest_mixed.load_brown(0, 1) == ([], [], [])


# --------------------------------------------------------------- load_mixed


def test_load_mixed_counts_per_source(fake_hf, fake_brown):
    doc_ids, docs, sources = ingest_mixed.load_mixed(2, 3, 4, 1, 1, 1)
    assert len(docs) == 9
    assert collections.Counter(sources) == {"brown": 2, "quotes": 3, "wiki": 4}


def test_load_mixed_doc_id_prefix_matches_source(fake_hf, fake_brown):
    doc_ids, docs, sources = ingest_mixed.load_mixed(2, 3, 4, 1, 1, 1)
    for did, src in zip(doc_ids, sources):
        assert did.split("/", 1)[0] == src


def test_load_mixed_order_is_brown_quotes_wiki(fake_hf, fake_brown):
    doc_ids, docs, sources = ingest_mixed.load_mixed(2, 3, 4, 1, 1, 1)
    assert sources == ["brown"] * 2 + ["quotes"] * 3 + ["wiki"] * 4
    assert len(doc_ids) == len(docs) == len(sources)


# --------------------------------------------------------------------- main


def test_main_defaults_label_mixed(stubbed):
    ingest_mixed.main([])
    assert stubbed["save"][1] == "mixed"


def test_main_reads_label_and_counts_from_argv(stubbed):
    ingest_mixed.main(["mixed-smoke", "--brown", "2", "--quotes", "3", "--wiki", "4"])
    docs, doc_ids = stubbed["cg"].fit_args
    assert len(docs) == 9
    assert len(doc_ids) == 9
    assert stubbed["save"][1] == "mixed-smoke"


def test_main_stride_args_reach_only_their_own_loader(monkeypatch, stubbed):
    recorders = {}

    def make_recorder(name, orig):
        def _rec(n, stride, **kw):
            recorders[name] = (n, stride)
            return orig(n, stride)
        return _rec

    monkeypatch.setattr(ingest_mixed, "load_brown", make_recorder("brown", ingest_mixed.load_brown))
    monkeypatch.setattr(ingest_mixed, "load_quotes", make_recorder("quotes", ingest_mixed.load_quotes))
    monkeypatch.setattr(ingest_mixed, "load_wiki", make_recorder("wiki", ingest_mixed.load_wiki))

    ingest_mixed.main(["lbl", "--brown", "1", "--quotes", "1", "--wiki", "1", "--wiki-stride", "7"])

    assert recorders["wiki"] == (1, 7)
    assert recorders["brown"] == (1, 10)
    assert recorders["quotes"] == (1, 1)


def test_main_sparse_only_wiring_when_model_dir_unset(monkeypatch, stubbed):
    monkeypatch.delenv("CHUNKGRAPH_MODEL_DIR", raising=False)
    ingest_mixed.main([])
    assert stubbed["cg"].init_kwargs == {"embed_fn": None}


def test_main_fit_forward_seam_passes_sources(monkeypatch, fake_hf, fake_brown):
    rec = {}

    def fake_chunkgraph(**kwargs):
        rec["cg"] = FakeGraphWithSources(**kwargs)
        return rec["cg"]

    monkeypatch.setattr(ingest_mixed, "ChunkGraph", fake_chunkgraph)
    monkeypatch.setattr(ingest_mixed.pg_store, "save", lambda cg, label: "run-id-sentinel")

    ingest_mixed.main(["lbl", "--brown", "2", "--quotes", "3", "--wiki", "4"])

    sources = rec["cg"].fit_kwargs["sources"]
    assert len(sources) == 9
    assert collections.Counter(sources) == {"brown": 2, "quotes": 3, "wiki": 4}


def test_main_fit_backward_seam_completes_without_sources(stubbed):
    run_id = ingest_mixed.main(["lbl", "--brown", "1", "--quotes", "1", "--wiki", "1"])
    assert run_id == "run-id-sentinel"
    assert stubbed["cg"].fit_kwargs == {}


def test_main_reaches_communities_and_save(stubbed):
    ingest_mixed.main([])
    assert stubbed["cg"].min_size == 5
    assert stubbed["save"][0] is stubbed["cg"]


def test_wiki_title_parses_head_and_rescues_stride_victims(fake_hf, monkeypatch):
    """--wiki-title matches by EQUALITY on the parsed '= Title =' head -- no
    pattern scanning (operator, 2026-09-03: regex selection was an ad hoc
    hack; parse once at the boundary, compare exactly). Stride victims are
    rescued, already-sampled docs are not duplicated, output stays sorted."""
    rows = ["= Article %d = body text" % i for i in range(10)]
    rows[3] = "= Battle of Midway = The Battle of Midway was decisive"
    rows[5] = "= AC/DC (band) = rock band"          # metachars stay literal
    monkeypatch.setattr(ingest_mixed, "_hf_rows", lambda *a, **k: rows)
    ids, docs, src = ingest_mixed.load_wiki(
        3, 4, titles=["battle of midway", "AC/DC (band)"])
    idx = [int(i.split("/")[1]) for i in ids]
    assert 3 in idx and 5 in idx                    # rescued, case-insensitive
    assert idx == sorted(idx) and len(idx) == len(set(idx))
    assert all(s_ == "wiki" for s_ in src)
    # equality means no substring surprises: 'Midway' alone matches nothing
    ids2, _, _ = ingest_mixed.load_wiki(0, 1, titles=["Midway"])
    assert ids2 == []


def test_wiki_title_parser_is_the_single_authority():
    """wiki_title(): the one head parser -- consumers never scan text."""
    assert ingest_mixed.wiki_title("= Battle of Midway = text") == "Battle of Midway"
    assert ingest_mixed.wiki_title("  = Spaced = body") == "Spaced"
    assert ingest_mixed.wiki_title("no head here") is None
    assert ingest_mixed.wiki_title("= = ") is None
