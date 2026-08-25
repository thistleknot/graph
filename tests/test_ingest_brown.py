"""Exercises ingest_brown.load_docs()/main().

load_docs() is checked against the real Brown corpus (cheap at n_docs=3).
main() is checked with ChunkGraph/pg_store.save stubbed out, so argv parsing,
the sparse-only wiring (embed_fn=None) and the label handed to save() are
covered without touching Postgres or the embedding stack.

Run:  pytest tests/test_ingest_brown.py -v      (no container needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ingest_brown

# ---------------------------------------------------------------- load_docs


@pytest.fixture(scope="module")
def loaded():
    return ingest_brown.load_docs(3)


def test_load_docs_returns_paired_ids_and_docs(loaded):
    doc_ids, docs = loaded
    assert len(doc_ids) == 3
    assert len(docs) == 3


def test_load_docs_strides_for_genre_spread(loaded):
    """STRIDE=10 over brown.fileids(), which starts with 44 'ca' files."""
    doc_ids, _ = loaded
    assert list(doc_ids) == ["ca01", "ca11", "ca21"]


def test_load_docs_returns_real_brown_text(loaded):
    """ca01 is the Fulton County grand jury report — fails if body is emptied."""
    _, docs = loaded
    assert "Fulton County Grand Jury" in docs[0]


def test_load_docs_joins_paragraphs_with_blank_line(loaded):
    _, docs = loaded
    assert "\n\n" in docs[0]
    for d in docs:
        assert d.strip()
        assert not d.startswith("\n")


def test_load_docs_honours_n_docs():
    doc_ids, docs = ingest_brown.load_docs(1)
    assert doc_ids == ["ca01"]
    assert len(docs) == 1


# ---------------------------------------------------------------- the stubs


class FakeGraph:
    """Duck-types the ChunkGraph surface main() touches."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.n = 42
        self.blend_mode = "sparse-only"
        self.fit_args = None

    def fit(self, docs, doc_ids=None):
        self.fit_args = (docs, doc_ids)
        return self

    def edges(self):
        return [{"src": 0, "dst": 1}]

    def communities(self, min_size=5):
        self.min_size = min_size
        return {0: [0, 1]}


@pytest.fixture
def stubbed(monkeypatch):
    """Stubs ChunkGraph + pg_store.save; returns the recorder."""
    rec = {}

    def fake_chunkgraph(**kwargs):
        rec["cg"] = FakeGraph(**kwargs)
        return rec["cg"]

    def fake_save(cg, label):
        rec["save"] = (cg, label)
        return "run-id-sentinel"

    monkeypatch.setattr(ingest_brown, "ChunkGraph", fake_chunkgraph)
    monkeypatch.setattr(ingest_brown.pg_store, "save", fake_save)
    monkeypatch.setattr(
        ingest_brown, "load_docs",
        lambda n_docs: (["ca01"] * n_docs, ["text one"] * n_docs))
    return rec


def test_main_defaults_to_brown_50_label(monkeypatch, stubbed):
    monkeypatch.setattr(sys, "argv", ["ingest_brown.py"])
    run_id = ingest_brown.main()
    assert stubbed["save"][1] == "brown-50"
    assert run_id == "run-id-sentinel"


def test_main_reads_label_and_n_docs_from_argv(monkeypatch, stubbed):
    monkeypatch.setattr(sys, "argv", ["ingest_brown.py", "mylabel", "7"])
    ingest_brown.main()
    docs, doc_ids = stubbed["cg"].fit_args
    assert stubbed["save"][1] == "mylabel"
    assert len(docs) == 7
    assert len(doc_ids) == 7


def test_main_fits_sparse_only(monkeypatch, stubbed):
    """embed_fn=None is what keeps the dense arm (and model2vec) out."""
    monkeypatch.setattr(sys, "argv", ["ingest_brown.py"])
    ingest_brown.main()
    assert stubbed["cg"].init_kwargs == {"embed_fn": None}


def test_main_passes_doc_ids_through_to_fit(monkeypatch, stubbed):
    """R8: doc_id[i] must name the source document, so fit() needs the ids."""
    monkeypatch.setattr(sys, "argv", ["ingest_brown.py", "lbl", "2"])
    ingest_brown.main()
    docs, doc_ids = stubbed["cg"].fit_args
    assert doc_ids == ["ca01", "ca01"]
    assert stubbed["save"][0] is stubbed["cg"]


def test_main_uses_min_size_5_for_communities(monkeypatch, stubbed):
    monkeypatch.setattr(sys, "argv", ["ingest_brown.py"])
    ingest_brown.main()
    assert stubbed["cg"].min_size == 5
