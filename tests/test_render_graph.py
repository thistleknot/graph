"""Exercises render_graph.main().

Same shape as test_ingest_brown.py: ChunkGraph is stubbed, so argv parsing, the
default output naming, the sparse-only wiring (embed_fn=None) and the path
handed to draw() are covered without spending a spring_layout or touching
Postgres/matplotlib.

Run:  pytest tests/test_render_graph.py -v      (no container needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import render_graph

# ---------------------------------------------------------------- the stubs


class FakeGraph:
    """Duck-types the ChunkGraph surface render_graph.main() touches."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.n = 42
        self.blend_mode = "sparse-only"
        self.fit_args = None
        self.drawn = None
        self.min_size = None
        self.A = np.ones((4, 4))          # sum()//2 == 8 edges

    def fit(self, docs, doc_ids=None):
        self.fit_args = (docs, doc_ids)
        return self

    def communities(self, min_size=5):
        self.min_size = min_size
        return [{"members": [0, 1]}, {"members": [2, 3]}]

    def draw(self, path):
        self.drawn = path
        return path


@pytest.fixture
def stubbed(monkeypatch):
    """Stubs ChunkGraph + load_docs; returns the recorder."""
    rec = {}

    def fake_chunkgraph(**kwargs):
        rec["cg"] = FakeGraph(**kwargs)
        return rec["cg"]

    monkeypatch.setattr(render_graph, "ChunkGraph", fake_chunkgraph)
    monkeypatch.setattr(
        render_graph, "load_docs",
        lambda n_docs: (["ca01"] * n_docs, ["text one"] * n_docs))
    return rec


# ---------------------------------------------------------------- argv / naming


def test_main_defaults_to_n_docs_50_and_matching_png(monkeypatch, stubbed):
    """No argv -> ingest_brown.N_DOCS docs, named after that count."""
    monkeypatch.setattr(sys, "argv", ["render_graph.py"])
    out = render_graph.main()
    assert render_graph.N_DOCS == 50
    assert out == "graph_brown50.png"
    assert len(stubbed["cg"].fit_args[0]) == 50


def test_main_reads_n_docs_from_argv_and_names_png_after_it(monkeypatch, stubbed):
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "7"])
    out = render_graph.main()
    docs, doc_ids = stubbed["cg"].fit_args
    assert out == "graph_brown7.png"
    assert len(docs) == 7
    assert len(doc_ids) == 7


def test_main_explicit_out_path_overrides_default_name(monkeypatch, stubbed):
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "3", "custom/spot.png"])
    out = render_graph.main()
    assert out == "custom/spot.png"
    assert stubbed["cg"].drawn == "custom/spot.png"


def test_main_coerces_argv_n_docs_to_int(monkeypatch, stubbed):
    """argv is str; load_docs slicing needs a real int."""
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "4"])
    render_graph.main()
    assert stubbed["cg"].fit_args[0] == ["text one"] * 4


def test_main_rejects_non_numeric_n_docs(monkeypatch, stubbed):
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "fifty"])
    with pytest.raises(ValueError):
        render_graph.main()


# ---------------------------------------------------------------- wiring


def test_main_fits_sparse_only(monkeypatch, stubbed):
    """embed_fn=None is what keeps the dense arm (and model2vec) out."""
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "2"])
    render_graph.main()
    assert stubbed["cg"].init_kwargs == {"embed_fn": None}


def test_main_passes_doc_ids_through_to_fit(monkeypatch, stubbed):
    """R8: doc_id[i] must name the source document, so fit() needs the ids."""
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "2"])
    render_graph.main()
    docs, doc_ids = stubbed["cg"].fit_args
    assert doc_ids == ["ca01", "ca01"]


def test_main_uses_min_size_5_for_communities(monkeypatch, stubbed):
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "2"])
    render_graph.main()
    assert stubbed["cg"].min_size == 5


def test_main_draws_to_the_returned_path(monkeypatch, stubbed):
    """The returned path is only meaningful if draw() actually got it."""
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "2"])
    out = render_graph.main()
    assert stubbed["cg"].drawn == out == "graph_brown2.png"


# ---------------------------------------------------------------- progress log


def test_main_logs_counts_before_the_expensive_draw(monkeypatch, stubbed, capsys):
    """The 0-byte-log bug: the draw notice must be flushed before layout runs."""
    monkeypatch.setattr(sys, "argv", ["render_graph.py", "2"])
    render_graph.main()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines[0].startswith("corpus: 2 docs")
    assert "42 chunks" in lines[1]
    assert "blend_mode=sparse-only" in lines[1]
    assert lines[2].startswith("graph : 8 edges, 2 communities")
    assert "spring_layout over 42 nodes" in lines[3]
    assert lines[4].startswith("wrote : graph_brown2.png")
