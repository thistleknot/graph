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


# ------------------------------------------------- _community_layout (R8.11)


class TestCommunityLayout:
    """ChunkGraph._community_layout must make the partition VISIBLE. Pins R13.

    Pins the property that a single global spring pass failed: same-community
    nodes co-located, communities separated. The prior layout
    (spring_layout(k=0.9), ~19x the networkx default at n=445) rendered a
    76.4%-intra-community graph as a featureless disc.
    """

    @staticmethod
    def _planted(seed=11, n_comm=6, per=20, p_in=0.168, p_out=0.010):
        """Planted-partition fixture matched to the MEASURED brown-50 graph.

        Defaults give ~3.4 mean degree and ~80% intra-community edges against
        the live run's 4.2 / 76.4%.  Density matters: a fixture of dense
        cliques is separable by ANY layout and would not discriminate.
        """
        import random
        import networkx as nx
        rng = random.Random(seed)
        Gx = nx.Graph()
        part = {c * per + i: c for c in range(n_comm) for i in range(per)}
        Gx.add_nodes_from(part)
        nodes = list(part)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                if rng.random() < (p_in if part[a] == part[b] else p_out):
                    Gx.add_edge(a, b, weight=1.0)
        return Gx, part

    @staticmethod
    def _misplaced(pos, part):
        """Nodes sitting nearer a foreign community centroid than their own."""
        import math
        cents = {}
        for c in set(part.values()):
            ms = [v for v in pos if part[v] == c]
            cents[c] = (sum(pos[v][0] for v in ms) / len(ms),
                        sum(pos[v][1] for v in ms) / len(ms))
        return [v for v in pos
                if min(cents, key=lambda c: math.dist(pos[v], cents[c])) != part[v]]

    def _layout(self, Gx, part):
        import networkx as nx
        from chunkgraph import ChunkGraph
        return ChunkGraph._community_layout(Gx, part, nx)

    def test_covers_every_node(self):
        Gx, part = self._planted()
        pos = self._layout(Gx, part)
        assert set(pos) == set(Gx.nodes()), "every node needs a position"
        assert all(len(p) == 2 for p in pos.values())

    def test_every_node_nearest_its_own_centroid(self):
        """The property the old layout violated: blobs are spatially distinct.

        Stated as centroid membership, not raw pairwise distance -- a blob's
        DIAMETER is not comparable to the GAP between blobs, so demanding
        max(intra) < min(inter) would require separation wider than the blobs
        themselves.
        """
        Gx, part = self._planted()
        pos = self._layout(Gx, part)
        misplaced = self._misplaced(pos, part)
        assert not misplaced, f"{len(misplaced)} nodes nearer a foreign centroid: {misplaced[:5]}"

    def test_blobs_do_not_overlap(self):
        """Centre separation must exceed the two blob radii combined."""
        import itertools, math
        Gx, part = self._planted()
        pos = self._layout(Gx, part)
        geom = {}
        for c in set(part.values()):
            ms = [v for v in pos if part[v] == c]
            cx = sum(pos[v][0] for v in ms) / len(ms)
            cy = sum(pos[v][1] for v in ms) / len(ms)
            geom[c] = ((cx, cy), max(math.dist(pos[v], (cx, cy)) for v in ms))
        for a, b in itertools.combinations(geom, 2):
            (ca, ra), (cb, rb) = geom[a], geom[b]
            assert math.dist(ca, cb) > ra + rb, (
                f"communities {a},{b} overlap: centres {math.dist(ca, cb):.3f} "
                f"<= radii {ra + rb:.3f}")

    def test_blob_area_scales_with_size(self):
        """A big community claims a bigger radius, and still must not collide."""
        import itertools, math
        import networkx as nx
        Gx = nx.Graph()
        part = {}
        sizes = [3, 25]
        base = 0
        for c, sz in enumerate(sizes):
            members = list(range(base, base + sz))
            for v in members:
                part[v] = c
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    Gx.add_edge(a, b, weight=1.0)
            base += sz
        Gx.add_edge(0, sizes[0], weight=0.05)
        pos = self._layout(Gx, part)
        geom = {}
        for c in range(len(sizes)):
            ms = [v for v in pos if part[v] == c]
            cx = sum(pos[v][0] for v in ms) / len(ms)
            cy = sum(pos[v][1] for v in ms) / len(ms)
            geom[c] = ((cx, cy), max(math.dist(pos[v], (cx, cy)) for v in ms))
        assert geom[1][1] > geom[0][1], "larger community should occupy more area"
        (ca, ra), (cb, rb) = geom[0], geom[1]
        assert math.dist(ca, cb) > ra + rb, "big blob swallowed the small one"

    def test_singleton_community_is_placed(self):
        import networkx as nx
        Gx, part = self._planted(n_comm=2, per=5)
        Gx.add_node(99)
        part[99] = 7
        pos = self._layout(Gx, part)
        assert 99 in pos, "an unconnected singleton community still needs a slot"

    def test_fixture_is_hard_enough_to_discriminate(self):
        """Guards the guard: a global spring pass must FAIL this fixture.

        Without this, the suite could pass against the very layout it replaced.
        Measured on the default fixture: single-pass spring misplaces ~39% of
        nodes at k=0.9 and ~58% at the networkx default -- tuning k does not
        rescue it, which is why the fix is two-pass and not a parameter change.
        """
        import networkx as nx
        Gx, part = self._planted()
        for k in (0.9, None):
            single = nx.spring_layout(Gx, weight="weight", k=k, iterations=200, seed=7)
            bad = self._misplaced(single, part)
            assert len(bad) > 0.15 * len(single), (
                f"single-pass spring (k={k}) separated the fixture "
                f"({len(bad)}/{len(single)} misplaced) -- fixture too easy")
        assert not self._misplaced(self._layout(Gx, part), part)
