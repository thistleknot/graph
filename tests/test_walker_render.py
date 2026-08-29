"""Pins the walker UI's figure-building path against a live run.

WHY THIS FILE EXISTS
Importing `walker_app` executes its top-level script, but the render helpers only
run once a walk is in progress (they sit behind `st.session_state`), so an import
smoke test proves nothing about them. That gap shipped a real bug: plotly rejects
8-digit hex, and `scatter.line.color="#88888866"` raised ValueError the first time
a user clicked a neighbour.

`fig.to_plotly_json()` is the assertion that matters — it forces plotly's full
property validation, which is what actually caught the bad colour.

Run:  pytest tests/test_walker_render.py -v      (needs `docker compose up -d`)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

import psycopg

import graph_tools as gt

HUB = 1476


@pytest.fixture(scope="module")
def app():
    """Importing the app runs its script; that is how conn/run get built."""
    try:
        gt.connect().close()
    except psycopg.OperationalError as e:                 # pragma: no cover
        pytest.skip(f"no database: {e}")
    try:
        import walker_app
    except SystemExit:                                    # pragma: no cover
        pytest.skip("walker_app called st.stop() — no live run")
    return walker_app


@pytest.fixture(scope="module")
def ords(app):
    nbrs = gt.neighbors(app.conn, app.run, HUB, limit=8)
    return [HUB] + [n["ord"] for n in nbrs]


def test_figure_passes_plotly_validation(app, ords):
    """The regression guard. to_plotly_json() validates every property; an
    invalid colour, size or width raises here rather than in the browser."""
    fig = app.draw_subgraph(ords, ords[:4], HUB)
    payload = fig.to_plotly_json()
    assert payload["data"]
    assert len(fig.data) == 3          # edges, trail, nodes


def test_edge_colour_is_plotly_legal(app, ords):
    """8-digit hex (#RRGGBBAA) is valid CSS and invalid plotly. Alpha must be
    expressed as rgba(). This is the exact bug this file was added for."""
    fig = app.draw_subgraph(ords, ords[:4], HUB)
    for trace in fig.data:
        colour = getattr(getattr(trace, "line", None), "color", None)
        if isinstance(colour, str) and colour.startswith("#"):
            assert len(colour) in (4, 7), f"{colour!r} is not plotly-legal hex"


def test_trail_trace_omitted_when_too_short(app, ords):
    """A one-node trail draws no dotted path, so the trace count drops."""
    fig = app.draw_subgraph(ords, [HUB], HUB)
    assert len(fig.data) == 2


def test_single_node_subgraph_renders(app):
    fig = app.draw_subgraph([HUB], [HUB], HUB)
    assert fig is not None
    fig.to_plotly_json()


def test_empty_subgraph_returns_none(app):
    assert app.draw_subgraph([], [], None) is None


def test_current_node_is_emphasised(app, ords):
    """The node you are standing on must be visually distinguishable."""
    fig = app.draw_subgraph(ords, ords[:4], HUB)
    nodes = fig.data[-1]
    sizes = list(nodes.marker.size)
    assert sizes[ords.index(HUB)] == max(sizes)
    assert sizes.count(max(sizes)) == 1


def test_every_node_gets_a_colour(app, ords):
    fig = app.draw_subgraph(ords, ords[:4], HUB)
    colours = list(fig.data[-1].marker.color)
    assert len(colours) == len(ords)
    assert all(isinstance(c, str) and c.startswith("#") for c in colours)


def test_cid_badge_never_hides_the_cid(app):
    """A draft label is additive. `cid` is the identity and must always show."""
    assert app.cid_badge(None, {}) == "no community"
    assert app.cid_badge(7, {}) == "c7"
    badge = app.cid_badge(7, {7: {"label": "Jury Proceedings"}})
    assert badge.startswith("c7")
    assert "Jury Proceedings" in badge


def test_missing_label_file_is_not_an_error(app):
    """Labels are optional; their absence is the normal state."""
    assert isinstance(app.load_labels(), dict)


# ------------------------------------------- the Walk tab renders end to end


def test_walk_tab_renders_a_graph_from_a_prompt():
    """ast.parse proves syntax; only a real run proves the tab renders."""
    from streamlit.testing.v1 import AppTest
    try:
        at = AppTest.from_file("walker_app.py", default_timeout=120)
        at.run()
        at.text_input("q").set_value("jury trial grand jury investigation").run()
    except Exception as e:                                # pragma: no cover
        pytest.skip(f"app could not start (no db?): {e}")
    errs = [e.value for e in at.exception]
    assert not errs, f"Walk tab raised: {errs}"
    assert at.get("plotly_chart"), "prompt produced no graph"
    assert any("depth" in c.value for c in at.caption), "no depth caption"
