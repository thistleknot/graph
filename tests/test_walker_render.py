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

# HUB is derived from the run the app loaded -- see the `hub` fixture; a
# constant pinned to one ingest broke on the next (ord 1476 on a 51-node run).


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
def hub(app):
    """Highest-degree node of the run the app actually loaded."""
    with app.conn.cursor() as cur:
        cur.execute("""SELECT a FROM edge_sym WHERE run_id = %s AND valid_to IS NULL
                        GROUP BY a ORDER BY count(*) DESC, a LIMIT 1""", (app.run.run_id,))
        return cur.fetchone()["a"]


@pytest.fixture(scope="module")
def ords(app, hub):
    nbrs = gt.neighbors(app.conn, app.run, hub, limit=8)
    return [hub] + [n["ord"] for n in nbrs]


def test_figure_passes_plotly_validation(app, ords, hub):
    """The regression guard. to_plotly_json() validates every property; an
    invalid colour, size or width raises here rather than in the browser."""
    fig = app.draw_subgraph(ords, ords[:4], hub)
    payload = fig.to_plotly_json()
    assert payload["data"]
    assert len(fig.data) == 3          # edges, trail, nodes


def test_edge_colour_is_plotly_legal(app, ords, hub):
    """8-digit hex (#RRGGBBAA) is valid CSS and invalid plotly. Alpha must be
    expressed as rgba(). This is the exact bug this file was added for."""
    fig = app.draw_subgraph(ords, ords[:4], hub)
    for trace in fig.data:
        colour = getattr(getattr(trace, "line", None), "color", None)
        if isinstance(colour, str) and colour.startswith("#"):
            assert len(colour) in (4, 7), f"{colour!r} is not plotly-legal hex"


def test_trail_trace_omitted_when_too_short(app, ords, hub):
    """A one-node trail draws no dotted path, so the trace count drops."""
    fig = app.draw_subgraph(ords, [hub], hub)
    assert len(fig.data) == 2


def test_single_node_subgraph_renders(app, hub):
    fig = app.draw_subgraph([hub], [hub], hub)
    assert fig is not None
    fig.to_plotly_json()


def test_empty_subgraph_returns_none(app):
    assert app.draw_subgraph([], [], None) is None


def test_current_node_is_emphasised(app, ords, hub):
    """The node you are standing on must be visually distinguishable."""
    fig = app.draw_subgraph(ords, ords[:4], hub)
    nodes = fig.data[-1]
    sizes = list(nodes.marker.size)
    assert sizes[ords.index(hub)] == max(sizes)
    assert sizes.count(max(sizes)) == 1


def test_every_node_gets_a_colour(app, ords, hub):
    fig = app.draw_subgraph(ords, ords[:4], hub)
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
    assert isinstance(app.load_labels("no-such-run"), dict)


# ------------------------------------------- the Walk tab renders end to end


def test_walk_tab_renders_communities_from_a_prompt():
    """Design 6.7: a prompt yields ONE Evidence expander (open, since no answer
    exists yet) holding the community graph and the ranked communities; the
    medoid cards are titled by their own salient terms (6.8). No exception."""
    from streamlit.testing.v1 import AppTest
    try:
        at = AppTest.from_file("walker_app.py", default_timeout=240)
        at.run()
        at.text_input("q").set_value("jury trial grand jury investigation").run()
    except Exception as e:                                # pragma: no cover
        pytest.skip(f"app could not start (no db?): {e}")
    errs = [e.value for e in at.exception]
    assert not errs, f"Walk tab raised: {errs}"
    assert at.get("plotly_chart"), "prompt produced no community graph"
    labels = [x.label for x in at.expander]
    ev = [l for l in labels if l.startswith("Evidence")]
    assert len(ev) == 1, f"expected exactly one Evidence expander, got {labels}"
    assert "depth" in ev[0] and "communities" in ev[0]
    md = " ".join(m.value for m in at.markdown)
    assert "Communities, most present first" in md
    assert "Local medoid" in md and "Global medoid" in md
    caps = " ".join(c.value for c in at.caption)
    assert "salient:" in caps, "medoid cards must list their salient terms"


def test_draw_communities_labels_nodes_with_terms(app):
    """The community graph is read by its TERMS, never by chunk ids."""
    touched = [{"cid": 1, "hits": 5, "size": 20}, {"cid": 2, "hits": 2, "size": 9}]
    terms = {1: ["jury", "trial", "verdict"], 2: ["congo", "belgian"]}
    fig = app.draw_communities(touched, terms, {(1, 2): 3})
    assert fig is not None
    labels = [tr for tr in fig.data if tr.mode and "text" in tr.mode][0].text
    assert list(labels) == ["jury / trial / verdict", "congo / belgian"]
    assert app.draw_communities([], {}, {}) is None


# ------------------------------------------------ labels are run-scoped


def test_labels_from_another_run_are_rejected_not_applied(app, tmp_path):
    """cid is run-local. A labels file for a different run_id must not be
    applied by cid -- it put 'early electrical science history' on the
    Moroccan elections community."""
    import json
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"run_id": "aaaa-1111",
                             "communities": [{"cid": 7, "label": "Elections"}]}),
                 encoding="utf-8")
    old = app.LABEL_FILE
    app.LABEL_FILE = f
    try:
        app.load_labels.clear()
        stale = app.load_labels("bbbb-2222")
        assert stale == {"__stale_run__": "aaaa-1111"}
        assert 7 not in stale
        good = app.load_labels("aaaa-1111")
        assert good[7]["label"] == "Elections"
    finally:
        app.LABEL_FILE = old
        app.load_labels.clear()
