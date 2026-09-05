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

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

import graph_tools as gt
from conftest import require_gt_conn

pytestmark = pytest.mark.live_db

# HUB is derived from the run the app loaded -- see the `hub` fixture; a
# constant pinned to one ingest broke on the next (ord 1476 on a 51-node run).


@pytest.fixture(scope="module")
def app():
    """Importing the app runs its script; that is how conn/run get built."""
    require_gt_conn().close()
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


def test_hovertext_shows_the_title_when_the_run_has_one(app, ords, hub, monkeypatch):
    """Spec: .spec/specs/graph-explorer/design.md sec 6.15 R22 · Task: playbook.md T6"""
    monkeypatch.setattr(app.pg_store, "node_titles", lambda conn, run_id, o=None: {ords[0]: "Battle of Midway"})
    fig = app.draw_subgraph(ords, ords[:4], hub)
    node_trace = fig.data[-1]
    idx = ords.index(ords[0])
    assert "Battle of Midway" in node_trace.hovertext[idx]
    with app.conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM node WHERE run_id = %s AND ord = %s",
                    (app.run.run_id, ords[0]))
        doc_id = cur.fetchone()["doc_id"]
    assert doc_id not in node_trace.hovertext[idx]


def test_hovertext_degrades_to_doc_id_without_titles(app, ords, hub, monkeypatch):
    """Spec: .spec/specs/graph-explorer/design.md sec 6.15 R22 · Task: playbook.md T6"""
    monkeypatch.setattr(app.pg_store, "node_titles", lambda conn, run_id, o=None: {})
    fig = app.draw_subgraph(ords, ords[:4], hub)
    payload = fig.to_plotly_json()
    assert payload["data"]
    node_trace = fig.data[-1]
    for i, o in enumerate(ords):
        with app.conn.cursor() as cur:
            cur.execute("SELECT doc_id FROM node WHERE run_id = %s AND ord = %s",
                        (app.run.run_id, o))
            doc_id = cur.fetchone()["doc_id"]
        assert doc_id in node_trace.hovertext[i]


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
    node_trace = fig.data[-1]
    assert "d=" not in node_trace.hovertext[0]         # T22: absent case degrades cleanly


def test_community_hovertext_shows_density_and_conductance(app):
    """design.md §6.17 T22: the community map hovertext gains d=/c= when the
    touched rows carry density/conductance (walk_state's merge); the map is
    the sole display surface -- PPR is not drawn here."""
    touched = [{"cid": 1, "hits": 5, "size": 20, "density": 0.036, "conductance": 0.87},
               {"cid": 2, "hits": 2, "size": 9}]
    terms = {1: ["jury", "trial", "verdict"], 2: ["congo", "belgian"]}
    fig = app.draw_communities(touched, terms, {(1, 2): 3})
    node_trace = fig.data[-1]
    assert "d=0.036" in node_trace.hovertext[0] and "c=0.870" in node_trace.hovertext[0]
    assert "d=" not in node_trace.hovertext[1]         # cid 2 has no metrics


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


def test_reason_and_judge_disagreement_is_shown():
    """One combined result (I13): the Walk tab names the chunks Reason leaned
    on that Judge called neutral (with the Judge's reason) and the ones Judge
    entailed that Reason never used. Injected; no model call."""
    from streamlit.testing.v1 import AppTest
    q = "jury trial grand jury investigation"
    rr = {"ok": True, "backend": "test", "answer": "A #11.", "briefs": [{}], "briefs_text": "b",
          "hypotheses": [], "hypothesis": "", "why": "", "foreign": [], "stages": {},
          "self_contradicting": [], "supported_ids": [11, 12], "cited": [11],
          "premises": [{"text": "p", "ids": [11, 12], "verdict": "supports", "why": ""}],
          "shown": [11, 12, 13], "entailed": [12, 13], "contradicts": [],
          "coverage": 1.0, "evidence": "",
          "verdicts": [{"ord": 11, "verdict": "neutral", "why": "about lunch counters"},
                       {"ord": 12, "verdict": "entails", "why": "yes"},
                       {"ord": 13, "verdict": "entails", "why": "yes"}]}
    try:
        at = AppTest.from_file("walker_app.py", default_timeout=240)
        at.session_state["assess"] = (q, rr)
        at.run()
        at.text_input("q").set_value(q).run()
    except Exception as e:                                # pragma: no cover
        pytest.skip(f"app could not start (no db?): {e}")
    errs = [e.value for e in at.exception]
    assert not errs, f"Walk tab raised: {errs}"
    info = " ".join(i.value for i in at.info)
    assert "Reason vs Judge" in info
    assert "#11" in info and "about lunch counters" in info      # Reason-only, with Judge's why
    assert "#13" in info                                        # Judge-only
    assert "#12" not in info                                    # agreed on: not a disagreement


def test_draw_global_map_labels_communities_by_keywords(app):
    """Map tab's prompt-independent community map: every community drawn, sized
    by member count, labelled by its keywords -- not just tables (operator,
    2026-09-01)."""
    with app.conn.cursor() as cur:
        cur.execute("""SELECT cid, size, keywords, medoid_text
                         FROM community WHERE run_id = %s ORDER BY size DESC""",
                    (app.run.run_id,))
        comms = cur.fetchall()
    fig = app.draw_global_map(comms, gt.quotient(app.conn, app.run, limit=200))
    assert fig is not None
    node_trace = fig.data[-1]
    assert len(node_trace.x) == len(comms)
    texts = list(node_trace.text)
    kw = [c for c in comms if c["keywords"]]
    assert kw and any(" / ".join(c["keywords"][:3]) in texts for c in kw)
    assert app.draw_global_map([], []) is None


def test_draw_term_graph_tristate_membership_and_cooccurrence(app):
    """Term graph (operator, 2026-09-02): terms are nodes; colour groups carry
    the tri-state provenance (walk-only / global-only / both); edges need >=2
    shared walked chunks."""
    cond = {0: ["carrier", "torpedo"], 1: ["midway"]}
    unsup = {0: ["carrier", "aircraft"], 1: ["navy"]}
    members = {"carrier": {1, 2, 3}, "torpedo": {2, 3}, "midway": {3},
               "aircraft": {1, 2}, "navy": {9}}
    fig = app.draw_term_graph(cond, unsup, members)
    assert fig is not None
    named = {tr.name: list(tr.text) for tr in fig.data if tr.name}
    assert named["both sets"] == ["carrier"]
    assert "torpedo" in named["walk (prompt-conditioned BM25)"]
    assert "midway" in named["walk (prompt-conditioned BM25)"]
    assert named["global concept"] == ["aircraft", "navy"]
    edge_traces = [tr for tr in fig.data if not tr.name]
    # carrier-torpedo share {2,3}, carrier-aircraft share {1,2}, torpedo-aircraft {2}
    assert len(edge_traces) == 2
    assert app.draw_term_graph({}, {}, {}) is None


def test_draw_layers3d_separates_planes_and_bridges_them(app):
    """3D layers view: chunk markers all at z=0, term markers all at z=1,
    cross-layer membership lines spanning 0->1, chain backbones present."""
    state = {
        "chunks": {"pos": {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, 1.0)},
                   "backbone": [(1, 2)], "sig": [(1, 2, 0.9), (2, 3, 0.5)],
                   "chain_of": {1: 0, 2: 0, 3: 1}, "chains": [[1, 2], [3]]},
        "terms": {"pos": {"war": (0.0, 0.5), "navy": (0.8, 0.2)},
                  "backbone": [("war", "navy")], "sig": [("war", "navy", 0.7)],
                  "chain_of": {"war": 0, "navy": 0}, "chains": [["war", "navy"]]},
        "kept": [1, 2, 3],
        "cross": [(1, "war", 2.0), (2, "navy", 0.4), (3, "war", 1.1)],
    }
    fig = app.draw_layers3d(state, {1: 0, 2: 0, 3: 1},
                            {1: "wiki", 2: "brown", 3: None})
    named = {tr.name: tr for tr in fig.data if tr.name}
    assert set(named) == {"chunks", "terms"}
    assert set(named["chunks"].z) == {0.0}
    assert set(named["terms"].z) == {1.0}
    cross = [tr for tr in fig.data
             if not tr.name and tr.z is not None
             and {v for v in tr.z if v is not None} == {0.0, 1.0}]
    assert cross, "no cross-layer trace spanning both planes"


def test_mirror_is_the_real_neo4j_browser_under_map():
    """T26 (final): Mirror is the ACTUAL neo4j browser iframed inside the Map
    tab -- possible because the container is launched with a CSP whose
    frame-ancestors names http://localhost:8501 (stock image sends DENY).
    Preceded by the dendrite-sorted Partitions table; one input field."""
    src = open("walker_app.py", encoding="utf-8").read()
    assert "tab_mirror" not in src                      # no third tab
    mirror = src.split("MIRROR (T26)")[1]
    assert "components.iframe" in mirror
    # T32: the browser URL comes from config (single source), not a literal
    import config as _cfg
    assert "config.NEO4J_BROWSER" in mirror
    assert _cfg.NEO4J_BROWSER.startswith("http://localhost:7474/browser/")
    assert 'selectbox("Walk"' not in mirror             # one input field
    assert "st.radio" not in mirror
    # partitions live at the TOP of the Map tab now, before the community map
    assert "Partitions (dendrite sort)" in src
    assert src.find("Partitions (dendrite sort)") < src.find('subheader("Community map")')
