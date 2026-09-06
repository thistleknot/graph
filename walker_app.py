"""
walker_app.py — interactive graph walker over a persisted ChunkGraph run.

Two ways into the same graph, all sharing one deterministic tool surface
(`graph_tools`): read-only at the server, run-scoped, traversing the indexed
src/dst columns only.

    Walk       type a prompt, get the walked graph. sampler.ef_evidence()
               expands from BM25 anchors and stops itself (HNSW rule); the
               result set is drawn coloured by stored cid. Nothing else.
    Map        the community quotient graph and draft labels

Nothing here re-partitions anything. Communities are the run's own stored cids.

Run (PowerShell, from the repo root; needs the `chunkgraph-pg` container,
which normally stays up -- `docker compose up -d` only if it is not):
    $env:CHUNKGRAPH_MODEL_DIR = 'C:/Users/user/models/m2v-minilm-l6-256'   # overrides the default ~/models/m2v-minilm-l6-256
    $env:OPENROUTER_API_KEY   = '...'                                       # Reason / Judge
    streamlit run walker_app.py --server.port 8501
Then open http://localhost:8501. Ctrl+C in that terminal stops it.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components

import config
import evidence
import graph_tools as gt
import interpret
import pg_store
import react
import sampler
import walker_core

st.set_page_config(page_title="ChunkGraph Walker", layout="wide")

PALETTE = walker_core.PALETTE
cid_color = walker_core.cid_color
group_color = walker_core.group_color
rgba = walker_core.rgba
PRIMARY, GOOD, BAD, WARN, MUTED = (walker_core.PRIMARY, walker_core.GOOD,
                                   walker_core.BAD, walker_core.WARN, walker_core.MUTED)
PROV_COLOR = {"both": "#E45756", "dense": "#4C78A8", "sparse": "#9E9E9E"}
LABEL_FILE = Path(os.environ.get("LABEL_OUT", "community_labels.json"))
DEFAULT_MODEL_DIR = config.MODEL_DIR   # used when CHUNKGRAPH_MODEL_DIR is unset

ACCENT = {"Answer": GOOD, "Judged evidence": PRIMARY,
          "Groups": "#B279A2", "Partitions": WARN}   # P14(c), P15(f)


def panel_head(title, sub=""):
    """Accent bar + section title, rendered inside a bordered panel (P14(c))."""
    c = ACCENT[title]
    st.markdown(
        f'<div style="border-left:4px solid {c};padding-left:.6rem;margin:.1rem 0 .5rem">'
        f'<div style="font-size:1.25rem;font-weight:700;color:{c}">{html.escape(title)}</div>'
        + (f'<div style="opacity:.7;font-size:.85rem">{html.escape(sub)}</div>' if sub else "")
        + "</div>", unsafe_allow_html=True)


def dark(fig):
    """P15(a): one dark treatment for every plotly figure -- plotly_dark template
    with fully transparent paper/plot grounds so the panel card shows through.
    Applied at each draw_* return so a direct caller (tests) and the app get the
    identical figure. `None` passes through, since every draw_* may return None.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P15(a)
    Task: playbook.md T50
    """
    if fig is None:
        return None
    fig.update_layout(template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e6e8ee"))
    return fig


@st.cache_resource
def get_conn():
    return gt.connect()


@st.cache_resource
def get_embed(model_dir: str | None):
    """model2vec static embedder for query-conditioned terms (design 6.1).
    None when no model dir: ranking degrades to lexical-then-unsupervised."""
    return evidence.load_embed(model_dir)


@st.cache_data(ttl=30)
def load_labels(run_id: str) -> dict:
    """Draft labels, if label_communities.py has been run. Absent is normal.

    cid is RUN-LOCAL (steering): a re-ingest mints new communities under the
    same numbers. A labels file drafted against another run is not merely
    stale, it is wrong -- it put "early electrical science history" on the
    Moroccan elections. Labels apply only when the file names THIS run."""
    return walker_core.load_labels(LABEL_FILE, run_id)


def cid_badge(cid, labels) -> str:
    """A cid always shows as a cid. A draft label is additive and marked."""
    if cid is None:
        return "no community"
    lab = labels.get(cid)
    return f"c{cid} · *{lab['label']}*" if lab else f"c{cid}"


conn = get_conn()

# ------------------------------------------------- top row (P17: no sidebar)
runs = gt.list_runs(conn)
live = [r for r in runs if r["superseded_at"] is None]
if not live:
    st.error("No live runs in the database.")
    st.stop()

_c_title, _c_run, _c_stats, _c_details = st.columns([2.2, 1.6, 3.2, 1.0],
                                                    vertical_alignment="bottom")
with _c_title:
    st.markdown(f'<div style="font-size:1.35rem;font-weight:700;'
                f'color:{PRIMARY}">ChunkGraph Walker</div>',
                unsafe_allow_html=True)
with _c_run:
    label = st.selectbox("Run", [r["label"] for r in live],
                         label_visibility="collapsed")
run = gt.get_run(conn, label)
labels = load_labels(str(run.run_id))
with _c_stats:
    st.caption(f"{run.n_chunks} chunks · {run.n_edges} edges · "
               f"{run.n_communities} communities")

embed = get_embed(config.MODEL_DIR)
if "__stale_run__" in labels:
    labels = {}                      # written for another run; cid is run-local

with _c_details:
    with st.popover("details"):
        st.caption(f"run `{run.run_id}`")
        st.caption("retrieval: " + (f"fused, embed_dim {run.embed_dim}" if run.dense else "sparse-only"))
        st.caption("query terms: " + ("lexical + dense" if embed else "lexical only"))
        if run.single_provenance:
            st.caption(f"every edge is **{run.single_provenance}** "
                       f"({run.provenance[run.single_provenance]}); nothing here is fused")
        st.caption(f"draft labels: {len(labels)} loaded (italic = model-authored)"
                   if labels else "draft labels: none for this run")
        mix = gt.run_sources(conn, run)
        if mix:
            st.caption("sources: " + gt.format_source_mix(mix))

# Only a degraded signal is worth an inline line of its own (P17).
if not run.dense:
    st.warning("Sparse-only run: anchors are lexical, no dense edges.")
if embed is None:
    st.warning("No embedding model found: query-conditioned terms are "
               "lexical only. Set CHUNKGRAPH_MODEL_DIR.")

def draw_subgraph(ords, trail, current=None, height=340):
    """Shared renderer: nodes coloured by STORED cid, trail dotted."""
    if not ords:
        return None
    edges = gt.subgraph_edges(conn, run, ords)
    G = nx.Graph()
    G.add_nodes_from(ords)
    for e in edges:
        G.add_edge(e["src"], e["dst"], weight=max(e["strength"], 0.01))
    pos = nx.spring_layout(G, seed=7, k=0.9, iterations=120)
    meta = {o: gt.node(conn, run, o) for o in ords}
    titles = pg_store.node_titles(conn, run.run_id, ords)                      # R22

    ex, ey = [], []
    for e in edges:
        x0, y0 = pos[e["src"]]
        x1, y1 = pos[e["dst"]]
        ex += [x0, x1, None]
        ey += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                             line=dict(color="rgba(136,136,136,0.4)", width=1)))
    if len(trail) > 1:
        fig.add_trace(go.Scatter(
            x=[pos[o][0] for o in trail if o in pos],
            y=[pos[o][1] for o in trail if o in pos],
            mode="lines", hoverinfo="skip",
            line=dict(color="#E45756", width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=[pos[o][0] for o in ords], y=[pos[o][1] for o in ords],
        mode="markers+text", text=[str(o) for o in ords],
        textposition="top center", textfont=dict(size=9),
        marker=dict(
            size=[26 if o == current else 15 for o in ords],
            color=[cid_color(meta[o]["cid"]) for o in ords],
            line=dict(width=[3 if o == current else 1 for o in ords],
                      color="#0f1117")),
        hovertext=[f"#{o} · {interpret.chunk_label(meta[o], titles)} · {cid_badge(meta[o]['cid'], labels)}"
                   f"<br>{meta[o]['body'][:120]}…" for o in ords],
        hoverinfo="text"))
    fig.update_layout(showlegend=False, height=height,
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return dark(fig)


def community_panel(ords):
    touched = gt.communities_touched(conn, run, ords)
    st.caption(f"{len(ords)} chunks · **{len(touched)} of {run.n_communities} "
               f"communities** touched")
    for t in touched:
        lab = labels.get(t["cid"])
        name = f" — *{lab['label']}*" if lab else ""
        src = f" · {gt.format_source_mix(t['sources'])}" if t.get("sources") else ""
        st.markdown(
            f"<span style='color:{cid_color(t['cid'])}'>●</span> "
            f"**c{t['cid']}**{name} · {t['hits']} of {t['size']} members · "
            f"{', '.join(t['keywords'][:4])}{src}", unsafe_allow_html=True)
    return touched


def draw_communities(touched, terms, xedges, height=520):
    """Community-level view of a walk: nodes are the communities present,
    sized by presence, LABELLED BY THEIR TOP TERMS. Chunks are never drawn
    here -- the graph is read by its terms (design §6)."""
    if not touched:
        return None
    G = nx.Graph()
    for t in touched:
        G.add_node(t["cid"], hits=t["hits"], size=t["size"])
    for (a, b), w in xedges.items():
        G.add_edge(a, b, weight=w)
    pos = nx.spring_layout(G, weight="weight", seed=7, k=1.6)
    ex, ey = [], []
    for a, b in G.edges():
        ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure()
    if ex:
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                 line=dict(color="rgba(160,160,175,0.35)", width=1.2),
                                 hoverinfo="none", showlegend=False))
    mx = max(t["hits"] for t in touched)

    def _hover(t):
        s = f"c{t['cid']} · {t['hits']} of {t['size']}"
        if t.get("density") is not None:
            s += f" · d={t['density']:.3f} c={t['conductance']:.3f}"
        return s

    fig.add_trace(go.Scatter(
        x=[pos[t["cid"]][0] for t in touched],
        y=[pos[t["cid"]][1] for t in touched],
        mode="markers+text",
        text=[" / ".join(terms.get(t["cid"], [])[:3]) for t in touched],
        textposition="top center",
        textfont=dict(size=12),
        marker=dict(size=[14 + 40 * t["hits"] / mx for t in touched],
                    color=[cid_color(t["cid"]) for t in touched],
                    line=dict(color="#0f1117", width=1)),
        hovertext=[_hover(t) for t in touched],
        hoverinfo="text", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return dark(fig)


def draw_term_graph(terms_cond: dict, terms_unsup: dict, members: dict, height=560):
    """TERM graph of the walk (operator, 2026-09-02): nodes are TERMS, not
    chunks. Membership is provenance, the repo's tri-state applied to
    vocabulary -- a term can belong to the subgraph's prompt-conditioned BM25
    set (walk), the unsupervised global concept set (global), or both.
    Edges = co-occurrence within the walked chunks (>=2 shared chunks).
    Deterministic: spring seed 7 over co-occurrence weights."""
    cond = {t for ts in terms_cond.values() for t in ts}
    unsup = {t for ts in terms_unsup.values() for t in ts}
    drawn = sorted((cond | unsup) & set(members))
    if not drawn:
        return None
    G = nx.Graph()
    G.add_nodes_from(drawn)
    for i, a in enumerate(drawn):
        for b in drawn[i + 1:]:
            w = len(members[a] & members[b])
            if w >= 2:
                G.add_edge(a, b, weight=w)
    pos = nx.spring_layout(G, weight="weight", seed=7, k=1.2)
    wmax = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    fig = go.Figure()
    for a, b, d in G.edges(data=True):
        fig.add_trace(go.Scatter(
            x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
            line=dict(color="rgba(160,160,175,0.35)", width=0.6 + 3.5 * d["weight"] / wmax),
            hoverinfo="none", showlegend=False))
    groups = (("both sets", [t for t in drawn if t in cond and t in unsup], "#B279A2"),
              ("walk (prompt-conditioned BM25)", [t for t in drawn if t in cond and t not in unsup], "#E45756"),
              ("global concept", [t for t in drawn if t in unsup and t not in cond], "#4C78A8"))
    dfmax = max((len(members[t]) for t in drawn), default=1)
    for name, ts, color in groups:
        if not ts:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[t][0] for t in ts], y=[pos[t][1] for t in ts],
            mode="markers+text", name=name,
            text=ts, textposition="top center", textfont=dict(size=11),
            marker=dict(size=[10 + 26 * len(members[t]) / dfmax for t in ts],
                        color=color, line=dict(color="#0f1117", width=1)),
            hovertext=[f"{t} · in {len(members[t])} walked chunks" for t in ts],
            hoverinfo="text", showlegend=True))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.02))
    return dark(fig)


def plane(out):
    """Spring-lays-out one dendrite_sort plane (chunk or term). Closure-free
    (reads only its argument and module-level `nx`) -- lifted out of the old
    `_dendrite_state` unchanged. Layout-only: the digest never reads pos/
    backbone/chain_of, only `dendrite_sort`'s own `names`/`r`/`sig`/`chains`,
    so no spring layout touches the byte-pinned digest (design 6.21(a))."""
    G = nx.Graph()
    G.add_nodes_from(out["names"])
    idx = {n_: i for i, n_ in enumerate(out["names"])}
    for a in out["names"]:
        for b in out["names"]:
            if a < b and out["sig"][idx[a], idx[b]]:
                G.add_edge(a, b, weight=abs(out["r"][idx[a], idx[b]]))
    pos = nx.spring_layout(G, weight="weight", seed=7, k=1.3)
    backbone = [(c[i], c[i + 1]) for c in out["chains"] for i in range(len(c) - 1)]
    sig_edges = [(a, b, G[a][b]["weight"]) for a, b in G.edges()]
    chain_of = {n_: ci for ci, c in enumerate(out["chains"]) for n_ in c}
    return {"pos": pos, "backbone": backbone, "sig": sig_edges,
            "chain_of": chain_of, "chains": out["chains"]}


@st.cache_data(show_spinner=False, max_entries=16)
def dendrite_state(run_id: str, q: str, _ev=None):
    """Layout wrapper over Evidence.dendrite (raw dendrite_sort output);
    cache wrapper keyed by (run, prompt). None when the walk kept <5
    embeddings. Output shape is byte-for-byte what the old _dendrite_state
    returned, so the Map partitions block (walker_core.partition_rows) is
    untouched by the split."""
    d = _ev.dendrite
    if d is None:
        return None
    return {"chunks": plane(d["chunks"]), "terms": plane(d["terms"]),
            "kept": d["kept"], "cross": d["cross"], "sal": d["sal"]}


def draw_global_map(comms, qrows, height=560):
    """Whole-run community map for the Map tab: every community, sized by
    member count, LABELLED BY ITS KEYWORDS, edges weighted by inter-community
    edge counts. Prompt-independent twin of draw_communities (design §6:
    the graph is read by its terms)."""
    if not comms:
        return None
    G = nx.Graph()
    for c in comms:
        G.add_node(c["cid"], size=c["size"])
    for r in qrows:
        G.add_edge(r["cid_a"], r["cid_b"], weight=r["edges"])
    pos = nx.spring_layout(G, weight="weight", seed=7, k=1.6)
    wmax = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    fig = go.Figure()
    for a, b, d in G.edges(data=True):
        fig.add_trace(go.Scatter(
            x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
            line=dict(color="rgba(160,160,175,0.35)", width=0.8 + 4.0 * d["weight"] / wmax),
            hoverinfo="none", showlegend=False))
    smax = max(c["size"] for c in comms)
    fig.add_trace(go.Scatter(
        x=[pos[c["cid"]][0] for c in comms],
        y=[pos[c["cid"]][1] for c in comms],
        mode="markers+text",
        text=[" / ".join((c["keywords"] or [])[:3]) for c in comms],
        textposition="top center",
        textfont=dict(size=11),
        marker=dict(size=[14 + 44 * c["size"] / smax for c in comms],
                    color=[cid_color(c["cid"]) for c in comms],
                    line=dict(color="#0f1117", width=1)),
        hovertext=[f"c{c['cid']} · {c['size']} chunks · "
                   + ", ".join((c["keywords"] or [])[:5]) for c in comms],
        hoverinfo="text", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return dark(fig)


def _monotone_chain_hull(points):
    """Pure-python convex hull (Andrew's monotone chain), no scipy dependency."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _hull_shapes(pos, groups, pal, alpha=0.12) -> list:
    """P6 group-similarity shading as `layout.shapes` -- never a filled trace,
    so every existing test that indexes `fig.data[-1]` / counts traces is
    untouched by hulls (they live in `layout`, not `data`)."""
    by_group: dict = {}
    for o, gid in groups.items():
        if o in pos:
            by_group.setdefault(gid, []).append(o)

    shapes = []
    for gid, members in by_group.items():
        if not members:
            continue
        color = pal.get(gid, "#DDDDDD")
        pts = [(float(pos[o][0]), float(pos[o][1])) for o in members]
        if len(pts) >= 3:
            hull = _monotone_chain_hull(pts)
            cx = sum(x for x, _ in hull) / len(hull)
            cy = sum(y for _, y in hull) / len(hull)
            padded = [(cx + (x - cx) * 1.08, cy + (y - cy) * 1.08) for x, y in hull]
            path = "M " + " L ".join(f"{x},{y}" for x, y in padded) + " Z"
            shapes.append({"type": "path", "path": path,
                           "fillcolor": rgba(color, alpha), "line": {"width": 0},
                           "layer": "below", "xref": "x", "yref": "y"})
        else:
            xs = [x for x, _ in pts]; ys = [y for _, y in pts]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) if len(pts) > 1 else 1.0
            r = 0.06 * span
            cx = sum(xs) / len(xs); cy = sum(ys) / len(ys)
            shapes.append({"type": "circle", "x0": cx - r, "y0": cy - r,
                           "x1": cx + r, "y1": cy + r,
                           "fillcolor": rgba(color, alpha), "line": {"width": 0},
                           "layer": "below", "xref": "x", "yref": "y"})
    return shapes


def draw_group_graph(ords, edges, groups, pal, sal=None, height=420):
    """P9 per-panel relational figure: spring layout over the walked subgraph,
    coloured by `groups` ({ord: gid}), hulls shaded behind by group. `edges`
    is `evidence.analysis_inputs()["edges"]` -- no DB call here."""
    if not ords:
        return None
    G = nx.Graph()
    G.add_nodes_from(ords)
    for (a, b), w in edges.items():
        G.add_edge(a, b, weight=w)
    pos = nx.spring_layout(G, weight="weight", seed=7, k=0.9, iterations=120)

    ex, ey = [], []
    for a, b in G.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]

    def _hover(o):
        base = f"#{o} · g{groups.get(o)}"
        if sal and sal.get(o, {}).get("top"):
            base += " · " + "/".join(sal[o]["top"][:2])
        return base

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                             line=dict(color="rgba(136,136,136,0.4)", width=1)))
    fig.add_trace(go.Scatter(
        x=[pos[o][0] for o in ords], y=[pos[o][1] for o in ords],
        mode="markers+text", text=[str(o) for o in ords],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=15, color=[pal.get(groups.get(o)) for o in ords],
                    line=dict(width=1, color="#0f1117")),
        hovertext=[_hover(o) for o in ords], hoverinfo="text"))
    fig.update_layout(shapes=_hull_shapes(pos, groups, pal), showlegend=False,
                      height=height, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return dark(fig)


tab_analysis, tab_neo4j = st.tabs(["Analysis", "Neo4j"])

@st.cache_data(show_spinner=False, max_entries=32)
def walk_for(run_id: str, q: str):
    """The walk itself, once per (run, prompt). Deterministic (S1): same inputs,
    same bundle, so a rerun on a button click reads it back instead of walking."""
    return sampler.ef_evidence(conn, run, q)


@st.cache_data(show_spinner=False, max_entries=32)
def walk_state(run_id: str, q: str, _bnd=None, _embed=None):
    """Everything the tab shows that is computed, not model-authored: keyed by
    (run, prompt); the underscore args are inputs Streamlit must not hash.
    Thin wrapper over evidence.assemble (design 6.21(a))."""
    return evidence.assemble(conn, run, _bnd, embed=_embed)


@st.cache_data(show_spinner=False, max_entries=32)
def analysis_for(run_id: str, q: str, _ords=None):
    """P7: the one new DB touch point for the Groups panels, cached per
    (run, prompt, ords) so a rerun never re-queries."""
    return evidence.analysis_inputs(conn, run, _ords)


@st.cache_data(show_spinner=False, max_entries=8)
def walk_umap_for(run_id: str, q: str, _ords=None):
    """P16(i): the UMAP projection of the walked chunks, once per (run, prompt).
    None on sparse-only runs or when any walked chunk lacks an embedding."""
    return evidence.walk_umap(conn, run, _ords)


@st.cache_data(show_spinner=False, max_entries=16)
def louvain_for(run_id: str, q: str, _ords=None, _edges=None):
    """P4 LEFT panel: ephemeral louvain re-run on the walked subgraph only."""
    return walker_core.subgraph_louvain(_ords, _edges)


@st.cache_data(show_spinner=False, max_entries=16)
def assess_for(run_id: str, q: str, _bnd=None, _ws=None, _ds=None, _embed=None):
    """P8: the one model call, cached per (run, prompt) so the auto-fire never
    re-fires on an unrelated rerun."""
    return walker_core.assess(conn, run, _bnd, _ws, _ds, q, embed=_embed)


@st.cache_data(show_spinner=False, max_entries=8)
def react_for(run_id: str, q: str, _bnd=None, _ws=None, _rr=None, _embed=None):
    """A7: only ever called from the insufficient branch. Iteration 0 is the
    walk and the judge result the page ALREADY has (D3) -- injected through
    react's walk_fn/judge_fn seams so the loop costs one walk + one model call
    per EXTRA iteration and nothing for the base.

    Spec: design.md 6.23 A1-A9 · Task: playbook.md T62
    """
    used = {"walk": False, "judge": False}

    def _walk(q_, params):
        if not used["walk"] and q_ == q and params == react.BASE_PARAMS:
            used["walk"] = True
            return _bnd, _ws
        bnd_, _tele = sampler.ef_evidence(conn, run, q_, seed=0, **params)
        return bnd_, evidence.assemble(conn, run, bnd_, embed=_embed)

    def _judge(bnd_, ev_):
        if not used["judge"]:
            used["judge"] = True
            return _rr
        return interpret.reason(conn, run, bnd_, ev_.terms, ev_.concept,
                                embed=_embed, judge=True, pw=ev_.pathways,
                                digest=ev_.digest)

    return react.run(conn, run, q, embed=_embed, walk_fn=_walk, judge_fn=_judge)


# ================================================================ ANALYSIS
with tab_analysis:
    # P13: a fresh session seeds the prompt from ?q=, and a running walk writes
    # it back, so the current view is a shareable link (state itself is cached
    # process-wide per (run, prompt) -- the second browser renders from cache).
    if "q" not in st.session_state and st.query_params.get("q"):
        st.session_state["q"] = st.query_params["q"]
    q = st.text_input("Prompt", "", key="q", placeholder="ask the corpus")
    if q.strip() and st.query_params.get("q") != q:
        st.query_params["q"] = q
    if q.strip():
        bnd, tele = walk_for(str(run.run_id), q)
        if not bnd.sampled:
            st.warning("No lexical anchor matched. Nothing to walk from.")
        else:
            # ---- everything below is computed before any model call (S6),
            # once per (run, prompt): a button click reruns the script and
            # must not redo the walk.
            ws = walk_state(str(run.run_id), q, _bnd=bnd, _embed=embed)
            touched, cids, concept, terms = ws.touched, ws.cids, ws.concept, ws.terms
            cid_of, in_cid, xedges, medoids, salient = (ws.cid_of, ws.in_cid, ws.xedges,
                                                          ws.medoids, ws.salient)
            src_of_all = walker_core.src_of_map(conn, run, bnd.sampled)   # one DB loop, 3 readers
            pw = ws.pathways
            ai = analysis_for(str(run.run_id), q, _ords=bnd.sampled)
            ndw = walker_core.node_dwpc(pw)
            rel_groups = louvain_for(str(run.run_id), q, _ords=bnd.sampled, _edges=ai["edges"])
            glob_groups = {o: ws.cid_of[o] for o in bnd.sampled if ws.cid_of.get(o) is not None}
            org = getattr(bnd, "origin", {}) or {}
            n_br = sum(1 for v in org.values() if v == "bridge")
            def _mark(o):
                return f"**#{o}**" if org.get(o) == "bridge" else f"#{o}"
            ds = dendrite_state(str(run.run_id), q, _ev=ws)

            # ---- 1. the model's answer, right under the prompt (I13: one call,
            # P8: auto-fired -- no button, a walk always ends in an answer)
            got = st.session_state.get("assess")
            if not (got and got[0] == q):
                with st.spinner("one call: reasoning over briefs + judging every chunk …"):
                    try:
                        ds_ = dendrite_state(str(run.run_id), q, _ev=ws)
                        rr_, mirror_err = assess_for(str(run.run_id), q, _bnd=bnd, _ws=ws,
                                                     _ds=ds_, _embed=embed)
                    except Exception as e:                                # noqa: BLE001
                        rr_, mirror_err = {"ok": False, "error": str(e), "hypotheses": [],
                                           "premises": [], "foreign": [], "self_contradicting": [],
                                           "briefs_text": "", "stages": {}}, None
                    st.session_state["assess"] = (q, rr_)
                    got = st.session_state["assess"]
                    if mirror_err:
                        st.warning(f"neo4j mirror skipped: {mirror_err}")

            # ---- P15(b) KPI stat row: every number already in scope, no new
            # query (P7). Tiles 3/4 degrade gracefully when the model call
            # failed or has not produced an "ok" result yet.
            rr_stat = got[1] if (got and got[0] == q and got[1].get("ok")) else None
            _tiles = [
                ("🧩", "chunks walked", len(bnd.sampled),
                 f"depth {tele['depth']} · {tele['stop']}", PRIMARY),
                ("🫧", "communities", f"{len(touched)}/{run.n_communities}",
                 f"{len(cids)} in the walk", "#B279A2"),
                ("✅", "judged",
                 f"{len(rr_stat['entailed'])} / {len(rr_stat['contradicts'])}"
                 if rr_stat else "—",
                 "entail / contradict" if rr_stat else "no answer yet",
                 (GOOD if not rr_stat["contradicts"] else BAD) if rr_stat else MUTED),
                ("🧠", "model", rr_stat["backend"] if rr_stat else "—",
                 (f"{len(rr_stat['briefs'])} briefs · "
                  f"{len(rr_stat.get('shown', []))} judged") if rr_stat else "no answer yet",
                 PRIMARY),
            ]
            _c1, _c2, _c3, _c4 = st.columns(4)
            for _col, _args in zip((_c1, _c2, _c3, _c4), _tiles):
                with _col:
                    st.markdown(walker_core.stat_card(*_args), unsafe_allow_html=True)

            if got and got[0] == q:
                rr = got[1]
                _entails = len(rr.get("entailed") or []) if rr.get("ok") else 0
                rx = None
                if rr.get("ok") and _entails == 0:      # A7: insufficient path ONLY
                    with st.spinner("0 entails -- agentic retrieval: re-walking "
                                    "with new parameters …"):
                        try:
                            rx = react_for(str(run.run_id), q, _bnd=bnd, _ws=ws,
                                           _rr=rr, _embed=embed)
                        except Exception as e:                        # noqa: BLE001
                            rx = None
                            st.warning(f"agentic retrieval skipped: {e}")
                _z1L, _z1R = st.columns([1.1, 1])          # P18 Zone 1: VERDICT | EVIDENCE
                with _z1L:
                    with st.container(border=True):            # P10 panel 1 — Answer
                        panel_head("Answer")
                        if not rr["ok"]:
                            st.warning(f"Stopped: {rr['error']}")
                        else:
                            _model_line = (
                                f"{rr['backend']} · one call: {len(rr['briefs'])} "
                                f"community briefs + {len(rr.get('shown', []))} chunks judged"
                                + (f" · {rr['structure_note']}" if rr.get("structure_note") else ""))
                            _gated, _hero_text = walker_core.answer_gate(
                                rr["answer"] or "No premise was judged supported, so there is "
                                                "nothing to answer from. The premises below say why.",
                                _entails,
                                n_iters=(rx or {}).get("n_iters", 0),
                                n_chunks=len(bnd.sampled),
                                found_entails=len((rx or {}).get("entails", [])))
                            st.markdown(walker_core.hero_answer(
                                _hero_text,
                                [] if _gated else (rr.get("cited") or []),
                                _model_line,
                                color=WARN if _gated else PRIMARY),
                                unsafe_allow_html=True)
                        if rr["hypotheses"]:
                            st.markdown("**Hypothesis** " + (rr["hypothesis"] or ""))
                            others = [h for h in rr["hypotheses"] if h != rr["hypothesis"]]
                            if others:
                                st.caption("also considered: " + " | ".join(others))
                            if rr["why"]:
                                st.caption(f"chosen because: {rr['why']}")
                        if rr["premises"]:
                            st.markdown("**Premises**")
                            col = {"supports": GOOD, "contradicts": BAD,
                                   "insufficient": MUTED, "unsupported": "#666a78"}
                            for pr in rr["premises"]:
                                ids = ", ".join(f"#{o}" for o in pr["ids"]) or "no evidence cited"
                                st.markdown(
                                    f"<span style='color:{col[pr['verdict']]}'>●</span> "
                                    f"**{pr['verdict']}** — {pr['text']}<br>"
                                    f"<span style='opacity:.7;font-size:.86em'>{ids}"
                                    f"{' — ' + pr['why'] if pr['why'] else ''}</span>",
                                    unsafe_allow_html=True)
                        if rr["foreign"]:
                            st.error("Foreign ids named by the model (discarded): "
                                     + ", ".join(f"#{o}" for o in rr["foreign"]))
                        if rr["self_contradicting"]:
                            st.error("Answer cites ids outside the supported premises: "
                                     + ", ".join(f"#{o}" for o in rr["self_contradicting"]))

                with _z1R:
                    with st.container(border=True):            # P10 panel 2 — Judged evidence
                        # ---- the judge channel of the same call
                        if rr.get("verdicts"):
                            panel_head("Judged evidence")
                            st.caption(f"judged {rr['coverage']:.0%} of {len(rr['shown'])} shown · "
                                       f"{len(rr['entailed'])} entail · "
                                       f"{len(rr['contradicts'])} contradict")
                            why = {v["ord"]: v["why"] for v in rr["verdicts"]}
                            for label, ords_, verdict_word, colour in (
                                    ("Entails", rr["entailed"], "entails", GOOD),
                                    ("Contradicts", rr["contradicts"], "contradicts", BAD)):
                                if ords_:
                                    st.markdown(f"**{label}**")
                                    for o in ords_:
                                        nd = gt.node(conn, run, o)
                                        badge = walker_core.pill(
                                            f"{verdict_word} · walk {bnd.scores.get(o, 0):.2f}", colour)
                                        head = (f"#{o} · {nd['doc_id']} · c{nd['cid']} — "
                                                f"{why.get(o, '')}")
                                        snippet = walker_core.clip(
                                            interpret.excerpt(nd['body'], q, 300, embed), 300)
                                        st.markdown(walker_core.evidence_row(
                                            head, snippet, badge, cid_color(nd['cid'])),
                                            unsafe_allow_html=True)
                            neutral = [v for v in rr["verdicts"]
                                       if v["verdict"] == "neutral" and v["ord"] in set(rr["shown"])]
                            if neutral:
                                with st.expander(f"Neutral ({len(neutral)}) — the model's reason for each"):
                                    for v in neutral:
                                        nd = gt.node(conn, run, v["ord"])
                                        badge = walker_core.pill("neutral", MUTED)
                                        head = f"#{v['ord']} · {nd['doc_id']} · c{nd['cid']} — {v['why']}"
                                        st.markdown(walker_core.evidence_row(
                                            head, "", badge, cid_color(nd['cid'])),
                                            unsafe_allow_html=True)

                            # ---- where the two channels disagree (Judge is stricter)
                            sup, ent = set(rr["supported_ids"]), set(rr["entailed"])
                            reason_only = sorted(sup - ent)
                            judge_only = sorted(ent - sup)
                            if reason_only or judge_only:
                                lines = ["**Reason vs Judge.** Reason argues from community briefs and "
                                         "accepts a chunk that supports a premise; Judge asks whether a "
                                         "chunk literally answers the prompt. When they disagree, trust "
                                         "Judge for *what the corpus says* and Reason for *how it hangs "
                                         "together*."]
                                if reason_only:
                                    lines.append("Reason leaned on, Judge called neutral: " + "; ".join(
                                        f"#{o} — {why.get(o, 'not judged')}" for o in reason_only))
                                if judge_only:
                                    lines.append("Judge found entailing, Reason never used: " + ", ".join(
                                        f"#{o}" for o in judge_only))
                                st.info(("  " + chr(10)).join(lines))

                        if rr["briefs_text"]:
                            with st.expander("Briefs + structure the model reasoned over"):
                                st.code(rr["briefs_text"]
                                        + ("\n\n" + rr["structure"] if rr.get("structure") else ""),
                                        language="text")
                        if rr.get("evidence"):
                            with st.expander("Exactly what the model was shown (judge channel)"):
                                st.code(rr["evidence"], language="text")
                        if rr["stages"]:
                            with st.expander("Exactly what the model returned (raw)"):
                                for name, txt in rr["stages"].items():
                                    st.markdown(f"**{name}**"); st.code(txt, language="json")

                        # ---- A1/A8/A9 iteration transcript. P18: this is
                        # EVIDENCE (why believe it), so it lands in Zone 1
                        # right, additive to the base walk's judged rows.
                        if rx:
                            panel_head("Agentic retrieval")
                            st.caption(
                                f"{rx['n_iters']} iterations beyond the base walk · "
                                f"stop: {rx['stop_reason']} · union {len(rx['ords'])} chunks · "
                                f"{len(rx['entails'])} entail / {len(rx['contradicts'])} contradict")
                            st.code(rx["history_table"], language="text")
                            _new_ent = [o for o in rx["entails"]
                                        if rx["found_at"].get(o, 0) > 0]
                            if _new_ent:
                                st.markdown("**Found only by the loop**")
                                for o in _new_ent[:12]:
                                    nd = gt.node(conn, run, o)
                                    badge = walker_core.pill(
                                        f"entails · iteration {rx['found_at'][o]}", GOOD)
                                    head = (f"#{o} · {nd['doc_id']} · c{nd['cid']} — "
                                            f"{rx['verdicts'].get(o, {}).get('why', '')}")
                                    snippet = walker_core.clip(
                                        interpret.excerpt(nd['body'], q, 300, embed), 300)
                                    st.markdown(walker_core.evidence_row(
                                        head, snippet, badge, cid_color(nd['cid'])),
                                        unsafe_allow_html=True)
                            elif rx["entails"]:
                                st.caption("the loop re-confirmed the base walk's "
                                           "entails and found no new ones")
                            if rx.get("missing"):
                                st.caption("still missing: " + ", ".join(rx["missing"]))

            with st.container(border=True):                # P10 panel 3 — Groups
                panel_head("Groups")
                # ---- figures: alternate whole-walk views of one thing, kept
                # as a sub-tab strip rather than stacked (P9 continuity -- see
                # T40 subplan for the full justification).
                t_3d, t_walk, t_louv, t_terms = st.tabs(
                    ["3D explorer", "2D walk", "Louvain rel | glob", "Term graph"])
                with t_3d:
                    st.caption("The walk as one live 3D scene: nodes are the walked "
                               "chunks (community colour, size ~ walk score), an "
                               "always-on label per community, DWPC best paths as "
                               "thick links. Hover a node for its card; drag to "
                               "orbit. Positions are a VIEW -- the force sim is not "
                               "deterministic and is never stored.")
                    _ords = list(bnd.sampled)
                    _bodies = {}
                    for _o in _ords:
                        _nd = gt.node(conn, run, _o)
                        _bodies[_o] = {"body": _nd["body"], "doc_id": _nd["doc_id"]}
                    _umap = walk_umap_for(str(run.run_id), q, _ords=_ords)
                    _edges3 = [(a, b, w) for (a, b), w in ai["edges"].items()]
                    _payload = walker_core.walk3d_payload(
                        _ords, cid_of=cid_of, src_of=src_of_all,
                        scores=bnd.scores, bodies=_bodies, kw=ws.kw,
                        salient=salient, pathways=pw, edges=_edges3,
                        umap=_umap)
                    _components.html(walker_core.walk3d_html(_payload, height=720),
                                     height=740)
                    st.caption(f"{len(_payload['nodes'])} chunks · "
                               f"{len(_payload['links'])} links · "
                               f"{len(_payload['sprites'])} community labels · "
                               f"{len(_payload['paths'])} pathway"
                               f"{'s' if len(_payload['paths']) != 1 else ''}"
                               + (" · UMAP toggle on" if _payload["has_umap"]
                                  else " · UMAP off (sparse run or missing embeddings)"))
                with t_walk:
                    fig = draw_communities(touched, terms, xedges)
                    st.plotly_chart(fig, use_container_width=True)
                with t_louv:
                    # P14(a) REVERSED (operator 2026-09-05): the left-right pair
                    # reads as one comparison -- both springs return, relative LH,
                    # global RH. The splits/merges text in the digests stays too.
                    L, R = st.columns(2)
                    for col, title, groups, note in (
                            (L, "relative (this walk only)", rel_groups,
                             "Louvain re-run on the walked subgraph. These ids are "
                             "EPHEMERAL — a view, never persisted, never joined to a "
                             "stored cid, never stable across reruns."),
                            (R, "global communities (stored cids)", glob_groups,
                             "The run's own ingest-time cids, restricted to the "
                             "walked chunks.")):
                        with col:
                            st.markdown(f"**{title}**")
                            st.caption(note)
                            pal = {g: group_color(g) for g in set(groups.values())}
                            gfig = draw_group_graph(sorted(groups), ai["edges"],
                                                    groups, pal, sal=salient)
                            if gfig is not None:
                                st.plotly_chart(gfig, use_container_width=True)
                with t_terms:
                    st.caption("Terms as nodes; colour = which set claims the term "
                               "(the walk's prompt-conditioned BM25 vocabulary vs the "
                               "global unsupervised concept, both = purple). Edges = "
                               "co-occurrence within the walked chunks.")
                    tfig = draw_term_graph(terms, concept, ws.term_members)
                    if tfig is not None:
                        st.plotly_chart(tfig, use_container_width=True)
                    else:
                        st.info("No drawn terms co-occur in this walk.")

                # ---- P11 factbook digests, same panel, largest group first
                rel_rows = walker_core.group_classes(rel_groups, ai["ents"], ai["rels"],
                                                     salient, ndw)
                glob_rows = walker_core.group_classes(glob_groups, ai["ents"], ai["rels"],
                                                      salient, ndw)
                merged = walker_core.dedup_groups(rel_rows, glob_rows)
                st.caption("Per-group factbook: dwpc-ranked terms with scores, entity "
                           "mentions (x-lift only where the entity also occurs outside "
                           "the group), relation templates with corpus-wide support. "
                           "`= c<id>` marks a local group that matches a global "
                           "community; `splits`/`merges` mark where they disagree.")
                _cards = []
                for _e in merged:
                    _txt = walker_core.group_digest(_e["row"], ai["ents"], src_of_all,
                                                    prefix=_e["prefix"], marker=_e["marker"])
                    _mk = _e["marker"] or ""
                    _outline = ("splits" in _mk) or ("merges" in _mk)   # P15(c): divergence = amber outline
                    _cards.append(walker_core.digest_card(
                        _txt, walker_core.card_color(_e),
                        badge=(_mk or None), badge_outline=_outline))

                # display cap 8 -> 9 (P15(c)) so the inline grid's last row is
                # full at 3x3; see T50 _Lessons: for the spec-drift note.
                _grid = _cards[:9]
                for _r in range(0, len(_grid), 3):      # ROW-MAJOR, largest first
                    for _col, _card in zip(st.columns(3), _grid[_r:_r + 3]):
                        with _col:
                            st.markdown(_card, unsafe_allow_html=True)
                _rest = _cards[9:]
                if _rest:
                    with st.expander(f"{len(_rest)} more groups"):
                        for _r in range(0, len(_rest), 3):
                            for _col, _card in zip(st.columns(3), _rest[_r:_r + 3]):
                                with _col:
                                    st.markdown(_card, unsafe_allow_html=True)

            _z3L, _z3R = st.columns([1.4, 1])          # P18 Zone 3: LENSES | REFERENCE
            with _z3L:
                with st.container(border=True):        # P10 panel 4 -- Partitions
                    panel_head("Partitions", "dendrite sort")
                    st.caption("Each row is one correlation chain over the walked chunks "
                               "(correlation sorting.md). Terms are the chain members' own "
                               "BM25-salient vocabulary, ranked by how many members carry "
                               "them -- what this partition talks about vs the others.")
                    if ds is None:
                        st.info("Walk too small (or no stored embeddings) for partitions.")
                    else:
                        _rows = walker_core.partition_rows(ds, src_of_all)
                        st.dataframe(_rows, use_container_width=True, hide_index=True)
                        # ---- P12: each chain's own global-community make-up
                        _clines = walker_core.chain_communities(ds["chunks"]["chains"], cid_of)
                        st.caption("Each chain's global communities (cid:count):")
                        for _row, _line in zip(_rows, _clines):
                            _tok = (_line.split(" ") or ["c?"])[0].split(":")[0]
                            _cid = int(_tok[1:]) if _tok[1:].isdigit() else None
                            _c = cid_color(_cid)
                            st.markdown(
                                f'<div style="background:{rgba(_c, 0.10)};'
                                f'border:1px solid {walker_core.BORDER};'
                                f'border-left:4px solid {_c};border-radius:4px;'
                                f'padding:.25rem .6rem;margin:.2rem 0;font-size:.85rem">'
                                f'chain {html.escape(str(_row["chain"]))} · '
                                f'<code>{html.escape(_line)}</code></div>',
                                unsafe_allow_html=True)
                        _tchains = ds["terms"]["chains"]
                        if _tchains:
                            st.caption("Term chains (terms that rise and fall together "
                                       "across the walked chunks):")
                            for _tch in _tchains:
                                if len(_tch) > 1:
                                    st.markdown("- `" + " > ".join(_tch) + "`")

            with _z3R:
                with st.container(border=True):        # P18 REFERENCE: run-level, prompt-independent
                    # ---- Community map: the cid legend for the lines just printed
                    # (moved from the old Map tab, bodies unchanged; the inline
                    # SELECT stays inline -- 6.21(c)'s "move it to a helper" is
                    # out of scope for T40/T44, see _Lessons:). The quotient-rows
                    # variable is renamed from the original `q` to `_qrows` --
                    # reusing `q` here would clobber the prompt text that
                    # Evidence below still needs (correctness fix, not a rewrite).
                    st.subheader("Community map")
                    st.caption(
                        "How the run's communities interconnect — pure aggregation over fixed "
                        "`cid`s, never a re-partition. Labels, where present, are model-authored "
                        "drafts over a partition Louvain fixed at ingest.")

                    with conn.cursor() as cur:
                        cur.execute("""SELECT cid, size, keywords, medoid_text
                                         FROM community WHERE run_id = %s ORDER BY size DESC""",
                                    (run.run_id,))
                        comms = cur.fetchall()

                    gfig2 = draw_global_map(comms, gt.quotient(conn, run, limit=200))
                    if gfig2 is not None:
                        st.plotly_chart(gfig2, use_container_width=True)

                    crows = []
                    for c in comms:
                        lab = labels.get(c["cid"])
                        crows.append({
                            "cid": c["cid"],
                            "draft label": lab["label"] if lab else "—",
                            "size": c["size"],
                            "keywords": ", ".join(c["keywords"][:5]),
                        })
                    st.dataframe(crows, use_container_width=True, hide_index=True, height=320)

                    if not labels:
                        st.info("No draft labels yet. Generate with "
                                "`python label_communities.py brown-50`.")

                    st.markdown("#### Strongest inter-community links")
                    _qrows = gt.quotient(conn, run, limit=25)
                    st.dataframe(
                        [{"A": f"c{r['cid_a']}" + (f" ({labels[r['cid_a']]['label']})"
                                                   if r["cid_a"] in labels else ""),
                          "B": f"c{r['cid_b']}" + (f" ({labels[r['cid_b']]['label']})"
                                                   if r["cid_b"] in labels else ""),
                          "edges": r["edges"],
                          "avg strength": round(r["avg_strength"], 3)} for r in _qrows],
                        use_container_width=True, hide_index=True, height=300)

            # ---- P18 WALK TRACE: diagnostic, not evidence -- last, always collapsed
            with st.expander(
                    f"Evidence — {len(bnd.sampled)} chunks · {len(cids)} communities · "
                    f"depth {tele['depth']} · {tele['stop']}"
                    f"{' · +' + str(tele['ring']) + ' one degree out' if tele.get('ring') else ''}"
                    f"{' · +' + str(tele['bridge']) + ' bridges' if tele.get('bridge') else ''}",
                    expanded=False):
                st.caption("Key: **bold #id** = a bridge chunk, discovered on the best "
                           "whole-graph path between two retrieved ideas, not found by "
                           "the walk itself; its *salient terms are italicised*. "
                           "Everything else was retrieved by the walk or its one-degree "
                           "ring.")
                st.markdown("#### Query terms")
                ts = gt.term_stats(conn, run, q, bnd.sampled)
                lex = {o for t in ts for o in t["hits"]}
                for t in ts:
                    miss = t["df"] == 0
                    st.markdown(
                        "<span style='font-family:monospace'>%s</span>"
                        "<span style='opacity:%s'>df %d%s</span>"
                        % (t["term"].ljust(16).replace(" ", "&nbsp;"), ".45" if miss else ".85",
                           t["df"], " — not in this corpus" if miss else
                           " · carried by %d of %d" % (len(t["hits"]), len(bnd.sampled))),
                        unsafe_allow_html=True)
                st.caption(f"found lexically {len(lex)} · reached via the graph "
                           f"{len(bnd.sampled) - len(lex)}")

                st.markdown("#### Communities, most present first")
                st.caption("Title terms are the community's OWN vocabulary re-ranked by the "
                           "prompt; the unsupervised concept follows. Each medoid is titled by "
                           "its own salient terms (BM25 vs the corpus, gated).")
                for t in touched:
                    c = t["cid"]; lab = labels.get(c)
                    st.markdown(
                        f"**c{c} · {t['hits']} of {t['size']} · {' / '.join(terms.get(c, []))}**"
                        f"{' · *' + lab['label'] + '*' if lab else ''}  "
                        f"<span style='opacity:.6;font-size:.86em'>concept: "
                        f"{' / '.join(concept.get(c, []))}</span>", unsafe_allow_html=True)
                    lm, gm = medoids[c]
                    m1, m2 = st.columns(2)
                    for col_, o, kind in ((m1, lm, "Local medoid — central to what the walk found here"),
                                          (m2, gm, "Global medoid — central to the whole community")):
                        sal = salient.get(o, {"top": [], "kept": [], "n_kept": 0, "n_in": 0})
                        nd = gt.node(conn, run, o)
                        with col_:
                            st.markdown(f"**{' / '.join(sal['top']) or '(no terms)'}**  "
                                        f"<span style='opacity:.55;font-size:.8em'>`#{o}` · "
                                        f"{nd['doc_id']} · {kind}</span>", unsafe_allow_html=True)
                            st.caption("salient: " + ", ".join(sal["kept"][:14])
                                       + (f" … (+{sal['n_kept'] - 14})" if sal["n_kept"] > 14 else ""))
                            st.markdown(f"<span style='opacity:.75;font-size:.86em'>"
                                        f"{walker_core.clip(interpret.excerpt(nd['body'], q, 600, embed), 600)}</span>",
                                        unsafe_allow_html=True)
                    mine = in_cid[c]
                    st.caption("retrieved here (walk score): " + ", ".join(
                        f"{_mark(o)} {bnd.scores.get(o, 0):.2f}" for o in mine[:12])
                        + (f" … (+{len(mine) - 12})" if len(mine) > 12 else ""))

                xc = gt.cross_community(conn, run, bnd.sampled)
                if xc:
                    st.markdown("#### In between")
                    st.caption("Retrieved chunks whose walked edges reach a different retrieved "
                               "community — exemplars of where the concepts meet.")
                    for x in xc[:8]:
                        nd = gt.node(conn, run, x["ord"])
                        reach = ", ".join(f"c{k}" for k in x["foreign_cids"])
                        st.markdown(
                            f"`#{x['ord']}` c{x['cid']} → {reach} · {x['n_foreign_edges']} edges<br>"
                            f"<span style='opacity:.75;font-size:.88em'>"
                            f"{walker_core.clip(interpret.excerpt(nd['body'], q, 220, embed), 220)}</span>",
                            unsafe_allow_html=True)

                st.markdown("#### Pathways between ideas")
                st.caption("Anchors: the medoids above plus each community's top-scored "
                           "chunk. Pairs are scored by degree-damped path count (DWPC): "
                           "many hub-free paths beat one path through a hub.")
                st.caption(f"subgraph: {pw['components']} WCC"
                           f"{'s' if pw['components'] != 1 else ''} · largest holds "
                           f"{pw['largest_component_frac']:.0%} · density {pw['density']:.2f} · "
                           f"conductance {pw['conductance']:.2f}")
                def _idea(o):
                    t = (salient.get(o) or gt.chunk_salient(conn, run, [o]).get(o)
                         or {}).get("top", [])
                    return " / ".join(t[:2]) or f"#{o}"
                for p in pw["pairs"][:8]:
                    chain = " → ".join(_mark(o) for o in p["path"])
                    st.markdown(
                        f"**{_idea(p['a'])}** ↔ **{_idea(p['b'])}** · dwpc {p['dwpc']:.3f} · "
                        f"{p['n_paths']} paths<br><span style='opacity:.7;font-size:.86em'>"
                        f"best: {chain}</span>", unsafe_allow_html=True)
                if not pw["pairs"]:
                    st.caption("no anchor pair is connected inside this walk")

                if n_br:
                    st.markdown("#### Discovered bridges")
                    st.caption("Chunks the walk never retrieved, pulled in because the "
                               "strongest whole-graph path between two retrieved ideas "
                               "runs through them.")
                    br_ords = [o for o in bnd.sampled if org.get(o) == "bridge"]
                    br_sal = gt.chunk_salient(conn, run, br_ords)
                    for o in br_ords:
                        nd = gt.node(conn, run, o)
                        terms_i = ", ".join(f"*{t}*" for t in br_sal.get(o, {}).get("top", []))
                        st.markdown(
                            f"**#{o}** · {nd['doc_id']} · c{nd['cid']} · "
                            f"walk {bnd.scores.get(o, 0):.2f} · {terms_i}<br>"
                            f"<span style='opacity:.75;font-size:.86em'>"
                            f"{walker_core.clip(interpret.excerpt(nd['body'], q, 300, embed), 300)}</span>",
                            unsafe_allow_html=True)

# ---------------------------------------------------------------- MIRROR (T26)

with tab_neo4j:
    st.markdown("### Mirror")
    st.caption("TEMPORARY TAB (design 6.22 P1): the embedded browser is buggy; "
               "when it is fixed this section returns to Analysis and the tab goes away.")
    st.caption("The actual neo4j browser, auto-connected (auth disabled on "
               "this local container; CSP re-issued with frame-ancestors "
               "http://localhost:8501 -- the stock image sends DENY).")
    _components.iframe(
        f"{config.NEO4J_BROWSER}&connectURL=neo4j%3A%2F%2Flocalhost%3A7687",
        height=760, scrolling=True)
