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

Run:  streamlit run walker_app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

import graph_tools as gt
import sampler

st.set_page_config(page_title="ChunkGraph Walker", layout="wide")

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#FF7F0E",
           "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
           "#BCBD22", "#17BECF"]
PROV_COLOR = {"both": "#E45756", "dense": "#4C78A8", "sparse": "#9E9E9E"}
LABEL_FILE = Path(os.environ.get("LABEL_OUT", "community_labels.json"))


def cid_color(cid):
    return "#DDDDDD" if cid is None else PALETTE[int(cid) % len(PALETTE)]


@st.cache_resource
def get_conn():
    return gt.connect()


@st.cache_data(ttl=30)
def load_labels() -> dict:
    """Draft labels, if label_communities.py has been run. Absent is normal."""
    if not LABEL_FILE.exists():
        return {}
    try:
        data = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
        return {c["cid"]: c for c in data.get("communities", []) if c.get("label")}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def cid_badge(cid, labels) -> str:
    """A cid always shows as a cid. A draft label is additive and marked."""
    if cid is None:
        return "no community"
    lab = labels.get(cid)
    return f"c{cid} · *{lab['label']}*" if lab else f"c{cid}"


conn = get_conn()
labels = load_labels()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("ChunkGraph Walker")

runs = gt.list_runs(conn)
live = [r for r in runs if r["superseded_at"] is None]
if not live:
    st.error("No live runs in the database.")
    st.stop()

label = st.sidebar.selectbox("Run", [r["label"] for r in live])
run = gt.get_run(conn, label)

st.sidebar.caption(
    f"`{run.run_id}`\n\n"
    f"{run.n_chunks} chunks · {run.n_edges} edges · {run.n_communities} communities")

if run.dense:
    st.sidebar.success(f"Retrieval mode: **fused** (embed_dim {run.embed_dim})")
else:
    st.sidebar.warning(
        "Retrieval mode: **sparse-only** — no dense space on this run, so "
        "anchors are lexical only.")

sp = run.single_provenance
if sp:
    st.sidebar.info(
        f"Every edge carries a single provenance value: **{sp}** "
        f"({run.provenance[sp]}). Nothing here is a fused result.")

if labels:
    st.sidebar.caption(f"{len(labels)} draft community labels loaded "
                       f"(italic = model-authored)")

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
                      color="#222")),
        hovertext=[f"#{o} · {meta[o]['doc_id']} · {cid_badge(meta[o]['cid'], labels)}"
                   f"<br>{meta[o]['body'][:120]}…" for o in ords],
        hoverinfo="text"))
    fig.update_layout(showlegend=False, height=height,
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def community_panel(ords):
    touched = gt.communities_touched(conn, run, ords)
    st.caption(f"{len(ords)} chunks · **{len(touched)} of {run.n_communities} "
               f"communities** touched")
    for t in touched:
        lab = labels.get(t["cid"])
        name = f" — *{lab['label']}*" if lab else ""
        st.markdown(
            f"<span style='color:{cid_color(t['cid'])}'>●</span> "
            f"**c{t['cid']}**{name} · {t['hits']} of {t['size']} members · "
            f"{', '.join(t['keywords'][:4])}", unsafe_allow_html=True)
    return touched


tab_walk, tab_map = st.tabs(["Walk", "Map"])

# ================================================================ WALK
with tab_walk:
    q = st.text_input("Prompt", "", key="q", placeholder="ask the corpus")
    if q.strip():
        b, tele = sampler.ef_evidence(conn, run, q)
        if not b.sampled:
            st.warning("No lexical anchor matched. Nothing to walk from.")
        else:
            fig = draw_subgraph(b.sampled, [], b.anchors[0], height=680)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{len(b.sampled)} chunks · depth {tele['depth']} · "
                       f"{tele['stop']}")

# ================================================================ MAP
with tab_map:
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

    rows = []
    for c in comms:
        lab = labels.get(c["cid"])
        rows.append({
            "cid": c["cid"],
            "draft label": lab["label"] if lab else "—",
            "size": c["size"],
            "keywords": ", ".join(c["keywords"][:5]),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True, height=320)

    if not labels:
        st.info("No draft labels yet. Generate with "
                "`python label_communities.py brown-50`.")

    st.markdown("#### Strongest inter-community links")
    q = gt.quotient(conn, run, limit=25)
    st.dataframe(
        [{"A": f"c{r['cid_a']}" + (f" ({labels[r['cid_a']]['label']})"
                                   if r["cid_a"] in labels else ""),
          "B": f"c{r['cid_b']}" + (f" ({labels[r['cid_b']]['label']})"
                                   if r["cid_b"] in labels else ""),
          "edges": r["edges"],
          "avg strength": round(r["avg_strength"], 3)} for r in q],
        use_container_width=True, hide_index=True, height=300)
