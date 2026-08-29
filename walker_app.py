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
                                 line=dict(color="#bbb", width=1.2),
                                 hoverinfo="none", showlegend=False))
    mx = max(t["hits"] for t in touched)
    fig.add_trace(go.Scatter(
        x=[pos[t["cid"]][0] for t in touched],
        y=[pos[t["cid"]][1] for t in touched],
        mode="markers+text",
        text=[" / ".join(terms.get(t["cid"], [])[:3]) for t in touched],
        textposition="top center",
        textfont=dict(size=12),
        marker=dict(size=[14 + 40 * t["hits"] / mx for t in touched],
                    color=[cid_color(t["cid"]) for t in touched],
                    line=dict(color="#333", width=1)),
        hovertext=[f"c{t['cid']} · {t['hits']} of {t['size']}" for t in touched],
        hoverinfo="text", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white")
    return fig


tab_walk, tab_map = st.tabs(["Walk", "Map"])

# ================================================================ WALK
with tab_walk:
    q = st.text_input("Prompt", "", key="q", placeholder="ask the corpus")
    if q.strip():
        bnd, tele = sampler.ef_evidence(conn, run, q)
        if not bnd.sampled:
            st.warning("No lexical anchor matched. Nothing to walk from.")
        else:
            touched = gt.communities_touched(conn, run, bnd.sampled)
            cids = [t["cid"] for t in touched]
            terms = gt.community_terms(conn, run, cids, k=3)
            cid_of = {o: gt.node(conn, run, o)["cid"] for o in bnd.sampled}
            in_cid = {c: [o for o in bnd.sampled if cid_of[o] == c] for c in cids}
            xedges: dict = {}
            for e in gt.subgraph_edges(conn, run, bnd.sampled):
                ca, cb = cid_of.get(e["src"]), cid_of.get(e["dst"])
                if ca is not None and cb is not None and ca != cb:
                    key = (min(ca, cb), max(ca, cb))
                    xedges[key] = xedges.get(key, 0) + 1

            fig = draw_communities(touched, terms, xedges)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{len(bnd.sampled)} chunks · {len(cids)} communities · "
                       f"depth {tele['depth']} · {tele['stop']}")

            st.markdown("#### Communities, most present first")
            st.caption("Terms are BM25 over the WHOLE community — the implied "
                       "evidence. Expand for the explicit evidence: the chunks "
                       "this walk retrieved there.")
            for t in touched:
                c = t["cid"]; mine = in_cid[c]
                lab = labels.get(c)
                head = (f"c{c} · **{t['hits']}** of {t['size']} · "
                        f"**{' / '.join(terms.get(c, []))}**"
                        + (f" · *{lab['label']}*" if lab else ""))
                with st.expander(head, expanded=(t is touched[0])):
                    lm = gt.local_medoid(conn, run, mine)
                    gm = gt.community(conn, run, c)["medoid"]
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"**Local medoid** · #{lm} — central to "
                                    f"what this walk found here")
                        st.caption(gt.node(conn, run, lm)["body"][:400])
                    with m2:
                        st.markdown(f"**Global medoid** · #{gm} — central to "
                                    f"the whole community")
                        st.caption(gt.node(conn, run, gm)["body"][:400])
                    st.markdown("**Retrieved chunks**")
                    for o in mine:
                        nd = gt.node(conn, run, o)
                        st.markdown(
                            f"`#{o}` · {nd['doc_id']}<br>"
                            f"<span style='opacity:.75;font-size:.88em'>"
                            f"{nd['body'][:300]}…</span>",
                            unsafe_allow_html=True)

            xc = gt.cross_community(conn, run, bnd.sampled)
            if xc:
                st.markdown("#### In between")
                st.caption("Retrieved chunks whose walked edges reach a "
                           "different retrieved community — exemplars of "
                           "where the concepts meet.")
                for x in xc[:8]:
                    nd = gt.node(conn, run, x["ord"])
                    reach = ", ".join(f"c{k}" for k in x["foreign_cids"])
                    st.markdown(
                        f"`#{x['ord']}` c{x['cid']} → {reach} · "
                        f"{x['n_foreign_edges']} edges<br>"
                        f"<span style='opacity:.75;font-size:.88em'>"
                        f"{nd['body'][:220]}…</span>", unsafe_allow_html=True)

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
