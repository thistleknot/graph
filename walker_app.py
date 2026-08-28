"""
walker_app.py — interactive graph walker over a persisted ChunkGraph run.

Two ways into the same graph, all sharing one deterministic tool surface
(`graph_tools`): read-only at the server, run-scoped, traversing the indexed
src/dst columns only.

    Walk       you drive, stepping neighbour by neighbour
    Reach      gt.walk() from here: every result carries the provenance and
               source-doc of each hop that reached it (graph_tools W7)
    Find       sampler.evidence() runs the deterministic walk and shows what
               it found: communities, evidence chunks, and the bundle params
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

st.sidebar.divider()
limit = st.sidebar.slider("Neighbours per step", 5, 60, 25)
min_strength = st.sidebar.slider("Min edge strength (the disclosed floor)",
                                 0.0, 0.6, 0.0, 0.01)
st.sidebar.caption(
    "The cookbook traversal hard-codes a 0.05 floor. Here it is a visible dial.")

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("trail", [])
ss.setdefault("visited", [])
ss.setdefault("sat", [])
ss.setdefault("hits", [])


def visit(ord_: int):
    ss.trail.append(ord_)
    if ord_ not in ss.visited:
        ss.visited.append(ord_)
    ss.sat.append(len(gt.communities_touched(conn, run, ss.visited)))


def reset():
    ss.trail, ss.visited, ss.sat = [], [], []


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


tab_walk, tab_find, tab_map = st.tabs(["Walk", "Find", "Map"])

# ================================================================ WALK
with tab_walk:
    c_search, c_reset = st.columns([4, 1])
    with c_search:
        query = st.text_input("Find an anchor", "", key="q_walk")
    with c_reset:
        st.write("")
        if st.button("Reset", use_container_width=True):
            reset()
    if query.strip() and st.button("Search", key="btn_search"):
        ss.hits = gt.search(conn, run, query, k=8)

    if ss.hits and not ss.visited:
        st.caption("Lexical hits — click one to start walking")
        for h in ss.hits:
            if st.button(
                    f"#{h['ord']} · {h['doc_id']} · score {h['score']:.1f} · "
                    f"{cid_badge(h['cid'], labels)}",
                    key=f"anchor{h['ord']}", use_container_width=True):
                reset()
                visit(h["ord"])
                st.rerun()

    if not ss.visited:
        st.info("Search above to drop an anchor, then step neighbour by neighbour.")
    else:
        cur_ord = ss.trail[-1]
        cur = gt.node(conn, run, cur_ord)
        left, right = st.columns([1.05, 1])

        with left:
            st.subheader(f"Chunk #{cur['ord']}")
            st.caption(f"doc **{cur['doc_id']}** · {cur['n_tok']} tokens · "
                       f"{cid_badge(cur['cid'], labels)}")
            if cur["cid"] is None:
                st.warning("This chunk belongs to no community (below min_size=5 "
                           "at ingest).")
            st.markdown(
                f"<div style='background:#00000010;padding:.7rem;"
                f"border-radius:.4rem;max-height:180px;overflow:auto'>"
                f"{cur['body'][:1400]}</div>", unsafe_allow_html=True)

            st.markdown("#### Step to a neighbour")
            nbrs = gt.neighbors(conn, run, cur_ord, limit=limit,
                                min_strength=min_strength)
            if not nbrs:
                st.info("No neighbours above the floor. Lower the dial.")
            for n in nbrs:
                c1, c2 = st.columns([1, 3.4])
                with c1:
                    seen = "· seen" if n["ord"] in ss.visited else ""
                    if st.button(f"→ #{n['ord']} {seen}",
                                 key=f"step{cur_ord}_{n['ord']}",
                                 use_container_width=True):
                        visit(n["ord"])
                        st.rerun()
                with c2:
                    st.markdown(
                        f"<span style='color:{PROV_COLOR.get(n['provenance'],'#999')}'>●</span> "
                        f"**{n['strength']:.3f}** · {n['provenance']} · "
                        f"{cid_badge(n['cid'], labels)} · {n['doc_id']}<br>"
                        f"<span style='opacity:.65;font-size:.86em'>"
                        f"{(n['preview'] or '')[:140]}…</span>",
                        unsafe_allow_html=True)

        with right:
            st.subheader("What you are walking into")
            touched = community_panel(ss.visited)
            fig = draw_subgraph(ss.visited, ss.trail, cur_ord)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            if len(ss.sat) > 1:
                st.caption("Communities touched per step — flattening means covered")
                st.line_chart(ss.sat, height=110)
            if len(touched) >= 2:
                a, b = touched[0]["cid"], touched[1]["cid"]
                st.markdown(f"#### Bridges: c{a} ↔ c{b}")
                for e in gt.bridges(conn, run, a, b, limit=5):
                    st.caption(f"#{e['src']} ({e['src_doc']}) ↔ #{e['dst']} "
                               f"({e['dst_doc']}) · {e['strength']:.3f}")

            st.markdown("#### Reach from here — with the reason")
            rc1, rc2 = st.columns(2)
            hops = rc1.slider("hops", 1, 3, 2, key="reach_hops")
            floor = rc2.slider("score floor", 0.0, 0.30, 0.05, 0.01,
                               key="reach_floor")
            reach = gt.walk(conn, run, cur_ord, hops=hops,
                            min_score=floor, cap=25)
            if not reach:
                st.info("Nothing survives the score floor. Lower it.")
            else:
                nx_doc = sum(1 for r in reach if r["cross_doc"])
                st.caption(f"{len(reach)} reached · {nx_doc} cross-document "
                           f"({100*nx_doc/len(reach):.0f}%)")
                for r in reach:
                    dot = "".join(
                        f"<span style='color:{PROV_COLOR.get(pv,'#999')}'>●</span>"
                        for pv in r["prov_path"])
                    arrow = " ".join(
                        f"{r['node_path'][i]}<span style='opacity:.5'>"
                        f"–{pv}→</span>" for i, pv in enumerate(r["prov_path"]))
                    st.markdown(
                        f"{dot} **#{r['ord']}** · {r['score']:.4f} · h{r['hop']}"
                        f"{' · CROSS-DOC' if r['cross_doc'] else ''}<br>"
                        f"<span style='font-family:monospace;font-size:.82em'>"
                        f"{arrow}{r['node_path'][-1]}</span><br>"
                        f"<span style='opacity:.6;font-size:.8em'>"
                        f"{' › '.join(r['doc_path'])}</span>",
                        unsafe_allow_html=True)

# ================================================================ FIND
with tab_find:
    st.subheader("Deterministic evidence walk")
    st.caption(
        "`sampler.evidence()` — anchors by BM25, expand 2 hops scoring each node "
        "by anchor score x product of edge strengths, take top-n, group by the "
        "run's **stored** cid. No model, and at T=0 no randomness: same query, "
        "same evidence, every time.")

    fq = st.text_input("Question", "jury trial grand jury investigation",
                       key="q_find")
    c1, c2, c3 = st.columns(3)
    with c1:
        f_n = st.slider("n — chunks sampled", 4, 80, sampler.DEFAULT_N)
    with c2:
        f_T = st.slider("T — 0 is argmax", 0.0, 3.0, sampler.DEFAULT_T, 0.1)
    with c3:
        f_k = st.slider("communities shown", 1, 10, 4)

    if fq.strip():
        b = sampler.evidence(conn, run, fq, n=f_n, T=f_T, k_comm=f_k)

        if not b.sampled:
            st.warning("No lexical anchor matched. Nothing to walk from.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("anchors", len(b.anchors))
            m2.metric("candidates", b.candidates)
            m3.metric("sampled", len(b.sampled))
            m4.metric("top-community share", f"{b.concentration():.0%}")
            if b.enumerated:
                st.info(f"n >= candidate set, so this enumerated all "
                        f"{b.candidates} rather than sampling.")

            st.markdown("#### Communities the evidence landed in")
            worst = max(c["hits"] for c in b.communities)
            for c in b.communities:
                bar = "█" * max(1, round(18 * c["hits"] / worst))
                lab = labels.get(c["cid"])
                st.markdown(
                    f"<span style='color:{cid_color(c['cid'])}'>{bar}</span> "
                    f"**c{c['cid']}** · {c['hits']} of {len(b.sampled)} "
                    f"{'· *' + lab['label'] + '*' if lab else ''}<br>"
                    f"<span style='opacity:.7;font-size:.86em'>"
                    f"{', '.join(c['keywords'])}</span>",
                    unsafe_allow_html=True)

            st.caption(
                "Keywords are corpus-derived tf*idf and are the trustworthy "
                "layer. Italic labels are model-authored drafts and can "
                "misdescribe their own community — c1 reads 'Rural road funding' "
                "but its keywords are jury, election, department, mayor.")

            st.markdown("#### The evidence")
            by_cid: dict = {}
            for o in b.sampled:
                nd = gt.node(conn, run, o)
                by_cid.setdefault(nd["cid"], []).append(nd)
            for c in b.communities:
                for nd in by_cid.get(c["cid"], []):
                    with st.expander(
                            f"#{nd['ord']} · {nd['doc_id']} · "
                            f"{cid_badge(nd['cid'], labels)}"):
                        st.write(nd["body"][:1200])

            fig = draw_subgraph(b.sampled, b.anchors,
                                b.anchors[0] if b.anchors else None, height=320)
            if fig:
                st.markdown("#### Sampled subgraph")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Bundle parameters — this is what makes it re-derivable"):
                st.json({"run_id": b.run_id, "anchors": b.anchors,
                         "sampled": b.sampled, "top_cids": b.top_cids,
                         **b.params})


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
