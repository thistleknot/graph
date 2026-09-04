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
    $env:CHUNKGRAPH_MODEL_DIR = 'C:/Users/user/models/m2v-minilm-l6-256'   # dense signal
    $env:OPENROUTER_API_KEY   = '...'                                       # Reason / Judge
    streamlit run walker_app.py --server.port 8501
Then open http://localhost:8501. Ctrl+C in that terminal stops it.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

import graph_tools as gt
import interpret
import pg_store
import sampler

st.set_page_config(page_title="ChunkGraph Walker", layout="wide")

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#FF7F0E",
           "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
           "#BCBD22", "#17BECF"]
PROV_COLOR = {"both": "#E45756", "dense": "#4C78A8", "sparse": "#9E9E9E"}
LABEL_FILE = Path(os.environ.get("LABEL_OUT", "community_labels.json"))
DEFAULT_MODEL_DIR = os.path.expanduser("~/models/m2v-minilm-l6-256")   # used when CHUNKGRAPH_MODEL_DIR is unset


def _clip(text: str, n: int) -> str:
    """Never cut inside a word (design 6.3). Clip at the last whitespace
    before n and mark the cut; short text is returned untouched."""
    text = text or ""
    if len(text) <= n:
        return text
    head = text[:n]
    cut = head.rsplit(None, 1)[0] if " " in head else head
    return cut + " …"


def cid_color(cid):
    return "#DDDDDD" if cid is None else PALETTE[int(cid) % len(PALETTE)]


@st.cache_resource
def get_conn():
    return gt.connect()


@st.cache_resource
def get_embed(model_dir: str | None):
    """model2vec static embedder for query-conditioned terms (design 6.1).
    None when no model dir: ranking degrades to lexical-then-unsupervised."""
    if not model_dir:
        return None
    try:
        import numpy as np
        from model2vec import StaticModel
        sm = StaticModel.from_pretrained(model_dir)
    except Exception:
        return None

    def embed(texts):
        E = np.asarray(sm.encode(list(texts), show_progress_bar=False),
                       dtype=np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return embed


@st.cache_data(ttl=30)
def load_labels(run_id: str) -> dict:
    """Draft labels, if label_communities.py has been run. Absent is normal.

    cid is RUN-LOCAL (steering): a re-ingest mints new communities under the
    same numbers. A labels file drafted against another run is not merely
    stale, it is wrong -- it put "early electrical science history" on the
    Moroccan elections. Labels apply only when the file names THIS run."""
    if not LABEL_FILE.exists():
        return {}
    try:
        data = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if str(data.get("run_id")) != str(run_id):
        return {"__stale_run__": data.get("run_id")}
    return {c["cid"]: c for c in data.get("communities", []) if c.get("label")}


def cid_badge(cid, labels) -> str:
    """A cid always shows as a cid. A draft label is additive and marked."""
    if cid is None:
        return "no community"
    lab = labels.get(cid)
    return f"c{cid} · *{lab['label']}*" if lab else f"c{cid}"


conn = get_conn()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("ChunkGraph Walker")

runs = gt.list_runs(conn)
live = [r for r in runs if r["superseded_at"] is None]
if not live:
    st.error("No live runs in the database.")
    st.stop()

label = st.sidebar.selectbox("Run", [r["label"] for r in live])
run = gt.get_run(conn, label)
labels = load_labels(str(run.run_id))
st.sidebar.caption(f"{run.n_chunks} chunks · {run.n_edges} edges · "
                   f"{run.n_communities} communities")

embed = get_embed(os.environ.get("CHUNKGRAPH_MODEL_DIR") or DEFAULT_MODEL_DIR)
if "__stale_run__" in labels:
    labels = {}                      # written for another run; cid is run-local

# Only a degraded signal is worth a line on its own; the rest is under details.
if not run.dense:
    st.sidebar.warning("Sparse-only run: anchors are lexical, no dense edges.")
if embed is None:
    st.sidebar.warning("No embedding model found: query-conditioned terms are "
                       "lexical only. Set CHUNKGRAPH_MODEL_DIR.")

with st.sidebar.expander("details"):
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
                      color="#222")),
        hovertext=[f"#{o} · {interpret.chunk_label(meta[o], titles)} · {cid_badge(meta[o]['cid'], labels)}"
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
                                 line=dict(color="#bbb", width=1.2),
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
                    line=dict(color="#333", width=1)),
        hovertext=[_hover(t) for t in touched],
        hoverinfo="text", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white")
    return fig


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
            line=dict(color="#bbb", width=0.6 + 3.5 * d["weight"] / wmax),
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
                        color=color, line=dict(color="#333", width=1)),
            hovertext=[f"{t} · in {len(members[t])} walked chunks" for t in ts],
            hoverinfo="text", showlegend=True))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.02))
    return fig


@st.cache_data(show_spinner=False, max_entries=16)
def dendrite_state(run_id: str, q: str, _bnd=None, _members=None):
    """See _dendrite_state; cache wrapper keyed by (run, prompt)."""
    return _dendrite_state(_bnd, _members)


def _dendrite_state(bnd, members):
    """Both planes of the 3D layers view, dendrite-sorted (correlation
    sorting.md; design: two 2D slices in 3D space, like two NN layers).

    Chunk plane: each walked chunk's variable is its dense-cosine profile
    against every other walked chunk; Pearson over those profiles with
    n = #chunks gives the significance-gated correlation graph and chains.
    Term plane: each drawn term's variable is its tf*idf column over the
    walked chunks (the SAME term-document matrix, read from the other end).
    Layouts are per-plane spring over |r| of significant links, seed 7."""
    import numpy as np
    ords = list(bnd.sampled)
    E, kept = gt.subgraph_embeddings(conn, run, ords)
    if len(kept) < 5:
        return None
    S = E @ E.T
    ch = gt.dendrite_sort(S, kept)
    # term pool: the walked chunks' OWN salient vocabulary (top-3 each) plus
    # the community-drawn sets -- the subgraph on its own terms.
    sal = gt.chunk_salient(conn, run, kept, k=3)
    pool = sorted({t for o in kept for t in (sal.get(o, {}).get("top") or [])}
                  | set(members))
    counts = {o: Counter(gt.tokenize(gt.node(conn, run, o)["body"])) for o in kept}
    def present(t, o):
        return all(p in counts[o] for p in t.split("_"))
    mem = {t: {o for o in kept if present(t, o)} for t in pool}
    dfs = {t: len(mem[t]) for t in pool}
    # tf of a phrase term = min part count (all parts must co-occur)
    X = np.array([[min(counts[o].get(p, 0) for p in t.split("_")) *
                   (np.log(len(kept) / dfs[t]) if dfs[t] else 0.0)
                   for t in pool] for o in kept])
    tm = gt.dendrite_sort(X, pool, min_support=4)
    members = mem

    def plane(out):
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
    tix = {t: j for j, t in enumerate(pool)}
    oix = {o: i for i, o in enumerate(kept)}
    cross = [(o, t, float(X[oix[o], tix[t]])) for t in tm["names"]
             for o in (members.get(t) or []) if o in set(kept)]
    return {"chunks": plane(ch), "terms": plane(tm), "kept": kept, "cross": cross,
            "sal": sal}


def draw_layers3d(state, cid_of, src_of, height=700):
    """The two planes drawn as flat slices at z=0 (chunks) and z=1 (terms),
    cross-layer membership edges falling wherever each plane's spring layout
    landed its endpoint. Deterministic."""
    cp, tp = state["chunks"], state["terms"]
    fig = go.Figure()
    def seg3(pairs, pos, z, color, width):
        xs, ys, zs = [], [], []
        for a, b in pairs:
            if a in pos and b in pos:
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
                zs += [z, z, None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=color, width=width),
                                       hoverinfo="none", showlegend=False))
    seg3([(a, b) for a, b, _ in cp["sig"]], cp["pos"], 0.0, "rgba(150,150,150,0.25)", 1)
    seg3(cp["backbone"], cp["pos"], 0.0, "rgba(228,87,86,0.75)", 3)
    seg3([(a, b) for a, b, _ in tp["sig"]], tp["pos"], 1.0, "rgba(150,150,150,0.25)", 1)
    seg3(tp["backbone"], tp["pos"], 1.0, "rgba(76,120,168,0.85)", 3)
    ws_ = sorted(w for _, _, w in state["cross"]) or [0.0]
    q3 = ws_[int(0.75 * (len(ws_) - 1))]
    for strong, color, width in ((True, "rgba(90,90,160,0.55)", 2.5),
                                 (False, "rgba(120,120,170,0.10)", 1)):
        xs, ys, zs = [], [], []
        for o, t, w in state["cross"]:
            if (w >= q3) is strong and o in cp["pos"] and t in tp["pos"]:
                xs += [cp["pos"][o][0], tp["pos"][t][0], None]
                ys += [cp["pos"][o][1], tp["pos"][t][1], None]
                zs += [0.0, 1.0, None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=color, width=width),
                                       hoverinfo="none", showlegend=False))
    sym = {"brown": "square", "quotes": "diamond", "wiki": "circle", None: "circle"}
    ords = [o for o in state["kept"] if o in cp["pos"]]
    fig.add_trace(go.Scatter3d(
        x=[cp["pos"][o][0] for o in ords], y=[cp["pos"][o][1] for o in ords],
        z=[0.0] * len(ords), mode="markers", name="chunks",
        marker=dict(size=5, color=[cid_color(cid_of.get(o, 0)) for o in ords],
                    symbol=[sym.get(src_of.get(o)) or "circle" for o in ords],
                    line=dict(color="#222", width=1)),
        hovertext=[f"#{o} · c{cid_of.get(o)} · {src_of.get(o) or 'unlabelled'}"
                   for o in ords],
        hoverinfo="text"))
    ts = [t for t in tp["pos"]]
    fig.add_trace(go.Scatter3d(
        x=[tp["pos"][t][0] for t in ts], y=[tp["pos"][t][1] for t in ts],
        z=[1.0] * len(ts), mode="markers+text", name="terms", text=ts,
        textfont=dict(size=9),
        marker=dict(size=4, color=["#B279A2" for _ in ts],
                    line=dict(color="#222", width=1)),
        hovertext=[f"{t} · chain {tp['chain_of'].get(t)}" for t in ts],
        hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False, range=[-0.15, 1.2]),
                                 aspectmode="manual",
                                 aspectratio=dict(x=1.5, y=1.5, z=0.65)),
                      legend=dict(orientation="h", y=0.02))
    return fig


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
            line=dict(color="#bbb", width=0.8 + 4.0 * d["weight"] / wmax),
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
                    line=dict(color="#333", width=1)),
        hovertext=[f"c{c['cid']} · {c['size']} chunks · "
                   + ", ".join((c["keywords"] or [])[:5]) for c in comms],
        hoverinfo="text", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white")
    return fig


tab_walk, tab_map = st.tabs(["Walk", "Map"])

@st.cache_data(show_spinner=False, max_entries=32)
def walk_for(run_id: str, q: str):
    """The walk itself, once per (run, prompt). Deterministic (S1): same inputs,
    same bundle, so a rerun on a button click reads it back instead of walking."""
    return sampler.ef_evidence(conn, run, q)


@st.cache_data(show_spinner=False, max_entries=32)
def walk_state(run_id: str, q: str, _bnd=None, _embed=None):
    """Everything the tab shows that is computed, not model-authored: keyed by
    (run, prompt); the underscore args are inputs Streamlit must not hash."""
    bnd = _bnd
    touched = gt.communities_touched(conn, run, bnd.sampled)
    cm = {c["cid"]: c for c in gt.community_metrics(conn, run)["communities"]}
    for t in touched:
        m = cm.get(t["cid"])
        if m:
            t["density"], t["conductance"] = m["density"], m["conductance"]
    cids = [t["cid"] for t in touched]
    concept = gt.community_terms(conn, run, cids, k=3)
    terms = gt.query_terms(conn, run, cids, q, k=3, embed=_embed)
    cid_of = {o: gt.node(conn, run, o)["cid"] for o in bnd.sampled}
    in_cid = {c: [o for o in bnd.sampled if cid_of[o] == c] for c in cids}
    xedges: dict = {}
    for e in gt.subgraph_edges(conn, run, bnd.sampled):
        ca, cb = cid_of.get(e["src"]), cid_of.get(e["dst"])
        if ca is not None and cb is not None and ca != cb:
            key = (min(ca, cb), max(ca, cb))
            xedges[key] = xedges.get(key, 0) + 1
    medoids = {c: (gt.local_medoid(conn, run, in_cid[c], weights=bnd.scores),
                   gt.community(conn, run, c)["medoid"]) for c in cids}
    med_ords = sorted({o for pair in medoids.values() for o in pair})
    # term graph data (W10 extension): where each drawn term lives among the
    # walked chunks, so terms can be graphed by co-occurrence. A phrase term
    # counts as present when all its parts are in the chunk's token set.
    drawn = sorted({t for ts in terms.values() for t in ts}
                   | {t for ts in concept.values() for t in ts})
    toks = {o: set(gt.tokenize(gt.node(conn, run, o)["body"])) for o in bnd.sampled}
    term_members = {t: {o for o, s in toks.items()
                        if all(p in s for p in t.split("_"))} for t in drawn}
    return {"touched": touched, "cids": cids, "concept": concept, "terms": terms,
            "cid_of": cid_of, "in_cid": in_cid, "xedges": xedges, "medoids": medoids,
            "salient": gt.chunk_salient(conn, run, med_ords, k=3),
            "term_members": term_members}


# ================================================================ WALK
with tab_walk:
    q = st.text_input("Prompt", "", key="q", placeholder="ask the corpus")
    if q.strip():
        bnd, tele = walk_for(str(run.run_id), q)
        if not bnd.sampled:
            st.warning("No lexical anchor matched. Nothing to walk from.")
        else:
            # ---- everything below is computed before any model call (S6),
            # once per (run, prompt): a button click reruns the script and
            # must not redo the walk.
            ws = walk_state(str(run.run_id), q, _bnd=bnd, _embed=embed)
            touched, cids, concept, terms = ws["touched"], ws["cids"], ws["concept"], ws["terms"]
            cid_of, in_cid, xedges, medoids, salient = (ws["cid_of"], ws["in_cid"], ws["xedges"],
                                                          ws["medoids"], ws["salient"])

            # ---- 1. the model's answer, right under the prompt (I13: one call)
            pa_anchors = sorted({o for pair in medoids.values() for o in pair}
                                | {max(in_cid[c], key=lambda o: bnd.scores.get(o, 0))
                                   for c in cids if in_cid[c]})
            pw = gt.pathways(conn, run, bnd.sampled, pa_anchors)
            if st.button("Reason + judge this walk", key="btn_assess",
                         help="ONE model call: hypothesis -> premises -> evaluate -> answer, "
                              "plus a verdict per retrieved chunk; sees the subgraph map"):
                with st.spinner("one call: reasoning over briefs + judging every chunk ..."):
                    ds_ = dendrite_state(str(run.run_id), q, _bnd=bnd,
                                         _members=ws["term_members"])
                    digest = None
                    if ds_ is not None:
                        src_counts: dict = {}
                        for o in ds_["kept"]:
                            c_ = cid_of.get(o)
                            s_ = gt.source_of(gt.node(conn, run, o)) or "unlabelled"
                            src_counts.setdefault(c_, {})
                            src_counts[c_][s_] = src_counts[c_].get(s_, 0) + 1
                        kw = {c_: (gt.community(conn, run, c_)["keywords"] or [])
                              for c_ in cids}
                        cset = {t for ts_ in terms.values() for t in ts_}
                        uset = {t for ts_ in concept.values() for t in ts_}
                        ws_ = sorted(w for _, _, w in ds_["cross"]) or [0.0]
                        q3 = ws_[int(0.75 * (len(ws_) - 1))]
                        strong = [b for b in ds_["cross"] if b[2] >= q3]
                        # I12 amendment (2026-09-03): resolve bare chunk ordinals
                        # to "source:top_term" so the model can cite digest ids
                        # (interpret.reason cited 0 on B3 with bare ordinals).
                        # ds_["sal"] is the salient-term lookup _dendrite_state
                        # already computed over ds_["kept"] -- reused as-is,
                        # zero additional chunk_salient calls.
                        sal_ = ds_["sal"]
                        resolver = {o: f"{gt.source_of(gt.node(conn, run, o)) or 'unlabelled'}:"
                                       f"{(sal_.get(o, {}).get('top') or ['?'])[0]}"
                                    for o in ds_["kept"]}
                        digest = interpret.render_digest(
                            touched, kw, src_counts, ds_["chunks"]["chains"],
                            ds_["terms"]["chains"], cset, uset, strong, pw, resolver,
                            metrics={t["cid"]: t for t in touched if "density" in t})
                    st.session_state["assess"] = (q, interpret.reason(
                        conn, run, bnd, terms, concept, embed=embed,
                        judge=True, pw=pw, digest=digest))
                    # T17: every judged walk lands in the neo4j mirror --
                    # best-effort, never blocks the answer (design.md I14 sect).
                    if os.environ.get("NEO4J_MIRROR", "1") != "0":
                        try:
                            import export_neo4j as xn
                            xn.write_walk(bnd, pw, prompt=q)
                            if ds_ is not None:
                                xn.write_digest(
                                    bnd,
                                    {"chunks": ds_["chunks"], "sal": ds_["sal"],
                                     "kept": ds_["kept"]},
                                    [{"cid": t["cid"],
                                      "keywords": kw.get(t["cid"], []),
                                      "size": t["size"], "hits": t["hits"],
                                      "density": t.get("density"),
                                      "conductance": t.get("conductance")}
                                     for t in touched],
                                    prompt=q)
                        except Exception as e:              # noqa: BLE001
                            st.warning(f"neo4j mirror skipped: {e}")

            got = st.session_state.get("assess")
            has_answer = False
            if got and got[0] == q:
                rr = got[1]
                st.markdown("### Answer")
                if not rr["ok"]:
                    st.warning(f"Stopped: {rr['error']}")
                else:
                    has_answer = True
                    st.markdown(rr["answer"] or
                                "_No premise was judged supported, so there is nothing to "
                                "answer from. The premises below say why._")
                    st.caption(f"{rr['backend']} · one call: {len(rr['briefs'])} community "
                               f"briefs + {len(rr.get('shown', []))} chunks judged"
                               f"{' · ' + rr['structure_note'] if rr.get('structure_note') else ''}")
                if rr["hypotheses"]:
                    st.markdown("**Hypothesis** " + (rr["hypothesis"] or ""))
                    others = [h for h in rr["hypotheses"] if h != rr["hypothesis"]]
                    if others:
                        st.caption("also considered: " + " | ".join(others))
                    if rr["why"]:
                        st.caption(f"chosen because: {rr['why']}")
                if rr["premises"]:
                    st.markdown("**Premises**")
                    col = {"supports": "#2a7", "contradicts": "#c33",
                           "insufficient": "#999", "unsupported": "#bbb"}
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

                # ---- the judge channel of the same call
                if rr.get("verdicts"):
                    st.markdown("### Judged evidence")
                    st.caption(f"judged {rr['coverage']:.0%} of {len(rr['shown'])} shown · "
                               f"{len(rr['entailed'])} entail · "
                               f"{len(rr['contradicts'])} contradict")
                    why = {v["ord"]: v["why"] for v in rr["verdicts"]}
                    for label, ords_, colour in (("Entails", rr["entailed"], "#2a7"),
                                                 ("Contradicts", rr["contradicts"], "#c33")):
                        if ords_:
                            st.markdown(f"**{label}**")
                            for o in ords_:
                                nd = gt.node(conn, run, o)
                                st.markdown(
                                    f"<span style='color:{colour}'>●</span> `#{o}` · {nd['doc_id']} · "
                                    f"c{nd['cid']} · walk {bnd.scores.get(o, 0):.2f} — <i>{why.get(o, '')}</i><br>"
                                    f"<span style='opacity:.75;font-size:.88em'>{_clip(interpret.excerpt(nd['body'], q, 300, embed), 300)}</span>",
                                    unsafe_allow_html=True)
                    neutral = [v for v in rr["verdicts"]
                               if v["verdict"] == "neutral" and v["ord"] in set(rr["shown"])]
                    if neutral:
                        with st.expander(f"Neutral ({len(neutral)}) — the model's reason for each"):
                            for v in neutral:
                                nd = gt.node(conn, run, v["ord"])
                                st.markdown(f"<span style='color:#999'>●</span> `#{v['ord']}` · "
                                            f"{nd['doc_id']} · c{nd['cid']} — <i>{v['why']}</i>",
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

            # ---- 2. the evidence: one expander, collapsed once an answer exists
            with st.expander(
                    f"Evidence — {len(bnd.sampled)} chunks · {len(cids)} communities · "
                    f"depth {tele['depth']} · {tele['stop']}"
                    f"{' · +' + str(tele['ring']) + ' one degree out' if tele.get('ring') else ''}"
                    f"{' · +' + str(tele['bridge']) + ' bridges' if tele.get('bridge') else ''}",
                    expanded=not has_answer):
                org = getattr(bnd, "origin", {}) or {}
                n_br = sum(1 for v in org.values() if v == "bridge")
                st.caption("Key: **bold #id** = a bridge chunk, discovered on the best "
                           "whole-graph path between two retrieved ideas, not found by "
                           "the walk itself; its *salient terms are italicised*. "
                           "Everything else was retrieved by the walk or its one-degree "
                           "ring.")
                def _mark(o):
                    return f"**#{o}**" if org.get(o) == "bridge" else f"#{o}"
                t_walk, t_terms, t_3d, t_glob = st.tabs(
                    ["This walk", "Term graph", "3D layers", "Global map"])
                with t_walk:
                    fig = draw_communities(touched, terms, xedges)
                    st.plotly_chart(fig, use_container_width=True)
                with t_terms:
                    st.caption("Terms as nodes; colour = which set claims the term "
                               "(the walk's prompt-conditioned BM25 vocabulary vs the "
                               "global unsupervised concept, both = purple). Edges = "
                               "co-occurrence within the walked chunks.")
                    tfig = draw_term_graph(terms, concept, ws["term_members"])
                    if tfig is not None:
                        st.plotly_chart(tfig, use_container_width=True)
                    else:
                        st.info("No drawn terms co-occur in this walk.")
                with t_3d:
                    st.caption("Two planes, like two layers of a network: chunks below "
                               "(community colour, source shape), terms above, each "
                               "spring-settled in its own slice; red/blue = dendrite "
                               "chain backbones (correlation sorting, significance-gated "
                               "with n = walked chunks); faint verticals = membership. "
                               "Short vertical edges mean the two Louvain worlds agree.")
                    ds = dendrite_state(str(run.run_id), q, _bnd=bnd,
                                        _members=ws["term_members"])
                    if ds is None:
                        st.info("Walk too small (or no stored embeddings) for the "
                                "layered view.")
                    else:
                        src_of = {o: gt.source_of(gt.node(conn, run, o))
                                  for o in ds["kept"]}
                        st.plotly_chart(draw_layers3d(ds, cid_of, src_of),
                                        use_container_width=True)
                        ct, tt = ds["chunks"]["chains"], ds["terms"]["chains"]
                        st.caption(f"chunk chains: {len(ct)} "
                                   f"(longest {max(map(len, ct)) if ct else 0}) · "
                                   f"term chains: {len(tt)} "
                                   f"(longest {max(map(len, tt)) if tt else 0})")
                with t_glob:
                    st.caption("Every community in the run; filled = reached by this "
                               "walk. The same map the model sees.")
                    st.image(interpret.render_walk_image(conn, run, bnd, pw),
                             use_container_width=True)

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
                                        f"{_clip(interpret.excerpt(nd['body'], q, 600, embed), 600)}</span>",
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
                            f"{_clip(interpret.excerpt(nd['body'], q, 220, embed), 220)}</span>",
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
                            f"{_clip(interpret.excerpt(nd['body'], q, 300, embed), 300)}</span>",
                            unsafe_allow_html=True)

# ================================================================ MAP
with tab_map:
    # ---- PARTITIONS (the operator's standing ask): the dendrite-sorted
    # chains, each partition's BM25 salient terms laid side by side so the
    # partitions can be COMPARED -- pregrouped text, same place as the graphs.
    _wp0 = (st.session_state.get("q") or "").strip()
    st.markdown("### Partitions (dendrite sort)")
    if not _wp0:
        st.info("Type a prompt on the Walk tab -- its dendrite-sorted "
                "partitions render here, salient terms side by side.")
    if _wp0:
        st.caption("Each row is one correlation chain over the walked chunks "
                   "(correlation sorting.md). Terms are the chain members' own "
                   "BM25-salient vocabulary, ranked by how many members carry "
                   "them -- what this partition talks about vs the others.")
        try:
            _bnd0, _ = walk_for(str(run.run_id), _wp0)
            _ws0 = walk_state(str(run.run_id), _wp0, _bnd=_bnd0, _embed=embed)
            _ds0 = dendrite_state(str(run.run_id), _wp0, _bnd=_bnd0,
                                  _members=_ws0["term_members"])
        except Exception as e:                              # noqa: BLE001
            _ds0 = None
            st.warning(f"partitions unavailable: {e}")
        if _ds0 is not None:
            _sal0 = _ds0["sal"]
            _rows = []
            for _ci, _chain in enumerate(_ds0["chunks"]["chains"]):
                _tc = Counter()
                _mix = Counter()
                for _o in _chain:
                    for _t in (_sal0.get(_o, {}).get("top") or []):
                        _tc[_t] += 1
                    _nd0 = gt.node(conn, run, _o)
                    _mix[gt.source_of(_nd0) or "?"] += 1
                _rows.append({
                    "chain": _ci + 1,
                    "chunks": len(_chain),
                    "sources": " ".join(f"{k}:{v}" for k, v in
                                        _mix.most_common()),
                    "salient terms (carried by N members)":
                        ", ".join(f"{t}({n})" if n > 1 else t
                                  for t, n in _tc.most_common(12)),
                })
            st.dataframe(_rows, use_container_width=True, hide_index=True)
            _tchains = _ds0["terms"]["chains"]
            if _tchains:
                st.caption("Term chains (terms that rise and fall together "
                           "across the walked chunks):")
                for _tch in _tchains:
                    if len(_tch) > 1:
                        st.markdown("- `" + " > ".join(_tch) + "`")


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

    gfig = draw_global_map(comms, gt.quotient(conn, run, limit=200))
    if gfig is not None:
        st.plotly_chart(gfig, use_container_width=True)

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

# ---------------------------------------------------------------- MIRROR (T26)

with tab_map:
    # ONE input field for the whole app: the Walk tab's prompt drives the
    # mirror section too. A judged walk lands in neo4j (T17) and shows here.
    st.markdown("### Mirror")
    st.caption("The actual neo4j browser, auto-connected (auth disabled on "
               "this local container; CSP re-issued with frame-ancestors "
               "http://localhost:8501 -- the stock image sends DENY).")
    import streamlit.components.v1 as _components
    _components.iframe(
        "http://localhost:7474/browser/",
        height=760, scrolling=True)
