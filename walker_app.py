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
import interpret
import sampler

st.set_page_config(page_title="ChunkGraph Walker", layout="wide")

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#FF7F0E",
           "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
           "#BCBD22", "#17BECF"]
PROV_COLOR = {"both": "#E45756", "dense": "#4C78A8", "sparse": "#9E9E9E"}
LABEL_FILE = Path(os.environ.get("LABEL_OUT", "community_labels.json"))


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

st.sidebar.caption(
    f"`{run.run_id}`\n\n"
    f"{run.n_chunks} chunks · {run.n_edges} edges · {run.n_communities} communities")

if run.dense:
    st.sidebar.success(f"Retrieval mode: **fused** (embed_dim {run.embed_dim})")
else:
    st.sidebar.warning(
        "Retrieval mode: **sparse-only** — no dense space on this run, so "
        "anchors are lexical only.")

embed = get_embed(os.environ.get("CHUNKGRAPH_MODEL_DIR"))
if embed is None:
    st.sidebar.warning("Query-conditioned terms: **lexical only** — set "
                       "CHUNKGRAPH_MODEL_DIR for the dense signal.")

sp = run.single_provenance
if sp:
    st.sidebar.info(
        f"Every edge carries a single provenance value: **{sp}** "
        f"({run.provenance[sp]}). Nothing here is a fused result.")

if "__stale_run__" in labels:
    st.sidebar.warning(f"Draft labels ignored: they were written for run "
                       f"`{str(labels['__stale_run__'])[:8]}`, not this one. "
                       f"cid is run-local; re-run label_communities.py.")
    labels = {}
elif labels:
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
            concept = gt.community_terms(conn, run, cids, k=3)      # unsupervised
            terms = gt.query_terms(conn, run, cids, q, k=3, embed=embed)  # prompt-conditioned
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
            st.caption("The three words are the community's OWN vocabulary, "
                       "re-ranked by your prompt — the face of the community "
                       "the prompt lights up. Expand for the unsupervised "
                       "concept and the explicit evidence.")
            for t in touched:
                c = t["cid"]; mine = in_cid[c]
                lab = labels.get(c)
                head = (f"c{c} · **{t['hits']}** of {t['size']} · "
                        f"**{' / '.join(terms.get(c, []))}**"
                        + (f" · *{lab['label']}*" if lab else ""))
                with st.expander(head, expanded=(t is touched[0])):
                    st.caption(f"Community concept (unsupervised BM25): "
                               f"**{' / '.join(concept.get(c, []))}**")
                    lm = gt.local_medoid(conn, run, mine, weights=bnd.scores)
                    gm = gt.community(conn, run, c)["medoid"]
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"**Local medoid** · #{lm} — central to "
                                    f"what this walk found here, weighted "
                                    f"by walk score")
                        st.caption(_clip(gt.node(conn, run, lm)["body"], 400))
                    with m2:
                        st.markdown(f"**Global medoid** · #{gm} — central to "
                                    f"the whole community")
                        st.caption(_clip(gt.node(conn, run, gm)["body"], 400))
                    st.markdown("**Retrieved chunks**")
                    for o in mine:
                        nd = gt.node(conn, run, o)
                        st.markdown(
                            f"`#{o}` · {nd['doc_id']}<br>"
                            f"<span style='opacity:.75;font-size:.88em'>"
                            f"{_clip(nd['body'], 300)}</span>",
                            unsafe_allow_html=True)

            st.markdown("#### Entailment")
            st.caption(f"`{interpret.OPENROUTER_MODEL}` (OpenRouter; local Ollama "
                       "as fallback) judges every shown chunk — entails / "
                       "contradicts / neutral — then answers from the entailed "
                       "ones only. **Draft**: model-authored, checkable chunk by "
                       "chunk, never an input to the graph.")
            if st.button("Judge this walk", key="btn_interpret"):
                with st.spinner("classifying the evidence…"):
                    res = interpret.answer(conn, run, bnd, terms, concept, embed=embed)
                st.session_state["interp"] = (q, res)
            got = st.session_state.get("interp")
            if got and got[0] == q:
                res = got[1]
                if not res["ok"]:
                    st.warning(f"No verdict: {res['error']}. The walk above is "
                               "unaffected.")
                else:
                    n_shown = len(res["shown"])
                    st.caption(f"{res['backend']} · {res['rerank_note']} · judged "
                               f"{res['coverage']:.0%} of {n_shown} shown · "
                               f"{len(res['entailed'])} entail · "
                               f"{len(res['contradicts'])} contradict")
                    if res["foreign"]:
                        st.error("Foreign ids not in this walk (unsupported): "
                                 + ", ".join(f"#{o}" for o in res["foreign"]))
                    if res["self_contradicting"]:
                        st.error("Cited in the answer but NOT judged entailing: "
                                 + ", ".join(f"#{o}" for o in res["self_contradicting"]))
                    st.markdown("**Answer**")
                    st.markdown(res["answer"] or
                                "_Empty by design: no chunk was judged to entail an "
                                "answer, so there is nothing to answer from. The "
                                "model's per-chunk reasons are under **Neutral** below._")
                    why = {v["ord"]: v["why"] for v in res["verdicts"]}
                    for label, ords_, colour in (("Entails", res["entailed"], "#2a7"),
                                                 ("Contradicts", res["contradicts"], "#c33")):
                        if ords_:
                            st.markdown(f"**{label}**")
                            for o in ords_:
                                nd = gt.node(conn, run, o)
                                st.markdown(
                                    f"<span style='color:{colour}'>●</span> `#{o}` · "
                                    f"{nd['doc_id']} · c{nd['cid']} — "
                                    f"<i>{why.get(o, '')}</i><br>"
                                    f"<span style='opacity:.75;font-size:.88em'>"
                                    f"{_clip(nd['body'], 300)}</span>",
                                    unsafe_allow_html=True)
                    neutral = [v for v in res["verdicts"]
                               if v["verdict"] == "neutral" and v["ord"] in set(res["shown"])]
                    if neutral:
                        with st.expander(f"Neutral ({len(neutral)}) — retrieved, judged "
                                         f"not evidence; the model's reason for each"):
                            for v in neutral:
                                nd = gt.node(conn, run, v["ord"])
                                st.markdown(
                                    f"<span style='color:#999'>●</span> `#{v['ord']}` · "
                                    f"{nd['doc_id']} · c{nd['cid']} — <i>{v['why']}</i>",
                                    unsafe_allow_html=True)
                    with st.expander("Exactly what the model was shown"):
                        st.code(res["evidence"], language="text")
                    with st.expander("Exactly what the model returned (raw)"):
                        st.code(res["text"], language="json")

            st.markdown("#### Reason over the communities")
            st.caption("Community briefs — terms both ways, local and global "
                       "medoid excerpts — go to the model, which proposes a "
                       "hypothesis, extracts the premises it needs, evaluates each "
                       "against the medoid evidence it cites, and answers from what "
                       "survived. **Draft**: every stage kept, every verdict "
                       "checkable against a chunk.")
            if st.button("Reason about this walk", key="btn_reason"):
                with st.spinner("hypothesis → premises → evaluate → answer…"):
                    rr = interpret.reason(conn, run, bnd, terms, concept, embed=embed)
                st.session_state["reason"] = (q, rr)
            got_r = st.session_state.get("reason")
            if got_r and got_r[0] == q:
                rr = got_r[1]
                if not rr["ok"]:
                    st.warning(f"Reasoning stopped: {rr['error']}. Stages that ran are "
                               f"below; the walk is unaffected.")
                else:
                    st.caption(f"{rr['backend']} · {len(rr['briefs'])} community briefs")
                if rr["hypotheses"]:
                    st.markdown("**Hypotheses**")
                    for i, h in enumerate(rr["hypotheses"]):
                        mark = "→" if h == rr["hypothesis"] else "·"
                        st.markdown(f"{mark} {h}")
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
                if rr["ok"]:
                    st.markdown("**Answer**")
                    st.markdown(rr["answer"] or
                                "_No premise was judged supported, so there is nothing "
                                "to answer from. The premises above say why._")
                if rr["briefs_text"]:
                    with st.expander("Community briefs the model reasoned over"):
                        st.code(rr["briefs_text"], language="text")
                if rr["stages"]:
                    with st.expander("Raw stage replies"):
                        for name, txt in rr["stages"].items():
                            st.markdown(f"**{name}**"); st.code(txt, language="json")

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
                        f"{_clip(nd['body'], 220)}</span>", unsafe_allow_html=True)

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
