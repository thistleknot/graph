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
    return {"touched": touched, "cids": cids, "concept": concept, "terms": terms,
            "cid_of": cid_of, "in_cid": in_cid, "xedges": xedges, "medoids": medoids,
            "salient": gt.chunk_salient(conn, run, med_ords, k=3)}


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
                    st.session_state["assess"] = (q, interpret.reason(
                        conn, run, bnd, terms, concept, embed=embed,
                        judge=True, pw=pw))

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
                t_walk, t_glob = st.tabs(["This walk", "Global map"])
                with t_walk:
                    fig = draw_communities(touched, terms, xedges)
                    st.plotly_chart(fig, use_container_width=True)
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
