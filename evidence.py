"""evidence.py -- ONE evidence assembly for the walker, diagnostics and backfill.

Spec: .spec/specs/graph-explorer/design.md 6.21(a) (evidence.assemble, byte-identical digest)
Task: playbook.md T35

Pure function of (DB contents at run_id, bundle, embed): no streamlit, no
plotly, no networkx here. Layout-only code (spring positions) stays in the
UI -- see walker_app.py's `plane()`, which consumes `Evidence.dendrite`'s raw
`dendrite_sort` output and only ADDS pos/backbone/chain_of on top. The digest
is independent of every spring layout.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import graph_tools as gt
import interpret


@dataclass(frozen=True)
class Evidence:
    """Everything a walk's assess block needs, minus layout.

    `dendrite` holds the RAW `gt.dendrite_sort` output for each plane
    (`{"chunks": {...names/r/sig/chains...}, "terms": {...}, "kept", "cross",
    "sal"}`), or None when fewer than 5 chunks kept embeddings. Documentation
    trap: this raw `["sig"]` is a boolean matrix; the UI's `plane()` turns it
    into an edge list under the SAME key -- same name, different type, one is
    the input to the other.
    """
    # community layer (was walk_state)
    touched: list
    cids: list
    concept: dict
    terms: dict
    cid_of: dict
    in_cid: dict
    xedges: dict
    medoids: dict
    salient: dict
    term_members: dict
    # pathway layer (was inline in the assess block)
    pa_anchors: list
    pathways: dict
    # dendrite layer -- RAW dendrite_sort output, no layout
    dendrite: dict | None
    # digest layer (all None/{}/set() when dendrite is None)
    src_counts: dict
    kw: dict
    cset: set
    uset: set
    strong: list
    resolver: dict
    metrics: dict
    digest: str | None


def top_quartile(cross) -> tuple[float, list]:
    """walker_app.py 569-571 == draw_layers3d 399-400, verbatim, twice.
    Empty cross -> (0.0, [])."""
    ws_ = sorted(w for _, _, w in cross) or [0.0]
    q3 = ws_[int(0.75 * (len(ws_) - 1))]
    strong = [b for b in cross if b[2] >= q3]
    return q3, strong


def src_counts_of(kept, cid_of, src_of) -> dict:
    """walker_app.py 560-564. `src_of` is {ord: source-or-None}; a chunk with
    no cid keys under None, a chunk with no source keys under 'unlabelled'."""
    src_counts: dict = {}
    for o in kept:
        c_ = cid_of.get(o)
        s_ = src_of.get(o) or "unlabelled"
        src_counts.setdefault(c_, {})
        src_counts[c_][s_] = src_counts[c_].get(s_, 0) + 1
    return src_counts


def resolver_of(kept, src_of, sal) -> dict:
    """walker_app.py 578-581: bare chunk ordinals -> 'source:top_term' so the
    model can cite digest ids. Empty top list renders '<src>:?'."""
    return {o: f"{src_of.get(o) or 'unlabelled'}:"
               f"{(sal.get(o, {}).get('top') or ['?'])[0]}"
            for o in kept}


def load_embed(model_dir: str | None):
    """walker_app.py get_embed (70-87) minus @st.cache_resource. model2vec
    static embedder for query-conditioned terms; None degrades to lexical."""
    if not model_dir:
        return None
    try:
        from model2vec import StaticModel
        sm = StaticModel.from_pretrained(model_dir)
    except Exception:
        return None

    def embed(texts):
        E = np.asarray(sm.encode(list(texts), show_progress_bar=False),
                       dtype=np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return embed


def assemble(conn, run, bundle, *, embed=None) -> Evidence:
    """Pure function of (DB contents at run_id, bundle, embed). Cache key is
    (run_id, bundle.query) -- valid only because walk_for pins the knobs to
    defaults for a given prompt; diag_rerun passes non-default knobs and does
    not cache, so no key collision exists. Do not cache inside this function;
    the `st.cache_data` wrapper in the walker owns that.

    `q = bundle.query`: sampler.py builds Bundle(query=query, ...) with the
    raw, unmodified query, and the walker passes that same string straight
    into ef_evidence -- so bundle.query IS the string gt.query_terms already
    receives. No separate prompt= parameter.
    """
    q = bundle.query

    # ---- community layer (was walker_app.py walk_state, 492-527) ----
    touched = gt.communities_touched(conn, run, bundle.sampled)
    cm = {c["cid"]: c for c in gt.community_metrics(conn, run)["communities"]}
    for t in touched:
        m = cm.get(t["cid"])
        if m:
            t["density"], t["conductance"] = m["density"], m["conductance"]
    cids = [t["cid"] for t in touched]
    concept = gt.community_terms(conn, run, cids, k=3)
    terms = gt.query_terms(conn, run, cids, q, k=3, embed=embed)
    cid_of = {o: gt.node(conn, run, o)["cid"] for o in bundle.sampled}
    in_cid = {c: [o for o in bundle.sampled if cid_of[o] == c] for c in cids}
    xedges: dict = {}
    for e in gt.subgraph_edges(conn, run, bundle.sampled):
        ca, cb = cid_of.get(e["src"]), cid_of.get(e["dst"])
        if ca is not None and cb is not None and ca != cb:
            key = (min(ca, cb), max(ca, cb))
            xedges[key] = xedges.get(key, 0) + 1
    medoids = {c: (gt.local_medoid(conn, run, in_cid[c], weights=bundle.scores),
                   gt.community(conn, run, c)["medoid"]) for c in cids}
    med_ords = sorted({o for pair in medoids.values() for o in pair})
    drawn = sorted({t for ts in terms.values() for t in ts}
                   | {t for ts in concept.values() for t in ts})
    toks = {o: set(gt.tokenize(gt.node(conn, run, o)["body"])) for o in bundle.sampled}
    term_members = {t: {o for o, s in toks.items()
                        if all(p in s for p in t.split("_"))} for t in drawn}
    salient = gt.chunk_salient(conn, run, med_ords, k=3)

    # ---- pathway layer (was inline in the assess block, 547-550) ----
    pa_anchors = sorted({o for pair in medoids.values() for o in pair}
                        | {max(in_cid[c], key=lambda o: bundle.scores.get(o, 0))
                           for c in cids if in_cid[c]})
    pw = gt.pathways(conn, run, bundle.sampled, pa_anchors)

    src_of = {o: gt.source_of(gt.node(conn, run, o)) for o in bundle.sampled}

    # ---- dendrite + digest layer (was _dendrite_state + assess block) ----
    ords = list(bundle.sampled)
    E, kept = gt.subgraph_embeddings(conn, run, ords)
    dendrite = None
    src_counts: dict = {}
    kw: dict = {}
    cset: set = set()
    uset: set = set()
    strong: list = []
    resolver: dict = {}
    metrics: dict = {}
    digest = None

    if len(kept) >= 5:
        S = E @ E.T
        ch = gt.dendrite_sort(S, kept)
        sal = gt.chunk_salient(conn, run, kept, k=3)
        pool = sorted({t for o in kept for t in (sal.get(o, {}).get("top") or [])}
                      | set(term_members))
        counts = {o: Counter(gt.tokenize(gt.node(conn, run, o)["body"])) for o in kept}

        def present(t, o):
            return all(p in counts[o] for p in t.split("_"))
        mem = {t: {o for o in kept if present(t, o)} for t in pool}
        dfs = {t: len(mem[t]) for t in pool}
        X = np.array([[min(counts[o].get(p, 0) for p in t.split("_")) *
                       (np.log(len(kept) / dfs[t]) if dfs[t] else 0.0)
                       for t in pool] for o in kept])
        tm = gt.dendrite_sort(X, pool, min_support=4)
        tix = {t: j for j, t in enumerate(pool)}
        oix = {o: i for i, o in enumerate(kept)}
        cross = [(o, t, float(X[oix[o], tix[t]])) for t in tm["names"]
                 for o in (mem.get(t) or []) if o in set(kept)]
        dendrite = {"chunks": ch, "terms": tm, "kept": kept, "cross": cross, "sal": sal}

        src_counts = src_counts_of(kept, cid_of, src_of)
        kw = {c_: (gt.community(conn, run, c_)["keywords"] or []) for c_ in cids}
        cset = {t for ts_ in terms.values() for t in ts_}
        uset = {t for ts_ in concept.values() for t in ts_}
        _, strong = top_quartile(cross)
        resolver = resolver_of(kept, src_of, sal)
        metrics = {t["cid"]: t for t in touched if "density" in t}
        digest = interpret.render_digest(
            touched, kw, src_counts, ch["chains"], tm["chains"],
            cset, uset, strong, pw, resolver, metrics=metrics)

    return Evidence(
        touched=touched, cids=cids, concept=concept, terms=terms, cid_of=cid_of,
        in_cid=in_cid, xedges=xedges, medoids=medoids, salient=salient,
        term_members=term_members, pa_anchors=pa_anchors, pathways=pw,
        dendrite=dendrite, src_counts=src_counts, kw=kw, cset=cset, uset=uset,
        strong=strong, resolver=resolver, metrics=metrics, digest=digest)
