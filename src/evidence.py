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

import math
import urllib.error
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import export_neo4j as xn_mod
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
    """walker_app.py 569-571, verbatim. (The former second call site,
    draw_layers3d 399-400, was retired with the plotly layers view in T54.)
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


def walk_umap(conn, run, ords) -> dict | None:
    """3D UMAP projection of the walked chunks' stored embeddings, as
    {ord: [x, y, z]}, for the walk3d scene's TF-projector toggle (P16(i)).

    Require: a dense run. Guarantee: either EVERY ord in `ords` has a
    coordinate, or None -- walk3d_payload's `has_umap` is all-or-nothing, so a
    partial map would silently drop the toggle. Positions are a VIEW (P16(g)):
    never persisted, never a join key.

    umap-learn is imported INSIDE the function: evidence.py is imported at
    walker_core/app import time and the UMAP import costs seconds (numba).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P16(i)
    Task: playbook.md T54
    """
    ords = list(ords)
    if not getattr(run, "dense", False) or len(ords) < 4:
        return None
    E, kept = gt.subgraph_embeddings(conn, run, ords)
    if len(kept) != len(ords) or E.shape[0] < 4:
        return None
    from umap import UMAP                      # lazy: numba import cost
    xyz = UMAP(n_components=3, random_state=42,
               n_neighbors=min(15, len(kept) - 1),
               init="random").fit_transform(E)
    xyz = np.asarray(xyz, dtype=float)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = float(max(hi - lo))
    if not np.isfinite(span) or span < 1e-9:
        return None
    ctr = (hi + lo) / 2.0
    xyz = (xyz - ctr) * (200.0 / span)          # ~[-100, 100] cube
    return {o: [round(float(v), 3) for v in row] for o, row in zip(kept, xyz)}


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


# ---- mirror writes (T36, design 6.21(c)) ----

def digest_payload(ds: dict | None) -> dict | None:
    """The three keys export_neo4j.write_digest reads out of dendrite_state's
    output. None in -> None out: a walk that kept <5 embeddings mirrors its
    walk but has no digest to mirror."""
    if ds is None:
        return None
    return {"chunks": ds["chunks"], "sal": ds["sal"], "kept": ds["kept"]}


def community_payload(touched: list, kw: dict) -> list[dict]:
    """One row per touched community for write_digest's CommunitySummary
    upsert. density/conductance are absent on communities community_metrics
    did not cover -- .get keeps them None rather than raising."""
    return [{"cid": t["cid"], "keywords": kw.get(t["cid"], []),
             "size": t["size"], "hits": t["hits"],
             "density": t.get("density"), "conductance": t.get("conductance")}
            for t in touched]


def mirror_walk(bundle, ev: Evidence, ds: dict | None, *, prompt: str,
                xn=None) -> str | None:
    """Best-effort neo4j mirror of a judged walk (T17). Returns None on
    success, the error text when the mirror is unreachable or the server
    rejected the transaction.

    Require: write_walk before write_digest (write_digest MATCHes the Walk node).
    Guarantee: never raises for a transport or server-side failure.
    Maintain: a PROGRAMMING error is NOT swallowed. design 6.21(c) narrows the
    catch from bare Exception to (URLError, RuntimeError, OSError) so a
    TypeError/KeyError/AttributeError in the payload surfaces as a traceback
    instead of reading to the operator as an unreachable server.
    """
    xn = xn or xn_mod
    try:
        xn.write_walk(bundle, ev.pathways, prompt=prompt)
        d = digest_payload(ds)
        if d is not None:
            xn.write_digest(bundle, d,
                            community_payload(ev.touched, ev.kw), prompt=prompt)
    except (urllib.error.URLError, RuntimeError, OSError) as e:   # design 6.21(c)
        return str(e)
    return None


# ---- analysis-view queries (T39, design 6.22 P5/P7) ----

def subgraph_edge_weights(conn, run, ords) -> dict:
    """Undirected {(min,max): strength} over the walked chunks, for the P4
    relative louvain. gt.subgraph_edges returns the DIRECTED edge rows, so the
    two directions collapse under max() -- a partition must not depend on row
    order (design 6.22 P4: fixed seed, reproducible view).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P4
    Task: playbook.md T39
    """
    w: dict = {}
    for e in gt.subgraph_edges(conn, run, list(ords)):
        src, dst = e["src"], e["dst"]
        if src == dst:
            continue
        key = (min(src, dst), max(src, dst))
        w[key] = max(w.get(key, 0.0), float(e["strength"]))
    return w


def walk_entities(conn, run, ords) -> dict:
    """ONE round trip for both the group numerator and the corpus denominator.

    Guarantee: {"mentions": {ord: {entity_id: cnt}}, "names": {entity_id: name},
    "corpus": {entity_id: total_cnt}, "corpus_total": int,
    "class_of": {entity_id: class_id_or_None},
    "class_names": {class_id: name} (non-NULL class_id only)}.

    E15 singleton rule: `class_id == entity_id` means unclassed/singleton --
    consumers must treat that as "no class", never as a class of one.

    `class_names` values come from `class_labels` (E15 amendment, design 6.24):
    argmax(mass x distinctiveness) over the class's members, DISPLAY ONLY --
    class_id itself stays min(members), the deterministic join key.

    Spec: .spec/specs/graph-explorer/design.md 6.22 P5, 6.24 E15-E19 (E17, E15
        amendment)
    Task: playbook.md T39, T72, T79
    """
    ords = list(ords)
    if not ords:
        return {"mentions": {}, "names": {}, "corpus": {}, "corpus_total": 0,
                "class_of": {}, "class_names": {}}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.ord, m.entity_id, m.cnt, e.name, e.class_id
              FROM mentions m
              JOIN entities e  ON e.run_id = m.run_id AND e.entity_id = m.entity_id
             WHERE m.run_id = %s AND m.ord = ANY(%s::int[])""",
            (run.run_id, ords))
        rows = cur.fetchall()

        mentions: dict = {}
        names: dict = {}
        eids: set = set()
        class_of: dict = {}
        class_ids: set = set()
        for r in rows:
            mentions.setdefault(r["ord"], {})[r["entity_id"]] = r["cnt"]
            names[r["entity_id"]] = r["name"]
            eids.add(r["entity_id"])
            class_of[r["entity_id"]] = r["class_id"]
            if r["class_id"] is not None:
                class_ids.add(r["class_id"])

        corpus: dict = {}
        if eids:
            cur.execute("""
                SELECT entity_id, sum(cnt)::bigint AS total
                  FROM mentions
                 WHERE run_id = %s AND entity_id = ANY(%s::int[])
                 GROUP BY entity_id""",
                (run.run_id, sorted(eids)))
            corpus = {r["entity_id"]: r["total"] for r in cur.fetchall()}

        cur.execute("SELECT coalesce(sum(cnt), 0)::bigint AS total FROM mentions"
                    " WHERE run_id = %s", (run.run_id,))
        corpus_total = cur.fetchone()["total"]

    all_labels = class_labels(conn, run) if class_ids else {}
    class_names = {cid: all_labels.get(cid, names.get(cid, cid)) for cid in class_ids}

    return {"mentions": mentions, "names": names, "corpus": corpus,
            "corpus_total": corpus_total, "class_of": class_of,
            "class_names": class_names}


def walk_relations(conn, run, entity_ids) -> list:
    """Relation templates scoped to the walk (P5): both endpoints must be
    among `entity_ids` (the entities mentioned in the walked chunks). Fetched
    once per (run, prompt) per P7; per-GROUP narrowing is pure and happens in
    walker_core.group_relations. Each row carries `rel_class` (None when the
    template has no relation_classes row -- a template with no class row is
    its own singleton, E15/E17).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P5/P7, 6.24 E15-E19 (E17 amendment)
    Task: playbook.md T39, T72
    """
    entity_ids = sorted(entity_ids)
    if not entity_ids:
        return []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT r.src, r.dst, r.template, r.connector, r.n, r.llr, r.npmi,
                   r.example_ord, es.name AS src_name, ed.name AS dst_name,
                   rc.rel_class
              FROM relations r
              JOIN entities es ON es.run_id = r.run_id AND es.entity_id = r.src
              JOIN entities ed ON ed.run_id = r.run_id AND ed.entity_id = r.dst
              LEFT JOIN relation_classes rc
                     ON rc.run_id = r.run_id AND rc.template = r.template
             WHERE r.run_id = %s
               AND r.src = ANY(%s::int[]) AND r.dst = ANY(%s::int[])
             ORDER BY r.llr DESC, r.src, r.dst, r.template""",
            (run.run_id, entity_ids, entity_ids))
        return cur.fetchall()


CLASS_LABEL_SQL = """
    SELECT e.class_id, e.name,
           sum(mn.cnt)::bigint            AS mass,
           count(DISTINCT mn.ord)::bigint AS df,
           (SELECT count(DISTINCT ord) FROM mentions WHERE run_id = %s) AS chunks
      FROM entities e
      JOIN mentions mn ON mn.run_id = e.run_id AND mn.entity_id = e.entity_id
     WHERE e.run_id = %s AND e.class_id IS NOT NULL
     GROUP BY e.class_id, e.name"""


def class_label_score(mass, df, chunks) -> float:
    """Pure scoring arithmetic for the E15 class-naming rule (design 6.24
    amendment): mass x distinctiveness, where distinctiveness = ln(chunks/df)
    demotes a term mentioned in nearly every chunk toward zero regardless of
    how often it fires -- the operator's PPMI law (demotion filter, not a term
    weight) expressed on document frequency. `greatest(df, 1)` guard avoids a
    div-by-zero on an entity with zero recorded mentions.ord rows.

    Spec: .spec/specs/graph-explorer/design.md 6.24 E15 amendment
    Task: playbook.md T79
    """
    return mass * math.log(chunks / max(df, 1))


def pick_class_label(candidates, chunks) -> str:
    """argmax(class_label_score) over `candidates` ((name, mass, df) tuples for
    ONE class), ties broken by mass desc then name asc for determinism -- the
    same order the E15-amendment SQL's `ORDER BY score DESC, mass DESC, name`
    picks rn=1 from.

    Spec: .spec/specs/graph-explorer/design.md 6.24 E15 amendment
    Task: playbook.md T79
    """
    def key(c):
        name, mass, df = c
        return (-class_label_score(mass, df, chunks), -mass, name)
    return min(candidates, key=key)[0]


def class_labels(conn, run) -> dict:
    """E15 amendment: a class is NAMED by argmax(mass x distinctiveness), not by
    its representative entity_id (an insertion-order artifact) and not by raw
    mass (which elects the function word -- live receipt: a song/album class
    named "later", whose mass 23360 beat song 16796).

    The label is DISPLAY ONLY. class_id stays min(members) -- the deterministic
    join key (determinism boundary). ONE round trip: mass/df per (class,
    member) plus the run's chunk count, all in one query; ranking happens in
    Python via `pick_class_label`.

    Spec: .spec/specs/graph-explorer/design.md 6.24 E15 amendment
    Task: playbook.md T79
    """
    with conn.cursor() as cur:
        cur.execute(CLASS_LABEL_SQL, (run.run_id, run.run_id))
        rows = cur.fetchall()
    groups: dict = {}
    chunks = 0
    for r in rows:
        groups.setdefault(r["class_id"], []).append((r["name"], r["mass"], r["df"]))
        chunks = r["chunks"]
    return {cid: pick_class_label(members, chunks) for cid, members in groups.items()}


def class_reference(conn, run, *, k_classes=10, k_members=6) -> dict:
    """P18 REFERENCE: the run's top entity/relation classes by mention/n mass,
    RUN-scoped and prompt-independent -- unlike walk_entities/walk_relations,
    which are cached per (run, prompt, ords). Called ONLY through the
    walker's cached `classes_for(run_id)` wrapper (P7); never hook this into
    analysis_inputs.

    Require: a run whose entity/relation class passes have run.
    Guarantee: {"entity_classes": [...], "relation_classes": [...]}, each list
    empty (never a raise, never a partial key) when nothing is classed.
    Singletons are dropped (HAVING count(*) > 1, E15: an unclassed entity/
    template is its own class and carries no information here).

    `label` per entity class comes from `class_labels` (E15 amendment): argmax
    (mass x distinctiveness) over the class's members, DISPLAY ONLY -- falls
    back to the raw top-mass member's name if class_labels has no entry.

    Spec: .spec/specs/graph-explorer/design.md 6.24 E15-E19 (E17, E15 amendment), P18
    Task: playbook.md T72, T79
    """
    labels = class_labels(conn, run)
    with conn.cursor() as cur:
        cur.execute("""
            WITH mass AS (
              SELECT e.class_id, e.entity_id, e.name,
                     coalesce(sum(m.cnt), 0)::bigint AS cnt
                FROM entities e
                LEFT JOIN mentions m ON m.run_id = e.run_id AND m.entity_id = e.entity_id
               WHERE e.run_id = %s AND e.class_id IS NOT NULL
               GROUP BY e.class_id, e.entity_id, e.name),
            cls AS (
              SELECT class_id, sum(cnt) AS mass, count(*)::int AS members
                FROM mass GROUP BY class_id HAVING count(*) > 1
               ORDER BY mass DESC, class_id LIMIT %s),
            rk AS (
              SELECT m.*, row_number() OVER (PARTITION BY m.class_id
                                             ORDER BY m.cnt DESC, m.name) AS rn
                FROM mass m JOIN cls c ON c.class_id = m.class_id)
            SELECT rk.class_id, c.mass, c.members, rk.entity_id, rk.name, rk.cnt
              FROM rk JOIN cls c ON c.class_id = rk.class_id
             WHERE rk.rn <= %s
             ORDER BY c.mass DESC, c.class_id, rk.rn""",
            (run.run_id, k_classes, k_members))
        ent_rows = cur.fetchall()

        cur.execute("""
            WITH tmass AS (
              SELECT rc.rel_class, r.template, sum(r.n)::bigint AS n, count(*)::int AS pairs
                FROM relations r
                JOIN relation_classes rc ON rc.run_id = r.run_id AND rc.template = r.template
               WHERE r.run_id = %s GROUP BY rc.rel_class, r.template),
            cls AS (
              SELECT rel_class, sum(n) AS mass, count(*)::int AS templates
                FROM tmass GROUP BY rel_class HAVING count(*) > 1
               ORDER BY mass DESC, rel_class LIMIT %s),
            rk AS (
              SELECT t.*, row_number() OVER (PARTITION BY t.rel_class
                                             ORDER BY t.n DESC, t.template) AS rn
                FROM tmass t JOIN cls c ON c.rel_class = t.rel_class)
            SELECT rk.rel_class, c.mass, c.templates, rk.template, rk.n, rk.pairs
              FROM rk JOIN cls c ON c.rel_class = rk.rel_class
             WHERE rk.rn <= %s
             ORDER BY c.mass DESC, c.rel_class, rk.rn""",
            (run.run_id, k_classes, k_members))
        rel_rows = cur.fetchall()

    entity_classes: dict = {}
    for r in ent_rows:
        c = entity_classes.setdefault(r["class_id"], {
            "class_id": r["class_id"], "label": labels.get(r["class_id"], r["name"]),
            "mass": r["mass"], "members": r["members"], "top": []})
        c["top"].append((r["name"], r["cnt"]))

    relation_classes: dict = {}
    for r in rel_rows:
        c = relation_classes.setdefault(r["rel_class"], {
            "rel_class": r["rel_class"], "mass": r["mass"],
            "templates": r["templates"], "top": []})
        c["top"].append((r["template"], r["n"]))

    return {
        "entity_classes": sorted(entity_classes.values(),
                                 key=lambda c: (-c["mass"], c["class_id"])),
        "relation_classes": sorted(relation_classes.values(),
                                   key=lambda c: (-c["mass"], c["rel_class"])),
    }


def analysis_inputs(conn, run, ords) -> dict:
    """The single P7 entry point the walker caches once per (run, prompt).
    The ONLY new DB touch point walker_app may call for the analysis view
    (P5: no inline queries in the UI).

    Spec: .spec/specs/graph-explorer/design.md 6.22 P7
    Task: playbook.md T39
    """
    ents = walk_entities(conn, run, ords)
    entity_ids = sorted({eid for ms in ents["mentions"].values() for eid in ms})
    return {"edges": subgraph_edge_weights(conn, run, ords),
            "ents": ents,
            "rels": walk_relations(conn, run, entity_ids)}
