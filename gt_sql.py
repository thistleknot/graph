"""gt_sql.py -- parameterized SELECTs, run identity, provenance narration.

Guards W1-W13, W21 (search half) and R20 are stated once in graph_tools.py;
this module implements them.

Spec: .spec/specs/graph-explorer/design.md (module split, no behaviour change)
Task: playbook.md T37
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row

import config

DSN = config.DSN


def _gt():
    """Sibling calls resolve through graph_tools at CALL time. Two reasons, both
    load-bearing: (1) the shim is the monkeypatch surface -- tests patch
    graph_tools.alias_map and graph_tools.CACHE_DIR and expect the call inside a
    sibling module to see it (test_graph_tools.py:1230, :652; test_sampler.py:860);
    (2) a call-time lookup makes the three modules acyclic and import-order-proof.
    After the first call this is a sys.modules dict hit."""
    import graph_tools
    return graph_tools


def connect(dsn: str = DSN):
    """W3: read-only at the server, not by convention."""
    return psycopg.connect(dsn, row_factory=dict_row, autocommit=True,
                           options="-c default_transaction_read_only=on")


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    label: str
    ingested_at: object
    n_chunks: int
    embed_dim: int | None
    dense: bool
    n_edges: int
    n_communities: int
    provenance: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return "fused" if self.dense else "sparse-only"

    @property
    def single_provenance(self):
        """State it when every edge carries one provenance value."""
        return next(iter(self.provenance)) if len(self.provenance) == 1 else None


def list_runs(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT run_id, label, ingested_at, n_chunks, embed_dim, superseded_at
              FROM graph_run ORDER BY ingested_at DESC""")
        return cur.fetchall()


def get_run(conn, label: str) -> RunHandle:
    """W1/W5/W6. Raises when a label has no live run -- a real state under
    supersede-never-delete; guessing would violate 'which run is an explicit
    choice'. The >1 case is foreclosed by graph_run_live_label UNIQUE."""
    with conn.cursor() as cur:
        cur.execute("""SELECT run_id, label, ingested_at, n_chunks, embed_dim
                         FROM live_run WHERE label = %s""", (label,))
        row = cur.fetchone()
        if row is None:
            raise LookupError(f"no live run for label {label!r}")
        rid = row["run_id"]
        cur.execute("SELECT count(*) AS c FROM node_embedding WHERE run_id=%s", (rid,))
        n_emb = cur.fetchone()["c"]
        cur.execute("""SELECT provenance, count(*) AS c FROM edge
                        WHERE run_id=%s AND valid_to IS NULL
                        GROUP BY provenance""", (rid,))
        prov = {r["provenance"]: r["c"] for r in cur.fetchall()}
        cur.execute("SELECT count(*) AS c FROM community WHERE run_id=%s", (rid,))
        n_comm = cur.fetchone()["c"]
    return RunHandle(
        run_id=rid, label=row["label"], ingested_at=row["ingested_at"],
        n_chunks=row["n_chunks"], embed_dim=row["embed_dim"],
        dense=bool(row["embed_dim"]) and n_emb > 0,
        n_edges=sum(prov.values()), n_communities=n_comm, provenance=prov)


def search(conn, run: RunHandle, query: str, k: int = 8,
           K1: float = 1.5, B: float = 0.75,
           expand_aliases: bool = False) -> list[dict]:
    """Lexical entry point: BM25 over the graph's OWN vocabulary (W12), scored
    in Python over the cached per-run postings (corpus_index). Was SQL over
    jsonb -- 7.3 s per query on 500 documents, called twice per walk.

    W2: attrs is used for SCORING here, never for traversal.

    W21: `expand_aliases` (default False) is the ONE place the default lives.
    With the flag off, this executes the same statements it does today --
    that byte-identity is what the diagnostic baseline pins. WHERE resolution
    has not run (alias_map reads back {}), `terms` is left untouched rather
    than round-tripped through expand_terms -- a no-op re-sort/re-set would
    still be functionally a no-op but could reorder the score accumulation
    and break float byte-identity for no reason."""
    gt = _gt()
    terms = gt.tokenize(query)
    if not terms:
        return []
    if expand_aliases:
        aliases = gt.alias_map(conn, run)
        if aliases:
            terms = gt.expand_terms(terms, aliases)
    ix = gt.corpus_index(conn, run)
    N, avgdl = ix["n"], ix["avgdl"]
    score: dict = {}; hit: dict = {}
    for t in set(terms):
        post = ix["post"].get(t)
        if not post:
            continue
        d = len(post)
        idf = math.log(1 + (N - d + 0.5) / (d + 0.5))
        for o, f in post.items():
            dl = ix["dl"].get(o, 1.0)
            score[o] = score.get(o, 0.0) + idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
            hit[o] = hit.get(o, 0) + 1
    top = sorted(score, key=lambda o: (-score[o], o))[:k]
    rows = [{"ord": o, "score": score[o], "terms_hit": hit[o]} for o in top]
    return _decorate(conn, run, rows)


def node(conn, run: RunHandle, ord_: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT n.ord, n.doc_id, n.body, (n.attrs->>'n_tok')::int AS n_tok,
                   n.attrs -> 'tf' AS tf, n.attrs->>'source' AS source,
                   c.cid, c.keywords
              FROM node n
              LEFT JOIN community c
                     ON c.run_id = n.run_id AND c.members @> ARRAY[n.ord]
             WHERE n.run_id = %s AND n.ord = %s""", (run.run_id, ord_))
        return cur.fetchone()


def neighbors(conn, run: RunHandle, ord_: int, limit: int = 25,
              min_strength: float = 0.0) -> list[dict]:
    """The walk step. W2: edge_sym only, indexed on (run_id, src/dst). Ordered
    by strength so a bounded fetch keeps the strongest, not the arbitrary."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.b AS ord, s.strength, s.provenance, s.sim_sparse,
                   s.sim_dense, n.doc_id, left(n.body, 240) AS preview,
                   n.attrs->>'source' AS source, c.cid, c.keywords
              FROM edge_sym s
              JOIN node n ON n.run_id = s.run_id AND n.ord = s.b
              LEFT JOIN community c
                     ON c.run_id = s.run_id AND c.members @> ARRAY[s.b]
             WHERE s.run_id = %s AND s.a = %s AND s.valid_to IS NULL
               AND s.strength >= %s
             ORDER BY s.strength DESC LIMIT %s""",
            (run.run_id, ord_, min_strength, limit))
        return cur.fetchall()


def community(conn, run: RunHandle, cid: int):
    with conn.cursor() as cur:
        cur.execute("""SELECT cid, size, keywords, medoid, medoid_text
                         FROM community WHERE run_id = %s AND cid = %s""",
                    (run.run_id, cid))
        return cur.fetchone()


def communities_touched(conn, run: RunHandle, ords: list[int]) -> list[dict]:
    """'What are we walking into.' Groups a visited set by its STORED cid --
    no partitioning, no clustering, no derived grouping."""
    if not ords:
        return []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.cid, c.size, c.keywords, c.medoid_text, count(*) AS hits,
                   array_agg(n.attrs->>'source')
                     FILTER (WHERE n.attrs ? 'source') AS sources
              FROM community c
              JOIN unnest(%s::int[]) AS v(ord) ON c.members @> ARRAY[v.ord]
              JOIN node n ON n.run_id = c.run_id AND n.ord = v.ord
             WHERE c.run_id = %s
             GROUP BY c.cid, c.size, c.keywords, c.medoid_text
             ORDER BY hits DESC""", (ords, run.run_id))
        rows = cur.fetchall()
    for r in rows:
        raw = r.pop("sources") or []
        counts: dict = {}
        for s in raw:
            counts[s] = counts.get(s, 0) + 1
        r["sources"] = {s: counts[s] for s in sorted(counts, key=lambda s: (-counts[s], s))}
    return rows


def run_sources(conn, run: RunHandle) -> dict:
    """R20: the run-level source mix, the sidebar's headline. NULL bucket
    (pre-R20 nodes) dropped. {} on a pre-R20 run."""
    with conn.cursor() as cur:
        cur.execute("""SELECT attrs->>'source' AS source, count(*) AS c
                         FROM node WHERE run_id=%s GROUP BY 1""", (run.run_id,))
        rows = cur.fetchall()
    counts = {r["source"]: r["c"] for r in rows if r["source"] is not None}
    return {s: counts[s] for s in sorted(counts, key=lambda s: (-counts[s], s))}


def bridges(conn, run: RunHandle, cid_a: int, cid_b: int, limit: int = 20):
    """The 'larger interconnection': actual edges joining two communities."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH m AS (SELECT cid, unnest(members) AS ord
                         FROM community WHERE run_id = %s AND cid IN (%s, %s))
            SELECT e.src, e.dst, e.strength, e.provenance,
                   e.src_doc, e.dst_doc, ma.cid AS src_cid, mb.cid AS dst_cid
              FROM edge e
              JOIN m ma ON ma.ord = e.src
              JOIN m mb ON mb.ord = e.dst
             WHERE e.run_id = %s AND e.valid_to IS NULL AND ma.cid <> mb.cid
             ORDER BY e.strength DESC LIMIT %s""",
            (run.run_id, cid_a, cid_b, run.run_id, limit))
        return cur.fetchall()


def quotient(conn, run: RunHandle, limit: int = 60):
    """Community-level graph: how communities interconnect, corpus-wide.
    Pure aggregation over fixed cids -- never a re-partition."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH m AS (SELECT cid, unnest(members) AS ord
                         FROM community WHERE run_id = %s)
            SELECT least(ma.cid, mb.cid) AS cid_a,
                   greatest(ma.cid, mb.cid) AS cid_b,
                   count(*) AS edges, avg(e.strength) AS avg_strength
              FROM edge e
              JOIN m ma ON ma.ord = e.src
              JOIN m mb ON mb.ord = e.dst
             WHERE e.run_id = %s AND e.valid_to IS NULL AND ma.cid <> mb.cid
             GROUP BY 1, 2 ORDER BY edges DESC LIMIT %s""",
            (run.run_id, run.run_id, limit))
        return cur.fetchall()


def subgraph_edges(conn, run: RunHandle, ords: list[int]) -> list[dict]:
    """Induced edges among a visited set, for drawing."""
    if len(ords) < 2:
        return []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT src, dst, strength, provenance FROM edge
             WHERE run_id = %s AND valid_to IS NULL
               AND src = ANY(%s::int[]) AND dst = ANY(%s::int[])""",
            (run.run_id, ords, ords))
        return cur.fetchall()


def walk(conn, run: RunHandle, seed: int, hops: int = 2,
         min_score: float = 0.05, cap: int = 50) -> list[dict]:
    """Multi-hop reach from one seed, carrying WHY each node was reached (W7).

    Mirrors cookbook #1's recursive CTE and adds two accumulators, so every row
    answers not just "what did I reach" but "through which edges, out of which
    documents". Best-scoring path per node wins; ties break on fewer hops.

    Require:  seed is a live node ord in run; hops >= 1; 0 <= min_score <= 1
    Guarantee: <= cap rows, each carrying node_path, prov_path, doc_path whose
              lengths satisfy len(node_path) == len(doc_path) == len(prov_path)+1.
              prov_path is cast to text[]: psycopg has no loader for the
              edge_provenance enum and would hand back the literal '{sparse}'
              string, which iterates as characters and silently corrupts any
              caller that zips it against node_path.
    Maintain: score decays multiplicatively and is cut at min_score, exactly as
              _gist_walk and cookbook #1 do -- this reports the walk that the
              graph actually performs, it does not define a second one.
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    with conn.cursor() as cur:
        cur.execute("""
            WITH RECURSIVE seed AS (
                SELECT n.ord, n.doc_id FROM node n
                 WHERE n.run_id = %(rid)s AND n.ord = %(seed)s
            ), walk AS (
                SELECT ord AS node, 1.0::float AS score, 0 AS hop,
                       ARRAY[ord] AS node_path,
                       ARRAY[]::edge_provenance[] AS prov_path,
                       ARRAY[doc_id] AS doc_path
                  FROM seed
                UNION ALL
                SELECT s.b, w.score * s.strength, w.hop + 1,
                       w.node_path || s.b, w.prov_path || s.provenance,
                       w.doc_path || n.doc_id
                  FROM walk w
                  JOIN edge_sym s ON s.run_id = %(rid)s AND s.a = w.node
                                 AND s.valid_to IS NULL
                  JOIN node n ON n.run_id = %(rid)s AND n.ord = s.b
                 WHERE w.hop < %(hops)s
                   AND NOT s.b = ANY(w.node_path)
                   AND w.score * s.strength > %(mins)s
            ), best AS (
                SELECT DISTINCT ON (node)
                       node, score, hop, node_path, prov_path, doc_path
                  FROM walk WHERE hop > 0
                 ORDER BY node, score DESC, hop ASC
            )
            SELECT b.node AS ord, b.score, b.hop, b.node_path,
                   b.prov_path::text[] AS prov_path,   -- psycopg has no
                   b.doc_path, n.doc_id, left(n.body, 240) AS preview,
                   n.attrs->>'source' AS source,
                   c.cid, c.keywords,
                   (b.doc_path[1] <> n.doc_id) AS cross_doc
              FROM best b
              JOIN node n ON n.run_id = %(rid)s AND n.ord = b.node
              LEFT JOIN community c
                     ON c.run_id = %(rid)s AND c.members @> ARRAY[b.node]
             ORDER BY b.score DESC LIMIT %(cap)s""",
            {"rid": run.run_id, "seed": seed, "hops": hops,
             "mins": min_score, "cap": cap})
        return cur.fetchall()


def why(row: dict) -> str:
    """One-line rendering of a walk() row's justification. Presentation only."""
    steps = []
    for i, prov in enumerate(row["prov_path"]):
        steps.append(f"{row['node_path'][i]} --{prov}--> {row['node_path'][i+1]}")
    return f"{' | '.join(steps)}  [{' > '.join(row['doc_path'])}]"


def source_of(row: dict) -> str | None:
    """R20: the corpus a row belongs to. Explicit `source` wins; else the
    doc_id prefix (`wiki/1234` -> `wiki`) for a mid-campaign run that has
    prefixed ids before the `source` key is persisted; else None. Never
    raises on a row missing both keys."""
    src = row.get("source")
    if isinstance(src, str) and src:
        return src
    doc_id = row.get("doc_id")
    if isinstance(doc_id, str) and "/" in doc_id:
        return doc_id.split("/", 1)[0]
    return None


def source_mix(rows) -> dict:
    """R20: counts of source_of() over row-dicts, None dropped. Deterministic
    order: count desc, then name asc."""
    counts: dict = {}
    for row in rows:
        src = source_of(row)
        if src is None:
            continue
        counts[src] = counts.get(src, 0) + 1
    return {s: counts[s] for s in sorted(counts, key=lambda s: (-counts[s], s))}


def format_source_mix(mix: dict) -> str:
    """R20: presentation only, e.g. 'brown 12 · wiki 3'; '' when empty."""
    if not mix:
        return ""
    return " · ".join(f"{src} {n}" for src, n in mix.items())


def term_stats(conn, run: RunHandle, query: str,
               ords: list[int] | None = None) -> list[dict]:
    """Per query term: corpus df, and which of `ords` carry it (W8).

    Separates the two ways a chunk enters a result set. A chunk carrying query
    terms was found LEXICALLY. A chunk carrying none was reached through the
    GRAPH -- by an edge, or by a term it shares with something that did match.
    Only the second kind is evidence the graph contributed anything.

    Require:  query is raw user text; ords, if given, are live node ordinals.
    Guarantee: one row per tokenized term, df 0 included, ordered by df ASC so
              the rarest (most discriminating) term reads first.
    """
    terms = _gt().tokenize(query)
    if not terms:
        return []
    with conn.cursor() as cur:
        cur.execute("""
            WITH q AS (SELECT unnest(%s::text[]) AS term)
            SELECT q.term,
                   count(n.ord) AS df,
                   COALESCE(array_agg(n.ord ORDER BY n.ord)
                            FILTER (WHERE n.ord = ANY(%s::int[])), '{}') AS hits
              FROM q LEFT JOIN node n
                ON n.run_id = %s AND n.attrs -> 'tf' ? q.term
             GROUP BY q.term ORDER BY df, q.term""",
            (terms, ords or [], run.run_id))
        return cur.fetchall()


_CT_CACHE: dict = {}


def community_terms(conn, run: RunHandle, cids: list[int], k: int = 3,
                    K1: float = 1.5, B: float = 0.75) -> dict:
    """Top-k terms per community, BM25 with the COMMUNITY as the document (W9).

    tf  = term frequency summed over every member chunk's stored tf map
    len = summed n_tok of members; avg over all communities in the run
    idf = log(1 + (N_c - df + .5)/(df + .5)), df = communities containing t

    Scored over the whole community, not the retrieved slice. The stored
    `keywords` column is tf*idf over the same scope; BM25 adds the length
    term, which matters because community sizes run 8..189 here.

    Require:  cids are live cids in run. Guarantee: {cid: [term, ...]} of
              length <= k each; empty list for a cid with no tf mass.
    """
    if not cids:
        return {}
    gt = _gt()
    ck = (str(run.run_id), k, K1, B)
    if ck not in _CT_CACHE:                   # prompt-independent: compute once per run
        got = gt._disk(f"cterms-{ck[0]}-{k}-{K1}-{B}")
        if got is not None:
            _CT_CACHE[ck] = got
    if ck in _CT_CACHE:
        return {c: list(_CT_CACHE[ck].get(c, [])) for c in cids}
    with conn.cursor() as cur:
        cur.execute("""
            WITH m AS (SELECT c.cid, unnest(c.members) AS ord
                         FROM community c WHERE c.run_id = %s),
            agg AS (SELECT m.cid, kv.key AS term, sum((kv.value)::int) AS tf
                      FROM m JOIN node n ON n.run_id = %s AND n.ord = m.ord,
                           jsonb_each_text(n.attrs -> 'tf') AS kv
                     GROUP BY m.cid, kv.key),
            len AS (SELECT m.cid, sum((n.attrs ->> 'n_tok')::int) AS len
                      FROM m JOIN node n ON n.run_id = %s AND n.ord = m.ord
                     GROUP BY m.cid),
            df AS (SELECT term, count(DISTINCT cid) AS df FROM agg GROUP BY term),
            nc AS (SELECT count(*) AS n, avg(len) AS avglen FROM len)
            SELECT a.cid, a.term,
                   ln(1 + (nc.n - df.df + 0.5)/(df.df + 0.5))
                   * a.tf * (%s + 1)
                   / (a.tf + %s * (1 - %s + %s * len.len / nc.avglen)) AS bm25
              FROM agg a JOIN df USING (term) JOIN len USING (cid), nc
            """,
            (run.run_id, run.run_id, run.run_id, K1, K1, B, B))
        rows = cur.fetchall()
    allc: dict = {}
    for r in sorted(rows, key=lambda r: (r["cid"], -r["bm25"], r["term"])):
        if len(allc.setdefault(r["cid"], [])) < k:
            allc[r["cid"]].append(r["term"])
    _CT_CACHE[ck] = allc
    gt._disk_put(f"cterms-{ck[0]}-{k}-{K1}-{B}", allc)
    return {c: list(allc.get(c, [])) for c in cids}


def local_medoid(conn, run: RunHandle, ords: list[int],
                 weights: dict | None = None) -> int | None:
    """The retrieved chunk most central to what the WALK found (W11).

    centrality(o) = weight(o) * sum_j weight(j) * strength(o, j)

    Both factors are the walk score. Weighting neighbours alone still tracks
    structural centrality -- well-connected chunks are reached with high
    scores -- and returned the global medoid at 25 of 110 retrieved on the
    colonial prompt. The chunk's OWN relevance multiplies in, so the medoid is
    central to what the prompt activated AND itself activated. Without
    weights this is plain structural centrality. Ties break on the lowest ord.
    """
    if not ords:
        return None
    if len(ords) == 1:
        return ords[0]
    w = weights or {}
    tot = {o: 0.0 for o in ords}
    for e in subgraph_edges(conn, run, ords):
        a, b, st = e["src"], e["dst"], e["strength"]
        tot[a] += w.get(b, 1.0) * st
        tot[b] += w.get(a, 1.0) * st
    return min(ords, key=lambda o: (-w.get(o, 1.0) * tot[o], o))


def cross_community(conn, run: RunHandle, ords: list[int]) -> list[dict]:
    """Retrieved chunks whose walked edges reach a DIFFERENT retrieved
    community -- the in-between exemplars.

    Defined by edges, not membership: in this schema a chunk belongs to
    exactly one community, so 'in several communities' is not expressible
    until the term-node layer lands. Ranked by the number of foreign
    communities reached, then by foreign edge count.

    Guarantee: [{ord, cid, foreign_cids, n_foreign_edges}] sorted desc;
               chunks with no foreign edge are omitted.
    """
    if len(ords) < 2:
        return []
    cid_of = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.ord, c.cid FROM unnest(%s::int[]) AS v(ord)
              JOIN community c ON c.run_id = %s AND c.members @> ARRAY[v.ord]""",
            (ords, run.run_id))
        for r in cur.fetchall():
            cid_of[r["ord"]] = r["cid"]
    foreign: dict = {}
    for e in subgraph_edges(conn, run, ords):
        a, b = e["src"], e["dst"]
        ca, cb = cid_of.get(a), cid_of.get(b)
        if ca is None or cb is None or ca == cb:
            continue
        foreign.setdefault(a, {}).setdefault(cb, 0)
        foreign[a][cb] += 1
        foreign.setdefault(b, {}).setdefault(ca, 0)
        foreign[b][ca] += 1
    out = [{"ord": o, "cid": cid_of[o], "foreign_cids": sorted(f),
            "n_foreign_edges": sum(f.values())} for o, f in foreign.items()]
    return sorted(out, key=lambda r: (-len(r["foreign_cids"]),
                                      -r["n_foreign_edges"], r["ord"]))


def query_terms(conn, run: RunHandle, cids: list[int], query: str,
                k: int = 3, embed=None, pool: int = 40) -> dict:
    """Top-k terms per community AS THE PROMPT SEES IT (W10, design 6.1).

    Candidates are the community's own BM25 top-`pool`. Ranking, in order:
      1. lexical  -- a candidate that is a query term, or a phrase part of
                     one, is forced to the top
      2. dense    -- cosine(embed(query), embed(term)) when `embed` is given
      3. fallback -- the unsupervised BM25 order

    `embed`: callable list[str] -> unit-norm ndarray (n, d). None means no
    dense signal; ranking is lexical-then-unsupervised (R5 posture).

    Guarantee: out[cid] is a subset of community_terms(pool)[cid]; len <= k.
    """
    cand = community_terms(conn, run, cids, k=pool)
    qtok = set(_gt().tokenize(query))
    qv = None
    if embed is not None and qtok:
        qv = embed([query])[0]
    out: dict = {}
    for cid, terms in cand.items():
        if not terms:
            out[cid] = []
            continue
        if qv is not None:
            T = embed([t.replace("_", " ") for t in terms])
            score = {t: float(T[i] @ qv) for i, t in enumerate(terms)}
        else:
            score = {t: -i for i, t in enumerate(terms)}      # unsupervised order
        lex = {t for t in terms
               if t in qtok or any(part in qtok for part in t.split("_"))}
        out[cid] = sorted(terms, key=lambda t: (t not in lex, -score[t], t))[:k]
    return out


def _decorate(conn, run: RunHandle, rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    ords = [r["ord"] for r in rows]
    with conn.cursor() as cur:
        cur.execute("""
            SELECT n.ord, n.doc_id, left(n.body, 240) AS preview,
                   n.attrs->>'source' AS source, c.cid, c.keywords
              FROM node n
              LEFT JOIN community c
                     ON c.run_id = n.run_id AND c.members @> ARRAY[n.ord]
             WHERE n.run_id = %s AND n.ord = ANY(%s::int[])""",
            (run.run_id, ords))
        meta = {r["ord"]: r for r in cur.fetchall()}
    for r in rows:
        r.update({k: v for k, v in meta.get(r["ord"], {}).items() if k != "ord"})
    return rows


def dense_search(conn, run: RunHandle, qvec, k: int = 8) -> list[dict]:
    """S20: the DENSE entry point -- pgvector top-k over node_embedding.

    The graph has been dual-space in its EDGES since R3/R5, but its front door
    was lexical only: `search()` above is BM25 over the graph's vocabulary, and
    every anchor came from it. The HNSW index that pg_store builds at ingest
    (vector_ip_ops, pg_store._ensure_hnsw) was never queried by anything --
    written, indexed, and read back only as text for the neo4j export.

    That gap is why a question whose wording differs from the corpus's wording
    cannot start in the right neighbourhood. Measured on ab-section for "who is
    the most famous musician of the 1990's?": the word "famous" appears in ZERO
    of the 58 chunks mentioning Nirvana or Kurt Cobain (Wikipedia writes
    "best-selling", "influential", "acclaimed"), so BM25 anchoring reached none
    of them -- while by dense similarity a gold chunk sits at rank 20 of 27,259
    and the ranking-claim chunk #20762 ("regularly considered one of the
    greatest music artists of all time") at rank 137.

    Embeddings are L2-normalized at fit (R14, |norm-1| <= 1e-6), so the inner
    product `<#>` ranks identically to cosine and is cheaper. `<#>` returns the
    NEGATIVE inner product, hence ASC ordering for most-similar-first.

    Returns gt.search-shaped rows so callers cannot tell the two entry points
    apart structurally; `score` is the recovered inner product in [-1, 1].
    """
    if qvec is None or k < 1:
        return []
    lit = "[" + ",".join(f"{float(x):.7g}" for x in qvec) + "]"
    with conn.cursor() as cur:
        # The HNSW index is GLOBAL across every run in the table, while this
        # query filters to one run_id. pgvector scans hnsw.ef_search candidates
        # across the WHOLE index and only then applies the filter, so a run
        # holding a fraction of the rows gets a fraction of the candidates and
        # LIMIT is silently unreachable. Measured on ab-section (27,259 of the
        # table's rows, alongside mixed-full-dual and ab-document):
        #     LIMIT 8 -> 8 rows | LIMIT 50 -> 8 | LIMIT 200 -> 8   (ef_search=40)
        #     LIMIT 200 -> 86 rows                                 (ef_search=400)
        # and the 8 that survived were all `quotes` chunks, which is what made
        # the dense anchors look length-biased rather than starved.
        #
        # Scale the candidate pool by this run's share of the table so the
        # filter has enough to survive. Plain SET, not SET LOCAL: connect() runs
        # autocommit=True, so every statement is its own transaction and a LOCAL
        # setting is discarded before the next query -- measured, it left the
        # result at 8 rows exactly as if unset. Session scope is the right scope
        # here anyway; a larger candidate pool only costs time and accuracy is
        # monotonic in it.
        cur.execute("SELECT count(*) AS c FROM node_embedding")
        total = int(cur.fetchone()["c"] or 1)
        cur.execute("SELECT count(*) AS c FROM node_embedding WHERE run_id = %s", (run.run_id,))
        mine = int(cur.fetchone()["c"] or 1)
        want = min(1000, max(40, int(k * max(1.0, total / max(mine, 1)) * 4)))
        try:
            cur.execute(f"SET hnsw.ef_search = {want}")
        except Exception:                                          # noqa: BLE001
            pass       # not an HNSW plan (seq scan on a small table): exact anyway
        cur.execute("""SELECT e.ord AS ord, n.doc_id AS doc_id,
                              -(e.embedding <#> %s::vector) AS score,
                              n.attrs->>'source' AS source
                         FROM node_embedding e
                         JOIN node n ON n.run_id = e.run_id AND n.ord = e.ord
                        WHERE e.run_id = %s
                        ORDER BY e.embedding <#> %s::vector
                        LIMIT %s""", (lit, run.run_id, lit, k))
        return [dict(r) for r in cur.fetchall()]
