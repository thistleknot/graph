"""
graph_tools.py — deterministic, index-safe traversal primitives over a persisted
ChunkGraph run. This is the tool surface an LLM caller is handed; it is NOT an
LLM itself. Every function here is a plain parameterized SELECT.

WHY THIS SHAPE
The graph is built deterministically -- chunking, edges, fusion and Louvain all
run at ingest with no model in the loop. A caller that walks it at query time
therefore chooses only the ROUTE; the substrate and every community label it
reports were fixed up front. That is what keeps an agentic walk auditable: the
sequence of calls below fully reconstructs what was seen and why.

The builder exists so a tool-caller CANNOT express an unsafe query. There is no
sql() primitive and no string interpolation of identifiers. Guardrail, not sugar.

TOOL SURFACE
| tool                | question it answers                          | bound  |
|---------------------|----------------------------------------------|--------|
| list_runs           | which runs exist                             | all    |
| get_run             | identity + capability of one run             | 1 row  |
| search              | where do I start, lexically                  | k      |
| node                | what is this chunk, who owns it              | 1 row  |
| neighbors           | where can I step from here                   | limit  |
| community           | what is this cid about                       | 1 row  |
| communities_touched | what does this node set partake in           | n cids |
| bridges             | which edges actually connect two communities | limit  |
| quotient            | how communities interconnect, corpus-wide    | limit  |
| subgraph_edges      | induced edges among a visited set (drawing)  | set    |
| walk                | reach + WHY: prov_path/doc_path per result   | cap    |
| term_stats          | which query terms exist, and who carries them| terms  |
| community_terms     | top-k BM25 terms, community as the document  | k/cid  |
| query_terms         | those terms re-ranked by the prompt          | k/cid  |
| chunk_salient       | a chunk's own BM25 terms, gated (W14), top-k | ord    |
| pathways            | idea-to-idea DWPC over the walked subgraph (W15) | ords, anchors |
| chunk_terms         | the top-k of that, for titles                | k/ord  |
| local_medoid        | most central retrieved chunk in a community  | 1/cid  |
| cross_community     | retrieved chunks bridging retrieved cids     | set    |
| second_order         | which terms keep this term's company (W17)   | k/term |
| node_metrics        | betweenness/pagerank/triangles per node (W18) | run   |
| community_metrics   | density + conductance per stored cid (W19)   | cids  |

GUARDS (EARS)
W1  Every statement SHALL be run-scoped: run_id is the first predicate, so the
    (run_id, ...) index prefixes are usable and cross-run bleed is impossible.
W2  Traversal SHALL read edge / edge_sym via the indexed integer src/dst columns
    only. attrs is read for payload and lexical scoring, NEVER as a traversal or
    join key (steering hard constraint 3).
W3  The connection SHALL be opened read-only at the server. A write raises
    psycopg.errors.ReadOnlySqlTransaction rather than relying on convention.
W4  Every tool returning rows SHALL take an explicit bound.
W5  WHERE a run has no dense space (embed_dim NULL or zero embedding rows),
    capability SHALL report sparse-only rather than failing.
W6  Live rows only: valid_to IS NULL on edges, superseded_at IS NULL on runs
    (the live_run view). Note graph_run has NO valid_to column.
W8  term_stats() SHALL report a query term the corpus does NOT contain, with
    df 0, rather than dropping it. A term absent from the run's vocabulary is
    the single most common reason a retrieval looks wrong, and silently
    omitting it makes the query look like it asked for less than it did.
W9  community_terms() SHALL score over EVERY member of the community, never
    only the retrieved ones. The ranked terms are the implied evidence -- the
    concept the community holds -- and scoring them on the retrieved subset
    collapses that back into a search result.
W10 query_terms() SHALL choose only from the community's OWN vocabulary
    (community_terms pool). The prompt re-ranks; it never imports a term the
    community does not carry. Communities stay unsupervised -- only the
    three words shown for each are conditioned on the prompt.
W11 local_medoid() SHALL weight BOTH the chunk and its neighbours by walk
    score: centrality(o) = w(o) * sum_j w(j) * strength(o, j). Neighbour
    weighting alone tracks structural centrality and returned the global
    medoid at 25 of 110 retrieved; the chunk's own relevance must multiply in.
W12 search() SHALL be BM25 -- tf saturation (k1) and length normalisation (b)
    on the stored n_tok -- not tf*idf. Measured at document-level nodes: plain
    tf*idf anchored on long essays dense in common query words and missed 19
    of 23 documents carrying the rare, decisive term.
W14 Which of a chunk's terms are SALIENT is a gate, not a top-k: log-normalise
    the BM25 scores and keep all at or above min(median - 1.4826*MAD, mean -
    sd). Both branches are one sigma below a centre on the same scale; the
    disjunction keeps the upper half and a little more (trigram.md's dual
    measure, measured 24/24 probe recall there vs 20-21/24 for a lone median).
    Titles show the top-k OF the kept set; the kept set is the vocabulary.
W7  walk() SHALL return, per reached node, the EDGE PROVENANCE and SOURCE DOC of
    every hop that reached it -- not merely the node ids traversed. A route that
    cannot name its own justification is a browser, not an evidence instrument.
    The arrays ride the recursive CTE at no extra join: edge_sym already exposes
    provenance and both endpoint docs.
W15 Evidence pathways are computed on the induced subgraph only, with GLOBAL
    degrees for damping and conductance; a pair's DWPC sums simple paths of
    <= max_len edges, each scored prod(strength) * prod(deg^-damp) over every
    node on the path, so no path outranks the same path through smaller hubs.
W16 Bridge discovery searches the WHOLE live edge set: best_path maximises
    prod(strength) x prod(deg^-damp) over interior nodes (Dijkstra on the
    negative log). It finds evidence the walk missed; pathways (W15) only
    scores what was already retrieved. Two jobs, two functions.
W17 second_order_terms() SHALL apply the ladder as a gate-then-fallback chain,
    in order: (a) Dunning-LLR co-occurrence gate (rung "llr") admits only
    candidates whose 2x2 association with the target clears g2_gate, and
    min_df floors BOTH the target and every candidate before anything else
    runs; (b) Schutze context-centroid cosine (rung "centroid") ranks the
    LLR survivors by shared company, not raw co-occurrence; (c) a skew
    diagnostic over the survivors' cosine scores decides whether the
    centroid ranking is trustworthy -- ONLY when it trips (|skew| >
    skew_trip, and only past min_skew_n survivors) does Mann-Whitney AUC
    re-rank the top_k (rung "auc"); otherwise the centroid ordering from (b)
    stands untouched. WHERE a run has no dense space (W5), the ladder
    degrades to the LLR ordering (rung "llr", cos/auc None) rather than
    failing -- rungs (b)/(c) are a dense-only refinement, never a
    requirement. The v0 nomen pool is a df-band + stoplist floor (df in
    [min_df, max_df_frac*n], len > 2, not stoplisted); true PPMI demotion of
    the high-frequency band is LATER, once entities v0's ppmi table exists.
W18 node_metrics() SHALL compute the centrality lane over the WHOLE run and cache
    it by (run_id, k, seed); `ords` filters the returned view, never the
    computation, so two callers asking about different chunks get numbers off the
    same graph. Betweenness is exact at or below BETWEENNESS_EXACT_MAX nodes and
    the k-sample estimator above it, with k and seed PINNED as constants -- an
    unseeded sample would make the number a coin flip and the cache a lie.
    Betweenness and clustering are computed unweighted: networkx reads `weight`
    as a DISTANCE and our strengths are similarities, so passing them would make
    the strongest edges the longest. PageRank, where higher weight genuinely is
    closer, uses strength, with alpha and tol pinned. Per-provenance degree
    (sparse/dense/both) is read from edge_sym's provenance column and SHALL sum
    to the node's degree.
W19 community_metrics() SHALL score the STORED partition -- density and
    conductance per cid read from community.members, never a re-partitioning of
    an induced subgraph -- and SHALL report the whole-run WCC (component count
    and sizes) beside it, because a modularity partition of a shattered graph is
    a number without a shape. The run-wide summary is median and p90, not a mean:
    community sizes are heavy-tailed and a mean reports the tail.
W20 Personalized PageRank from the walk anchors is an ADDITIVE named column in
    pathways(): every pair keeps its dwpc, the pair ordering stays DWPC's, and
    `ppr` rides beside it. Two measures of connectedness, named separately, so a
    later swap is a measured decision rather than a silent one. Nothing consumes
    these metrics yet -- surfacing is T22's job.
W21 Alias expansion SHALL be opt-in (`expand_aliases`, default OFF) and ADDITIVE:
    a query token that is itself an entity surface form SHALL contribute the other
    names sharing its `canonical_id` as extra OR-terms at full BM25 weight, and NO
    original query term SHALL be dropped or reweighted. Matching SHALL be exact
    against `entities.name` -- `tokenize` and the entity vocabulary are the same
    vocabulary, and a looser match reopens the false merges E7 exists to close.
    WHERE resolution has not run for a run -- no `entities` table, no rows, or
    `canonical_id` NULL -- the map SHALL read back empty and search SHALL be
    byte-identical to the unexpanded call, never an error.

NOT HERE, DELIBERATELY
- No LLM. No prompt, no model call, no NL->query translation.
- No re-partitioning. cid is read from community, never recomputed over an
  induced subgraph (subgraph Louvain does not restrict global Louvain).
- No writes of any kind.

SOURCE (R20)
`source` is read from `attrs->>'source'` and is metadata only -- it never
filters, weights, or steers a traversal (W2 still holds). A run predating R20
has no `source` key, so it reads back as `None` and renders unlabelled rather
than failing (design.md §6.14 R20).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row

import config

DSN = config.DSN

from stoplist import _STOP                   # R18: one stoplist, no heavy imports


def tokenize(text: str) -> list[str]:
    """Mirrors chunkgraph._tok: lowercase alpha, >2 chars, stopwords dropped.
    Phrases are NOT merged -- the engine does not merge query phrases either,
    and diverging would silently change what is matched."""
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in _STOP and len(w) > 2]


def expand_terms(terms: list[str], aliases: dict) -> list[str]:
    """W21: alias expansion is ADDITIVE. A query token that IS an entity
    surface form contributes its whole alias set as extra OR-terms; every
    other token passes through untouched. Deterministic: sorted, deduped.

    Full weight, no discount: siblings enter the BM25 loop as ordinary terms
    with their own idf. BM25 already penalises a rarer alias through idf and
    saturates its tf, so a hand-picked expansion discount would be a second,
    unmeasured knob stacked on the one the scorer applies -- and this campaign
    has three wins for additive-never-displace and none for re-weighting.
    """
    out = set(terms)
    for t in terms:
        out.update(aliases.get(t, ()))
    return sorted(out)


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
    terms = tokenize(query)
    if not terms:
        return []
    if expand_aliases:
        aliases = alias_map(conn, run)
        if aliases:
            terms = expand_terms(terms, aliases)
    ix = corpus_index(conn, run)
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
    terms = tokenize(query)
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
    ck = (str(run.run_id), k, K1, B)
    if ck not in _CT_CACHE:                   # prompt-independent: compute once per run
        got = _disk(f"cterms-{ck[0]}-{k}-{K1}-{B}")
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
    _disk_put(f"cterms-{ck[0]}-{k}-{K1}-{B}", allc)
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
    qtok = set(tokenize(query))
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


_DF_CACHE: dict = {}
_CT_CACHE: dict = {}
CACHE_DIR = config.CACHE_DIR


def _disk(key: str):
    """Runs are immutable (supersede-never-delete), so anything derived from a
    run_id can live on disk forever. Measured: the in-process index build is
    20 s on 500 documents; a pickle reload is well under a second."""
    import pickle
    path = os.path.join(CACHE_DIR, key + ".pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None


def _disk_put(key: str, obj) -> None:
    import pickle, tempfile
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, os.path.join(CACHE_DIR, key + ".pkl"))
    except OSError:
        pass                                         # cache is an optimisation, never a failure


def corpus_index(conn, run: RunHandle) -> dict:
    """Per-run inverted index, built ONCE and cached by run_id:
    {df: {term: df}, post: {term: {ord: tf}}, dl: {ord: n_tok}, avgdl, n}.
    One pass over the stored tf maps. Measured: the SQL BM25 over jsonb took
    7.3 s per query on 500 documents; this makes a query sub-millisecond."""
    key = str(run.run_id)
    if key not in _DF_CACHE:
        cached = _disk("index-" + key)
        if cached is not None:
            _DF_CACHE[key] = cached
            return cached
        post: dict = {}; dl: dict = {}
        with conn.cursor() as cur:
            cur.execute("""SELECT n.ord, (n.attrs->>'n_tok')::float AS dl,
                                  kv.key AS term, (kv.value)::float AS tf
                             FROM node n, jsonb_each_text(n.attrs -> 'tf') AS kv
                            WHERE n.run_id = %s""", (run.run_id,))
            for r in cur.fetchall():
                post.setdefault(r["term"], {})[r["ord"]] = r["tf"]
                dl[r["ord"]] = r["dl"] or 1.0
        n = len(dl) or 1
        _DF_CACHE[key] = {"df": {t: len(p) for t, p in post.items()}, "post": post,
                          "dl": dl, "avgdl": (sum(dl.values()) / n) if dl else 1.0, "n": n}
        _disk_put("index-" + key, _DF_CACHE[key])
    return _DF_CACHE[key]


_ALIAS_CACHE: dict = {}


def alias_map(conn, run: RunHandle) -> dict:
    """{name: (sibling, ...)} for every entity that has at least one alias.

    An alias set is exactly the rows sharing a canonical_id (entities.py E8).
    Siblings exclude the term itself and are sorted, so the map is a pure
    function of the run's rows.

    Degrades to {} where entity resolution has not run for this run -- a
    missing `entities` table, no rows, or canonical_id still NULL. That is a
    real state (T19 resolved planted fixtures only), and search must not
    depend on it.

    No disk cache. `_disk` is justified by runs being immutable; canonical_id
    is rewritten by every resolve_entities call, so this is not derived-from-
    an-immutable-run and must not outlive the process."""
    key = str(run.run_id)
    if key in _ALIAS_CACHE:
        return _ALIAS_CACHE[key]
    if len(_ALIAS_CACHE) > 32:
        _ALIAS_CACHE.clear()
    out: dict = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.name AS name, s.name AS sibling
                  FROM entities e
                  JOIN entities s ON s.run_id = e.run_id
                                 AND s.canonical_id = e.canonical_id
                                 AND s.entity_id <> e.entity_id
                 WHERE e.run_id = %s AND e.canonical_id IS NOT NULL""",
                (run.run_id,))
            for r in cur.fetchall():
                out.setdefault(r["name"], set()).add(r["sibling"])
    except psycopg.Error:
        out = {}
    out = {name: tuple(sorted(sibs)) for name, sibs in out.items()}
    _ALIAS_CACHE[key] = out
    return out


def corpus_df(conn, run: RunHandle) -> tuple[dict, float, int]:
    """(df per term, avgdl, n_chunks) -- a view over corpus_index."""
    ix = corpus_index(conn, run)
    return ix["df"], ix["avgdl"], ix["n"]


def salient_gate(scores: dict) -> dict:
    """Which terms count as salient (W14). Log-normalise the BM25 scores, then
    keep every term at or above  min(median - 1.4826*MAD, mean - sd)  -- the
    more permissive of the robust and parametric centres, each one sigma down.
    Both branches are on the same sigma scale (1.4826 makes MAD normal-
    consistent). Measured: keeps ~84% on real chunks.

    Guarantee: {kept: [terms by score desc], threshold_decile: int in 1..10,
                n_in: int, n_kept: int}. Fewer than 4 terms: keep all."""
    import numpy as np
    if not scores:
        return {"kept": [], "threshold_decile": 0, "n_in": 0, "n_kept": 0}
    terms = sorted(scores, key=lambda t: (-scores[t], t))
    if len(terms) < 4:
        return {"kept": terms, "threshold_decile": 1, "n_in": len(terms), "n_kept": len(terms)}
    x = np.log1p(np.array([max(scores[t], 0.0) for t in terms], float))
    med = float(np.median(x)); mad = float(np.median(np.abs(x - med)))
    thr = min(med - 1.4826 * mad, float(x.mean() - x.std()))
    kept = [t for t, v in zip(terms, x) if v >= thr]
    below = int((x < thr).sum())
    decile = max(1, min(10, int(round(10 * below / len(x))) + 1))
    return {"kept": kept, "threshold_decile": decile, "n_in": len(terms), "n_kept": len(kept)}


def chunk_salient(conn, run: RunHandle, ords: list[int], k: int = 3,
                  K1: float = 1.5, B: float = 0.75) -> dict:
    """Per chunk: BM25 of its OWN terms against the corpus (chunk as document,
    idf over the run), gated by salient_gate (W14), top-k for the title.

    Guarantee: {ord: {top, kept, threshold_decile, n_in, n_kept}}."""
    if not ords:
        return {}
    df, avgdl, N = corpus_df(conn, run)
    out: dict = {}
    for o in ords:
        nd = node(conn, run, o)
        tf = (nd or {}).get("tf") or {}
        dl = float((nd or {}).get("n_tok") or 1)
        sc = {}
        for term, f in tf.items():
            d = df.get(term, 1); f = float(f)
            idf = math.log(1 + (N - d + 0.5) / (d + 0.5))
            sc[term] = idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
        gate = salient_gate(sc)
        gate["top"] = gate["kept"][:k]
        out[o] = gate
    return out


def chunk_terms(conn, run: RunHandle, ords: list[int], k: int = 3) -> dict:
    """Top-k salient terms of each chunk -- what a medoid is titled with."""
    return {o: v["top"] for o, v in chunk_salient(conn, run, ords, k=k).items()}


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


BETWEENNESS_EXACT_MAX = 2000      # nodes; at or below this, betweenness is exact
BETWEENNESS_K         = 512       # pivot sample above the floor
BETWEENNESS_SEED      = 20260903  # fixed; the sample is a parameter, not a coin flip
PR_ALPHA, PR_TOL, PR_MAX_ITER  = 0.85, 1.0e-08, 200
PPR_ALPHA, PPR_TOL, PPR_MAX_ITER = 0.85, 1.0e-10, 200   # restart mass = 1 - PPR_ALPHA


def degrees(conn, run: RunHandle, ords: list[int]) -> dict:
    """Global degree of each ordinal in the run's live edge set (W15)."""
    if not ords:
        return {}
    with conn.cursor() as cur:
        cur.execute("""SELECT a AS ord, count(*) AS deg FROM edge_sym
                        WHERE run_id = %s AND valid_to IS NULL AND a = ANY(%s)
                        GROUP BY a""", (run.run_id, list(ords)))
        d = {r["ord"]: r["deg"] for r in cur.fetchall()}
    return {o: d.get(o, 0) for o in ords}


def pathways(conn, run: RunHandle, ords: list[int], anchors: list[int],
             max_len: int = 3, damp: float = 0.4, top_pairs: int = 12,
             ppr_alpha: float = PPR_ALPHA) -> dict:
    """W15 (design 6.11): critical connectedness between idea nodes.

    Induced subgraph = `ords` and the edges among them. Anchors are any
    ordinals in `ords` (medoids, top chunks -- the caller decides what an
    "idea" is). For every anchor pair: DWPC = sum over simple paths (<= max_len
    edges) of prod(edge strength) * prod(global deg(v)^-damp, every node on the
    path). Also returns the subgraph's shape: components, largest component
    share, density, and conductance of `ords` against the rest of the run.
    Deterministic; edge table only.
    """
    ords = list(dict.fromkeys(ords))
    anchors = [a for a in dict.fromkeys(anchors) if a in set(ords)]
    edges = subgraph_edges(conn, run, ords)
    adj: dict = {o: {} for o in ords}
    for e in edges:
        w = max(min(e["strength"], 1.0), 1e-9)
        adj[e["src"]][e["dst"]] = w
        adj[e["dst"]][e["src"]] = w
    deg = degrees(conn, run, ords)

    # ---- shape
    seen, comps = set(), []
    for o in ords:
        if o in seen:
            continue
        stack, comp = [o], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack += [v for v in adj[u] if v not in comp]
        seen |= comp
        comps.append(len(comp))
    n, e_in = len(ords), len(edges)
    vol_s = sum(deg.values())
    with conn.cursor() as cur:
        cur.execute("""SELECT count(*) AS m FROM edge_sym
                        WHERE run_id = %s AND valid_to IS NULL""", (run.run_id,))
        m_total = cur.fetchone()["m"]          # directed rows = 2x undirected edges
    cut = max(vol_s - 2 * e_in, 0)
    vol_rest = max(m_total - vol_s, 1)
    comps.sort(reverse=True)
    shape = {"n": n, "edges": e_in, "components": len(comps),
             "wcc_sizes": comps,
             "largest_component_frac": (max(comps) / n) if n else 0.0,
             "density": (2 * e_in / (n * (n - 1))) if n > 1 else 0.0,
             "conductance": cut / min(max(vol_s, 1), vol_rest)}

    # ---- DWPC per anchor pair
    dmp = {o: (deg.get(o, 0) or 1) ** (-damp) for o in ords}
    pairs = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            total, best, best_p, count = 0.0, 0.0, None, 0
            stack = [(a, [a], dmp[a])]
            while stack:
                u, path, sc = stack.pop()
                for v, w in adj[u].items():
                    if v in path:
                        continue
                    nsc = sc * w * dmp[v]
                    if v == b:
                        total += nsc
                        count += 1
                        if nsc > best:
                            best, best_p = nsc, path + [v]
                    elif len(path) <= max_len - 1:
                        stack.append((v, path + [v], nsc))
            if count:
                pairs.append({"a": a, "b": b, "dwpc": total, "n_paths": count,
                              "path": best_p})
    pairs.sort(key=lambda p: (-p["dwpc"], p["a"], p["b"]))

    # ---- W20: personalized PPR, additive -- DWPC still orders `pairs`
    node_ppr = ppr(adj, anchors, alpha=ppr_alpha)
    per_anchor = {a: ppr(adj, [a], alpha=ppr_alpha) for a in anchors}
    for pair in pairs:
        p_a, p_b = per_anchor.get(pair["a"], {}), per_anchor.get(pair["b"], {})
        pair["ppr"] = (p_a.get(pair["b"], 0.0) + p_b.get(pair["a"], 0.0)) / 2

    return {**shape, "pairs": pairs[:top_pairs], "ppr": node_ppr, "ppr_alpha": ppr_alpha}


_ADJ_CACHE: dict = {}


def full_adjacency(conn, run: RunHandle) -> dict:
    """W16: whole-run adjacency {a: {b: strength}}, cached per run (runs are
    immutable). ~2x edge count entries; fine to hold for graphs this size."""
    key = str(run.run_id)
    if key in _ADJ_CACHE:
        return _ADJ_CACHE[key]
    adj: dict = {}
    with conn.cursor() as cur:
        cur.execute("""SELECT a, b, strength FROM edge_sym
                        WHERE run_id = %s AND valid_to IS NULL""", (run.run_id,))
        for r in cur.fetchall():
            adj.setdefault(r["a"], {})[r["b"]] = max(min(r["strength"], 1.0), 1e-9)
    _ADJ_CACHE[key] = adj
    return adj


def best_path(conn, run: RunHandle, a: int, b: int, damp: float = 0.4):
    """W16 (design 6.13): the single strongest degree-damped path a..b over the
    WHOLE run -- Dijkstra minimising sum(-log strength) + damp*sum(log deg) over
    interior nodes, i.e. maximising the DWPC weight of one path. Returns
    (path list, weight in (0, 1]) or None when disconnected."""
    import heapq
    import math
    adj = full_adjacency(conn, run)
    if a not in adj or b not in adj:
        return None
    def node_cost(v):
        return damp * math.log(max(len(adj.get(v, {})), 1))
    dist = {a: 0.0}
    prev = {}
    heap = [(0.0, a)]
    seen = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        if u == b:
            break
        for v, w in adj[u].items():
            nd = d + -math.log(w) + (node_cost(v) if v != b else 0.0)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if b not in dist:
        return None
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    return path, math.exp(-dist[b])


# --------------------------------------- named graph metrics (W18/W19/W20)
def graph_metrics(adj: dict, k_sample: int | None = None,
                  seed: int = BETWEENNESS_SEED,
                  alpha: float = PR_ALPHA, tol: float = PR_TOL) -> dict:
    """W18. Centrality lane over {a: {b: strength}}. Pure -- no conn, unit-
    testable on planted data. Betweenness and clustering are UNWEIGHTED
    (networkx reads `weight` as a distance, and our strengths are
    similarities); PageRank is weighted, where higher strength genuinely
    means closer.

    Returns {"nodes": {ord: {"betweenness","pagerank","triangles",
    "clustering","degree"}}, "approx": bool,
    "params": {"k": k|None, "seed", "alpha", "tol"}, "n": int}.
    """
    import networkx as nx

    nodes = sorted(adj)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for a in nodes:
        for b in sorted(adj[a]):
            if a < b:
                G.add_edge(a, b, weight=adj[a][b])

    k = None if k_sample is None else min(k_sample, len(G))
    betweenness = nx.betweenness_centrality(G, k=k, seed=seed, weight=None,
                                            normalized=True)
    pagerank = nx.pagerank(G, alpha=alpha, tol=tol, max_iter=PR_MAX_ITER,
                           weight="weight")
    triangles = nx.triangles(G)
    clustering = nx.clustering(G, weight=None)
    approx = k is not None

    out_nodes = {o: {"betweenness": betweenness[o], "pagerank": pagerank[o],
                     "triangles": triangles[o], "clustering": clustering[o],
                     "degree": len(adj[o])} for o in nodes}
    return {"nodes": out_nodes, "approx": approx,
           "params": {"k": k, "seed": seed, "alpha": alpha, "tol": tol},
           "n": len(nodes)}


def _pct(sorted_xs: list, q: float) -> float:
    """Linear-interpolated percentile over an already-sorted list. 0.0 on
    empty input."""
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    idx = q * (len(sorted_xs) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return float(sorted_xs[lo])
    frac = idx - lo
    return float(sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac)


def partition_metrics(adj: dict, members: dict) -> dict:
    """W19. Density + conductance of the STORED partition -- members =
    {cid: [ord, ...]}; nothing is recomputed. Same algebra as pathways()'s
    shape block, deliberately, so the two numbers are comparable. Pure --
    no conn.

    Returns {"communities": [{cid,size,internal_edges,volume,cut,density,
    conductance}, ...] ordered by cid, "wcc": {"components","sizes",
    "largest_frac"}, "summary": {"density_median","density_p90",
    "conductance_median","conductance_p90","n_communities"}}.
    """
    e_total = sum(len(v) for v in adj.values()) / 2.0

    communities = []
    for cid in sorted(members):
        S = set(members[cid])
        n = len(S)
        volume = sum(len(adj.get(o, {})) for o in S)
        internal_edges = sum(1 for o in S for b in adj.get(o, {}) if b in S) // 2
        cut = max(volume - 2 * internal_edges, 0)
        vol_rest = max(2 * e_total - volume, 1)
        conductance = cut / min(max(volume, 1), vol_rest)
        density = (2 * internal_edges / (n * (n - 1))) if n > 1 else 0.0
        communities.append({"cid": cid, "size": n, "internal_edges": internal_edges,
                            "volume": volume, "cut": cut, "density": density,
                            "conductance": conductance})

    # ---- whole-run WCC (iterative DFS, no recursion -- mirrors pathways())
    seen, comps = set(), []
    for o in sorted(adj):
        if o in seen:
            continue
        stack, comp = [o], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack += [v for v in adj[u] if v not in comp]
        seen |= comp
        comps.append(len(comp))
    comps.sort(reverse=True)
    n_total = len(adj)
    wcc = {"components": len(comps), "sizes": comps,
          "largest_frac": (max(comps) / n_total) if n_total else 0.0}

    densities = sorted(c["density"] for c in communities)
    conductances = sorted(c["conductance"] for c in communities)
    summary = {"density_median": _pct(densities, 0.5),
              "density_p90": _pct(densities, 0.9),
              "conductance_median": _pct(conductances, 0.5),
              "conductance_p90": _pct(conductances, 0.9),
              "n_communities": len(communities)}
    return {"communities": communities, "wcc": wcc, "summary": summary}


def ppr(adj: dict, seeds: list[int], alpha: float = PPR_ALPHA,
       tol: float = PPR_TOL, max_iter: int = PPR_MAX_ITER) -> dict:
    """W20. Personalized PageRank over {a:{b:w}} restarting on `seeds`
    (uniform personalization). Hand-rolled power iteration -- no nx import,
    keeping the additive pathways() column free of a new dependency on the
    hot path. Dangling nodes send their mass back to the restart vector.

    Returns {ord: score} summing to 1.0 over adj's keys; {} when seeds is
    empty, adj is empty, or none of seeds are in adj.
    """
    nodes = sorted(adj)
    seeds_in = [s for s in dict.fromkeys(seeds) if s in adj]
    if not nodes or not seeds_in:
        return {}
    n = len(nodes)
    idx = {o: i for i, o in enumerate(nodes)}
    p = [0.0] * n
    share0 = 1.0 / len(seeds_in)
    for s in seeds_in:
        p[idx[s]] = share0

    x = list(p)
    for _ in range(max_iter):
        new_x = [0.0] * n
        dangling_mass = 0.0
        for i, o in enumerate(nodes):
            neigh = adj[o]
            deg_w = sum(neigh.values())
            if deg_w <= 0:
                dangling_mass += x[i]
                continue
            share = alpha * x[i] / deg_w
            for b, w in neigh.items():
                j = idx.get(b)
                if j is not None:
                    new_x[j] += share * w
        restart_mass = (1 - alpha) + alpha * dangling_mass
        for i in range(n):
            new_x[i] += restart_mass * p[i]
        diff = sum(abs(new_x[i] - x[i]) for i in range(n))
        x = new_x
        if diff < tol:
            break
    return {nodes[i]: x[i] for i in range(n)}


_METRIC_CACHE: dict = {}
_COMM_METRIC_CACHE: dict = {}


def node_metrics(conn, run: RunHandle, ords: list[int] | None = None) -> dict:
    """W18. Whole-run centrality lane + per-provenance degree split, cached
    per run. `ords` filters the RETURNED view; it never changes what is
    computed."""
    ck = str(run.run_id)
    disk_key = f"nodemetrics-{ck}-{BETWEENNESS_K}-{BETWEENNESS_SEED}"
    if ck not in _METRIC_CACHE:
        cached = _disk(disk_key)
        if cached is not None:
            _METRIC_CACHE[ck] = cached
    if ck not in _METRIC_CACHE:
        adj = full_adjacency(conn, run)
        k = None if len(adj) <= BETWEENNESS_EXACT_MAX else BETWEENNESS_K
        core = graph_metrics(adj, k_sample=k)
        with conn.cursor() as cur:
            cur.execute("""SELECT a AS ord, provenance, count(*) AS c FROM edge_sym
                            WHERE run_id = %s AND valid_to IS NULL
                            GROUP BY a, provenance""", (run.run_id,))
            prov_rows = cur.fetchall()
        prov: dict = {}
        for r in prov_rows:
            prov.setdefault(r["ord"], {})[r["provenance"]] = r["c"]
        for o, row in core["nodes"].items():
            p = prov.get(o, {})
            row["prov"] = p
            row["prov_degree_total"] = sum(p.values())
        _METRIC_CACHE[ck] = core
        _disk_put(disk_key, core)
    core = _METRIC_CACHE[ck]
    all_nodes = core["nodes"]
    if ords is None:
        view = dict(all_nodes)
    else:
        zero = {"betweenness": 0.0, "pagerank": 0.0, "triangles": 0,
               "clustering": 0.0, "degree": 0, "prov": {}}
        view = {o: all_nodes.get(o, dict(zero)) for o in ords}
    return {**core, "nodes": view}


def community_metrics(conn, run: RunHandle) -> dict:
    """W19. Per-stored-cid density + conductance, WCC sanity, run-wide
    median/p90."""
    ck = str(run.run_id)
    disk_key = f"commmetrics-{ck}"
    if ck not in _COMM_METRIC_CACHE:
        cached = _disk(disk_key)
        if cached is not None:
            _COMM_METRIC_CACHE[ck] = cached
    if ck not in _COMM_METRIC_CACHE:
        with conn.cursor() as cur:
            cur.execute("""SELECT cid, members FROM community
                            WHERE run_id = %s ORDER BY cid""", (run.run_id,))
            rows = cur.fetchall()
        members = {r["cid"]: list(r["members"]) for r in rows}
        adj = full_adjacency(conn, run)
        result = partition_metrics(adj, members)
        _COMM_METRIC_CACHE[ck] = result
        _disk_put(disk_key, result)
    return _COMM_METRIC_CACHE[ck]


# ---------------------------------------------------------------- dendrites
def dendrite_sort(M, names, alpha: float = 0.05, min_support: int = 0):
    """Correlation-chain decomposition -- the operator's 'dendrite sorting'
    (correlation sorting.md, 2026-09-02). Columns of M are the variables;
    rows are the observations (n for significance).

    Require: M is (n_obs, n_var) float array-like; names labels the columns.
    Guarantee: every kept variable lands in exactly ONE chain; chains are
      internally collinear threads (each hop is the tail's strongest
      significant unassigned partner), mutually decollinear groups (a new
      chain opens at the unassigned variable with the highest MEAN significant
      correlation -- the most connected remaining feature); a chain terminates
      WHEN no significant unassigned candidate remains. Deterministic:
      ties break by name order.
    Maintain: preprocessing per column is signed log1p -> Yeo-Johnson ->
      median/MAD z, so heavy-tailed, zero-heavy profiles (BM25 columns)
      correlate on rank-like scales. Columns with fewer than `min_support`
      nonzero observations are dropped before sorting (Pearson has nothing
      to grip on a near-empty profile).

    Returns {"chains": [[name,...],...], "r": (v,v) ndarray, "sig": bool
    ndarray, "names": kept names, "dropped": [...]} -- r/sig are aligned to
    the kept-name order.
    """
    import numpy as np
    from scipy.stats import t as _t, yeojohnson

    M = np.asarray(M, dtype=float)
    n_obs = M.shape[0]
    keep = [j for j in range(M.shape[1])
            if np.count_nonzero(M[:, j]) >= min_support and np.ptp(M[:, j]) > 0]
    dropped = [names[j] for j in range(M.shape[1]) if j not in set(keep)]
    names = [names[j] for j in keep]
    M = M[:, keep]
    v = len(names)
    if v == 0 or n_obs < 5:
        return {"chains": [], "r": np.zeros((0, 0)), "sig": np.zeros((0, 0), bool),
                "names": [], "dropped": dropped}

    P = np.empty_like(M)
    for j in range(v):
        x = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
        try:
            x = yeojohnson(x)[0]
        except Exception:
            pass                                     # degenerate: keep log scale
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        P[:, j] = (x - med) / (mad if mad else 1.0)

    R = np.atleast_2d(np.corrcoef(P, rowvar=False)) if v > 1 else np.zeros((1, 1))
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.abs(R) * np.sqrt((n_obs - 2) / np.maximum(1e-12, 1 - R ** 2))
    p = 2 * _t.sf(tt, df=n_obs - 2)
    sig = (p < alpha) & (R > 0)                       # chains ride positive links
    np.fill_diagonal(sig, False)

    def mean_sig(j, pool):
        vals = [R[j, k] for k in pool if k != j and sig[j, k]]
        return float(np.mean(vals)) if vals else 0.0

    unassigned = set(range(v))
    chains = []
    while unassigned:
        master = max(sorted(unassigned), key=lambda j: (mean_sig(j, unassigned), -j))
        chain = [master]
        unassigned.remove(master)
        tail = master
        while True:
            cands = [k for k in sorted(unassigned) if sig[tail, k]]
            if not cands:
                break                                 # the chain terminates here
            tail = max(cands, key=lambda k: (R[tail, k], -k))
            chain.append(tail)
            unassigned.remove(tail)
        chains.append([names[j] for j in chain])
    return {"chains": chains, "r": R, "sig": sig, "names": names,
            "dropped": dropped}


def subgraph_embeddings(conn, run: RunHandle, ords: list[int]):
    """Stored dense embeddings for a set of ords, L2-normalized, as
    (ndarray, kept_ords). Ords without a stored embedding are omitted."""
    import numpy as np
    with conn.cursor() as cur:
        cur.execute("""SELECT ord, embedding::text AS e FROM node_embedding
                        WHERE run_id = %s AND ord = ANY(%s) ORDER BY ord""",
                    (run.run_id, list(ords)))
        rows = cur.fetchall()
    if not rows:
        return np.zeros((0, 0)), []
    E = np.array([np.fromstring(r["e"].strip("[]"), sep=",") for r in rows])
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return E, [r["ord"] for r in rows]


# ------------------------------------------ second-order terms (W17)
def llr(k11, k12, k21, k22) -> float:
    """Dunning log-likelihood ratio (G2) on a 2x2 chunk-level co-occurrence
    table. Basis: .tmp/second_order_probe.py (measured probe, 2026-09-02);
    no governing REQ, W17 minted here."""
    def h(*ks):
        tot = sum(ks)
        return sum(k * math.log(k / tot) for k in ks if k > 0)
    return 2 * (h(k11, k12, k21, k22) - h(k11 + k12, k21 + k22)
                - h(k11 + k21, k12 + k22) + h(k11 + k12 + k21 + k22))


def second_order_terms(M, names, E, target, g2_gate: float = 10.83,
                       min_df: int = 5, skew_trip: float = 2.0,
                       min_skew_n: int = 8, top_k: int = 30) -> dict:
    """Which terms keep TARGET's company (W17). Pure, array-in/array-out --
    mirrors dendrite_sort's (M, names, ...) shape, no DB. Rows of M are
    chunks, columns are terms, nonzero = present; E is the matching chunk
    embedding matrix (or None/empty for a sparse-only run).

    Ladder: (1) Dunning-LLR gate on chunk-level co-occurrence admits only
    candidates significantly associated with the target and above min_df;
    (2) Schutze context-centroid cosine ranks the survivors by shared
    company; (3) a skew diagnostic over the survivors' cosine scores decides
    whether that ranking is trustworthy -- only when it trips does
    Mann-Whitney AUC re-rank the top_k. See W17.

    Basis: .tmp/second_order_probe.py (measured probe, 2026-09-02); no
    governing REQ, W17 minted here.

    Require: target in names, else LookupError.
    Guarantee: target never appears in its own ranked; sub-floor terms never
      enter the candidate pool and land in dropped (sorted); a sub-floor
      target returns ranked=[] with reason="target_df_below_min"; an empty
      candidate pool returns ranked=[] with reason="no_candidates"; every
      sort key is (-score, term) -- total and deterministic.
    """
    import numpy as np
    names = list(names)
    if target not in names:
        raise LookupError(f"target {target!r} not in vocabulary")
    ti = names.index(target)
    M = np.asarray(M)
    present = M != 0
    n_chunks = present.shape[0]
    df_arr = present.sum(axis=0)

    dropped = sorted(names[j] for j in range(len(names))
                     if j != ti and df_arr[j] < min_df)

    if df_arr[ti] < min_df:
        return {"target": target, "rung": "llr", "skew": 0.0, "ranked": [],
                "n_candidates": 0, "n_survivors": 0, "dropped": dropped,
                "reason": "target_df_below_min"}

    candidates = [j for j in range(len(names)) if j != ti and df_arr[j] >= min_df]
    n_candidates = len(candidates)
    if not candidates:
        return {"target": target, "rung": "llr", "skew": 0.0, "ranked": [],
                "n_candidates": 0, "n_survivors": 0, "dropped": dropped,
                "reason": "no_candidates"}

    tidx = np.nonzero(present[:, ti])[0]
    tset = set(tidx.tolist())
    N = n_chunks

    survivors = []                                    # (j, g2, member_idx)
    for j in candidates:
        idx = np.nonzero(present[:, j])[0]
        m = set(idx.tolist())
        k11 = len(tset & m)
        k12 = len(tset) - k11
        k21 = len(m) - k11
        k22 = N - k11 - k12 - k21
        g2 = llr(k11, k12, k21, k22)
        if g2 >= g2_gate:
            survivors.append((j, g2, idx))
    n_survivors = len(survivors)

    have_dense = E is not None and np.asarray(E).size > 0
    if n_survivors == 0 or not have_dense:
        ranked_rows = sorted(survivors, key=lambda x: (-x[1], names[x[0]]))[:top_k]
        ranked = [{"term": names[j], "g2": float(g2), "df": int(df_arr[j]),
                   "cos": None, "auc": None} for j, g2, _ in ranked_rows]
        return {"target": target, "rung": "llr", "skew": 0.0, "ranked": ranked,
                "n_candidates": n_candidates, "n_survivors": n_survivors,
                "dropped": dropped, "reason": None}

    E = np.asarray(E, dtype=float)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    def centroid(idx):
        v = E[idx].mean(axis=0)
        n = np.linalg.norm(v)
        return v / (n if n > 1e-12 else 1.0)

    v_target = centroid(tidx)
    scored = [(j, g2, idx, float(v_target @ centroid(idx))) for j, g2, idx in survivors]
    ranked_centroid = sorted(scored, key=lambda x: (-x[3], names[x[0]]))

    cos_vals = np.array([x[3] for x in ranked_centroid])
    skew = 0.0
    if len(cos_vals) > min_skew_n and cos_vals.std() > 0:
        m3 = ((cos_vals - cos_vals.mean()) ** 3).mean() / cos_vals.std() ** 3
        n_ = len(cos_vals)
        skew = m3 * math.sqrt(n_ * (n_ - 1)) / max(n_ - 2, 1)
    trip = abs(skew) > skew_trip

    if not trip:
        rows = ranked_centroid[:top_k]
        ranked = [{"term": names[j], "g2": float(g2), "df": int(df_arr[j]),
                   "cos": cos, "auc": None} for j, g2, _, cos in rows]
        return {"target": target, "rung": "centroid", "skew": skew,
                "ranked": ranked, "n_candidates": n_candidates,
                "n_survivors": n_survivors, "dropped": dropped, "reason": None}

    top_rows = ranked_centroid[:top_k]
    is_t = np.zeros(N, dtype=bool)
    if tset:
        is_t[list(tset)] = True
    m_ = len(tset)
    reranked = []
    for j, g2, idx, cos in top_rows:
        s_all = E @ centroid(idx)
        r = s_all.argsort().argsort() + 1
        auc = float((r[is_t].sum() - m_ * (m_ + 1) / 2) / (m_ * (N - m_)))
        reranked.append((j, g2, cos, auc))
    reranked.sort(key=lambda x: (-x[3], names[x[0]]))
    ranked = [{"term": names[j], "g2": float(g2), "df": int(df_arr[j]),
               "cos": cos, "auc": auc} for j, g2, cos, auc in reranked]
    return {"target": target, "rung": "auc", "skew": skew, "ranked": ranked,
            "n_candidates": n_candidates, "n_survivors": n_survivors,
            "dropped": dropped, "reason": None}


def second_order(conn, run: RunHandle, target, ords: list[int] | None = None,
                 nomen_pool: list[str] | None = None, top_nomens: int = 400,
                 min_df: int = 5, max_df_frac: float = 0.5, k: int = 8) -> dict:
    """Thin DB-facing assembler for second_order_terms (W17): builds M,
    names, E from the corpus index and stored embeddings, then delegates.
    No ladder logic lives here.

    v0 nomen pool (when nomen_pool is None): terms with len > 2, not
    stoplisted, and df in [min_df, max_df_frac*n_chunks], ranked by df desc
    then term asc, truncated to top_nomens. This is a df-band + stoplist
    floor, NOT PPMI -- the upper max_df_frac band is the PPMI demotion's
    cheap proxy (a term carried by half the corpus co-occurs with
    everything, so its PMI against any target is near zero).
    # LATER: true PPMI demotion (entities v0 already computes ppmi in
    # entity_edges); swap this band for it once that table exists.
    Pass nomen_pool explicitly to bypass the floor entirely.

    Basis: .tmp/second_order_probe.py (measured probe, 2026-09-02); no
    governing REQ, W17 minted here.
    """
    import numpy as np
    ix = corpus_index(conn, run)
    if ords is None:
        ords = sorted(ix["dl"])
    E, kept = subgraph_embeddings(conn, run, ords)
    if kept:
        universe = kept
    else:
        universe = ords
        E = None

    if nomen_pool is None:
        n = ix["n"]
        cands = [t for t, d in ix["df"].items()
                 if len(t) > 2 and t not in _STOP and min_df <= d <= max_df_frac * n]
        cands.sort(key=lambda t: (-ix["df"][t], t))
        pool = cands[:top_nomens]
    else:
        pool = list(nomen_pool)

    names = list(pool)
    if target not in names:
        names = names + [target]
    post = ix["post"]
    M = np.zeros((len(universe), len(names)))
    for j, t in enumerate(names):
        p = post.get(t, {})
        for i, o in enumerate(universe):
            v = p.get(o)
            if v:
                M[i, j] = v

    result = second_order_terms(M, names, E, target, min_df=min_df)
    result["ranked"] = result["ranked"][:k]
    result["n_nomens"] = len(pool)
    result["universe"] = len(universe)
    return result
