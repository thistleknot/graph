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
| chunk_terms         | the top-k of that, for titles                | k/ord  |
| local_medoid        | most central retrieved chunk in a community  | 1/cid  |
| cross_community     | retrieved chunks bridging retrieved cids     | set    |

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

NOT HERE, DELIBERATELY
- No LLM. No prompt, no model call, no NL->query translation.
- No re-partitioning. cid is read from community, never recomputed over an
  induced subgraph (subgraph Louvain does not restrict global Louvain).
- No writes of any kind.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("CHUNKGRAPH_DSN",
                     "postgresql://graph:graph@localhost:5433/graph")

from stoplist import _STOP                   # R18: one stoplist, no heavy imports


def tokenize(text: str) -> list[str]:
    """Mirrors chunkgraph._tok: lowercase alpha, >2 chars, stopwords dropped.
    Phrases are NOT merged -- the engine does not merge query phrases either,
    and diverging would silently change what is matched."""
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in _STOP and len(w) > 2]


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
           K1: float = 1.5, B: float = 0.75) -> list[dict]:
    """Lexical entry point: BM25 over the graph's OWN vocabulary (W12), scored
    in Python over the cached per-run postings (corpus_index). Was SQL over
    jsonb -- 7.3 s per query on 500 documents, called twice per walk.

    W2: attrs is used for SCORING here, never for traversal."""
    terms = tokenize(query)
    if not terms:
        return []
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
                   n.attrs -> 'tf' AS tf, c.cid, c.keywords
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
                   c.cid, c.keywords
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
            SELECT c.cid, c.size, c.keywords, c.medoid_text, count(*) AS hits
              FROM community c
              JOIN unnest(%s::int[]) AS v(ord) ON c.members @> ARRAY[v.ord]
             WHERE c.run_id = %s
             GROUP BY c.cid, c.size, c.keywords, c.medoid_text
             ORDER BY hits DESC""", (ords, run.run_id))
        return cur.fetchall()


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
CACHE_DIR = os.environ.get("CHUNKGRAPH_CACHE",
                           os.path.join(os.path.expanduser("~"), ".cache", "chunkgraph"))


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
            SELECT n.ord, n.doc_id, left(n.body, 240) AS preview, c.cid, c.keywords
              FROM node n
              LEFT JOIN community c
                     ON c.run_id = n.run_id AND c.members @> ARRAY[n.ord]
             WHERE n.run_id = %s AND n.ord = ANY(%s::int[])""",
            (run.run_id, ords))
        meta = {r["ord"]: r for r in cur.fetchall()}
    for r in rows:
        r.update({k: v for k, v in meta.get(r["ord"], {}).items() if k != "ord"})
    return rows
