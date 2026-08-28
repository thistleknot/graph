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

import os
import re
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("CHUNKGRAPH_DSN",
                     "postgresql://graph:graph@localhost:5433/graph")

_STOP = set("""the of and to a in that is was he for it with as his on be at by i
this had not are but from or have an they which one you were her all she there
would their we him been has when who will more no if out so said what up its
about into than them can only other new some could time these two may then do
first any my now such like our over man me even most made after also did many
before must through back years where much your way well down should because""".split())


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


def search(conn, run: RunHandle, query: str, k: int = 8) -> list[dict]:
    """Lexical entry point. tf*idf over the graph's OWN vocabulary, read off
    node.attrs->'tf' via node_tf_gin. idf recomputed exactly as fit() defines
    it: log(1 + (n - df + .5)/(df + .5)).

    W2: attrs is used for SCORING here, never for traversal."""
    terms = tokenize(query)
    if not terms:
        return []
    with conn.cursor() as cur:
        cur.execute("""
            WITH q AS (SELECT unnest(%s::text[]) AS term),
            df AS (SELECT q.term, count(n.ord) AS df
                     FROM q LEFT JOIN node n
                       ON n.run_id = %s AND n.attrs -> 'tf' ? q.term
                    GROUP BY q.term),
            hit AS (SELECT n.ord, q.term, (n.attrs -> 'tf' ->> q.term)::int AS tf
                      FROM node n JOIN q ON n.attrs -> 'tf' ? q.term
                     WHERE n.run_id = %s)
            SELECT h.ord,
                   sum(h.tf * ln(1 + ((%s::float - d.df + 0.5)/(d.df + 0.5)))) AS score,
                   count(DISTINCT h.term) AS terms_hit
              FROM hit h JOIN df d USING (term)
             GROUP BY h.ord ORDER BY score DESC LIMIT %s""",
            (terms, run.run_id, run.run_id, run.n_chunks, k))
        rows = cur.fetchall()
    return _decorate(conn, run, rows)


def node(conn, run: RunHandle, ord_: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT n.ord, n.doc_id, n.body, (n.attrs->>'n_tok')::int AS n_tok,
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
