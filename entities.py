"""entities.py -- Entities v0: bipartite entity/mention store beside the chunk
graph, built over the run's EXISTING salient vocabulary (node.attrs->'tf').

Spec: .spec/specs/graph-explorer/design.md §6.15 block C, guards E1-E5
Task: playbook.md T4

E1  Entities v0 SHALL be populated from the run's EXISTING salient/phrase vocabulary
    (the terms already persisted in node.attrs->'tf'), typed "term_v0", by
    deterministic NLP only -- no LLM call, no NER dependency. v0 exists to fix the
    SHAPE, a bipartite entity-mention store beside the chunk graph, not to improve
    extraction; real NER is v1 and SHALL replace the population without changing
    these tables.

E2  Every entities table SHALL be run-scoped with run_id as the first column and the
    leading key predicate (W1), and rows SHALL cascade with graph_run. Chunk identity
    SHALL be the run-local node ordinal `ord`, the key node, community.members and
    the walker already use; no new chunk id space SHALL be minted (X1's lesson one
    level down).

E3  The builder SHALL be READ-ONLY against node, edge and community -- it reads
    node.attrs and writes only entities, mentions and entity_edges -- and SHALL be
    re-runnable: a second build over the same run REPLACES that run's rows and leaves
    every other run's rows untouched.

E4  WHERE a pair of entities co-occurs in fewer than MIN_JOINT_CHUNKS (= 5) chunks,
    NO entity_edges row SHALL be written. NPMI on a joint count of one or two is
    dominated by rare-event bias -- a pair occurring exactly once, together, scores
    1.0 by construction -- so an unfloored ranking is a list of hapax coincidences
    rather than of associations.

E5  entity_edges SHALL carry npmi, ppmi and bm25 on the SAME row, computed over the
    same chunk-level co-occurrence counts, so choosing between weightings is a column
    choice at read time and never a rebuild. bm25 SHALL be the symmetric sum of both
    directions and SHALL use graph_tools.search's constants (K1 = 1.5, B = 0.75), so
    entity weighting and retrieval weighting share one ruler. a < b SHALL hold on
    every row: one row per undirected pair, the same law `edge` carries.

Require:  a live run exists for the given label (graph_tools.get_run).
Guarantee: build_entities() is read-only against node/edge/community and
          idempotent per run -- a rebuild replaces exactly that run's three
          tables' rows, byte-for-byte reproducible (deterministic entity ids),
          leaving every other run untouched (E3).

DDL lives HERE, not in sql/001_schema.sql: 001_schema.sql is an init migration
that runs once on an empty volume, so it cannot create tables on the existing
populated dev database. The three tables below are therefore created
idempotently by the builder itself (`ensure_schema`, CREATE TABLE IF NOT
EXISTS) -- design.md block C already states this is the correct call. Do not
"fix" this by moving the DDL into sql/.
"""
from __future__ import annotations

import math
import os
import sys

import psycopg
from psycopg.rows import dict_row

import graph_tools as gt

DSN = os.environ.get("CHUNKGRAPH_DSN",
                     "postgresql://graph:graph@localhost:5433/graph")
MIN_JOINT_CHUNKS = 5           # E4
ENTITY_TYPE = "term_v0"        # E1
K1, B = 1.5, 0.75              # E5: graph_tools.search's constants, one ruler

_DDL = (
    """CREATE TABLE IF NOT EXISTS entities (
        run_id    uuid    NOT NULL REFERENCES graph_run ON DELETE CASCADE,
        entity_id integer NOT NULL,
        name      text    NOT NULL,
        type      text    NOT NULL DEFAULT 'term_v0',
        PRIMARY KEY (run_id, entity_id),
        UNIQUE (run_id, name)
    )""",
    """CREATE TABLE IF NOT EXISTS mentions (
        run_id    uuid    NOT NULL,
        ord       int     NOT NULL,
        entity_id integer NOT NULL,
        cnt       integer NOT NULL,
        PRIMARY KEY (run_id, ord, entity_id),
        FOREIGN KEY (run_id, ord)       REFERENCES node (run_id, ord)          ON DELETE CASCADE,
        FOREIGN KEY (run_id, entity_id) REFERENCES entities (run_id, entity_id) ON DELETE CASCADE
    )""",
    """CREATE INDEX IF NOT EXISTS mentions_entity ON mentions (run_id, entity_id)""",
    """CREATE TABLE IF NOT EXISTS entity_edges (
        run_id uuid    NOT NULL,
        a      integer NOT NULL,
        b      integer NOT NULL,
        npmi   real    NOT NULL,
        ppmi   real    NOT NULL,
        bm25   real    NOT NULL,
        PRIMARY KEY (run_id, a, b),
        CHECK (a < b),
        FOREIGN KEY (run_id, a) REFERENCES entities (run_id, entity_id) ON DELETE CASCADE,
        FOREIGN KEY (run_id, b) REFERENCES entities (run_id, entity_id) ON DELETE CASCADE
    )""",
    """CREATE INDEX IF NOT EXISTS entity_edges_npmi ON entity_edges (run_id, npmi DESC)""",
)


def ensure_schema(conn) -> None:
    """Runtime-idempotent: every statement is CREATE ... IF NOT EXISTS, safe
    to run on every build (§0)."""
    with conn.cursor() as cur:
        for stmt in _DDL:
            cur.execute(stmt)


def read_vocab(conn, run) -> dict:
    """Wraps graph_tools.corpus_index -- reuse the cached per-run inverted
    index rather than re-querying. Transposes the term-major `post` map into
    a chunk-major `tf` map, which is the shape the builder needs.

    Returns {"tf": {ord: {term: cnt}}, "df": {term: int}, "dl": {ord: float},
             "avgdl": float, "n": int}.
    """
    ix = gt.corpus_index(conn, run)
    tf_by_ord: dict = {}
    for term, post in ix["post"].items():
        for ord_, cnt in post.items():
            tf_by_ord.setdefault(ord_, {})[term] = cnt
    return {"tf": tf_by_ord, "df": ix["df"], "dl": ix["dl"],
            "avgdl": ix["avgdl"], "n": ix["n"]}


def bm25_weight(tf: float, dl: float, df: int, n: int, avgdl: float,
                K1: float = K1, B: float = B) -> float:
    """One term's BM25 weight in one chunk (E5's w(y,c))."""
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))


def npmi_ppmi(df_a: int, df_b: int, joint: int, n: int) -> tuple[float, float]:
    """pmi = log(p(a,b) / (p(a)*p(b))); npmi = pmi / -log(p(a,b)); ppmi =
    max(pmi, 0).

    Degenerate case (decided, §5): joint == n means both terms are in every
    chunk, so p(a,b) == 1.0 and -log(p(a,b)) == 0 -- return (0.0, 0.0) rather
    than divide by zero. A pair present in every chunk distinguishes nothing
    [colloquial: convention chosen here, not a derived result].
    `joint >= MIN_JOINT_CHUNKS` guarantees p(a,b) > 0, so the low side needs
    no guard.
    """
    if joint >= n:
        return 0.0, 0.0
    p_a, p_b, p_ab = df_a / n, df_b / n, joint / n
    pmi = math.log(p_ab / (p_a * p_b))
    npmi = pmi / -math.log(p_ab)
    ppmi = max(pmi, 0.0)
    return npmi, ppmi


def pair_bm25(a, b, joint_ords: list, tf_by_ord: dict, df: dict, dl: dict,
             n: int, avgdl: float) -> float:
    """E5: symmetric sum of both directions. bm25_dir(a->b) sums w(b,c) over
    chunks containing a, and w(b,c) == 0 wherever tf(b,c) == 0, so only the
    joint chunks contribute -- this closed form over `joint_ords` IS
    bm25_dir(a->b) + bm25_dir(b->a), not a divergence from the spec."""
    df_a, df_b = df[a], df[b]
    total = 0.0
    for c in joint_ords:
        tf_a = tf_by_ord[c][a]
        tf_b = tf_by_ord[c][b]
        dl_c = dl[c]
        total += bm25_weight(tf_a, dl_c, df_a, n, avgdl)   # w(a,c)
        total += bm25_weight(tf_b, dl_c, df_b, n, avgdl)   # w(b,c)
    return total


def pair_counts(tf_by_ord: dict, df: dict, min_joint: int = MIN_JOINT_CHUNKS) -> dict:
    """Maps canonical (a, b), a < b -> list of joint ords, floor already
    applied (E4, inclusive boundary at min_joint).

    Prefilter to df >= min_joint before enumerating pairs: lossless, since
    joint(a, b) <= min(df(a), df(b)) -- a term below the floor can never
    reach it, so dropping it first is the only reason the O(V^2) pair
    enumeration stays bounded per chunk.
    """
    floor_ok = {t for t, d in df.items() if d >= min_joint}
    pairs: dict = {}
    for ord_, terms in tf_by_ord.items():
        present = sorted(t for t in terms if t in floor_ok)
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                pairs.setdefault((a, b), []).append(ord_)
    return {pair: ords for pair, ords in pairs.items() if len(ords) >= min_joint}


def build_entities(conn, run, min_joint: int = MIN_JOINT_CHUNKS) -> dict:
    """The only writer. `conn` must be a WRITABLE connection -- gt.connect()
    is read-only at the server (W3) and raises on the first CREATE TABLE.

    Read-only law (E3): the only statements issued against node/edge/community
    are the SELECTs inside gt.corpus_index. No UPDATE, INSERT or DDL on those
    tables.
    """
    ensure_schema(conn)
    v = read_vocab(conn, run)
    names = sorted(v["df"])
    eid = {name: i for i, name in enumerate(names)}   # ascending, zero-based:
                                                        # deterministic (E3)
    df_by_id = {eid[name]: v["df"][name] for name in names}
    tf_by_ord_id: dict = {
        ord_: {eid[t]: cnt for t, cnt in terms.items()}
        for ord_, terms in v["tf"].items()
    }
    pairs = pair_counts(tf_by_ord_id, df_by_id, min_joint)

    rid = run.run_id
    with conn.cursor() as cur:
        # E3: REPLACES this run's rows only -- FK order, never TRUNCATE, the
        # run_id predicate is mandatory on every DELETE.
        cur.execute("DELETE FROM entity_edges WHERE run_id = %s", (rid,))
        cur.execute("DELETE FROM mentions WHERE run_id = %s", (rid,))
        cur.execute("DELETE FROM entities WHERE run_id = %s", (rid,))

        with cur.copy("COPY entities (run_id, entity_id, name, type) FROM STDIN") as cp:
            for name in names:
                cp.write_row((rid, eid[name], name, ENTITY_TYPE))

        n_mentions = 0
        with cur.copy("COPY mentions (run_id, ord, entity_id, cnt) FROM STDIN") as cp:
            for ord_, terms in tf_by_ord_id.items():
                for tid, cnt in terms.items():
                    cp.write_row((rid, ord_, tid, int(cnt)))
                    n_mentions += 1

        with cur.copy("COPY entity_edges (run_id, a, b, npmi, ppmi, bm25) FROM STDIN") as cp:
            for (a, b), joint_ords in pairs.items():
                npmi, ppmi = npmi_ppmi(df_by_id[a], df_by_id[b], len(joint_ords), v["n"])
                bm25 = pair_bm25(a, b, joint_ords, tf_by_ord_id, df_by_id, v["dl"],
                                 v["n"], v["avgdl"])
                cp.write_row((rid, a, b, npmi, ppmi, bm25))
    conn.commit()

    return {"n_chunks": v["n"], "entities": len(names), "mentions": n_mentions,
            "edges": len(pairs)}


def main(argv: list) -> int:
    if len(argv) < 2:
        print("usage: python entities.py <label>", file=sys.stderr)
        return 2
    label = argv[1]
    conn = psycopg.connect(DSN, row_factory=dict_row)
    try:
        try:
            run = gt.get_run(conn, label)
        except LookupError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        result = build_entities(conn, run)
    finally:
        conn.close()
    print(f"{label} {run.run_id} N={result['n_chunks']} "
          f"entities={result['entities']} mentions={result['mentions']} "
          f"edges={result['edges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
