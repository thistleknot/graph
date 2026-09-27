"""sparsevec_store.py -- a BM25-over-pieces lexical index as pgvector sparsevec.

NO GOVERNING SPEC for this module's own requirements. Basis: operator instruction
2026-09-26 -- "use this vocab size for sparsevec in psql and test recall with it".
Task: playbook.md T115. Skill: ~/.skills/bpe-bm25 (steps 9-10, the psql lane).

The whole point, stated once: with chunk vector d[t] = BM25(t, chunk) over piece
t and query vector q[t] = 1 for each query piece, the inner product <d, q> is the
SUM of the chunk's BM25 weights over the query's pieces -- which is EXACTLY the
score every recall number in this layer was computed with (`BM[:, cols].sum(1)`).
So an exact scan in Postgres must reproduce the numpy recall to the digit, and
the HNSW index's only contribution is approximation loss and latency. That is
what this lane measures.

    STAGE     MECHANISM
    SCHEMA    lex_vocab (piece -> column) + one lex_chunk_<label> per vocabulary,
              because sparsevec needs its dimension in the column type and a
              vocabulary IS a dimension (V1)
    WRITE     COPY text rows '{i:w,...}/dim', 1-based indices, top-1000 by weight
              when a chunk has more nonzeros than pgvector allows (V2)
    INDEX     HNSW sparsevec_ip_ops (V3)
    QUERY     ORDER BY vec <#> q  -- negative inner product, smaller is better (V4)

Guards:

V1  A vocabulary SHALL be one table, dimension baked into the column type:
    `vec sparsevec(<n_pieces>)`. pgvector will not build an HNSW index on an
    untyped sparsevec, and two vocabularies of different size cannot share a
    typed column. Table-per-label is the honest shape for an experiment that
    compares vocabularies; it is NOT the shape for production, where one
    vocabulary wins and the others are dropped.

V2  WHERE a chunk carries more than SPARSEVEC_MAX_NNZ (1000) nonzero pieces, its
    vector SHALL keep the 1000 HEAVIEST by BM25 weight and drop the rest. This
    is the pgvector 0.8.1 hard limit, measured to bind on 293 of 168,794 arxiv
    chunks (0.17%) -- all of them docling units that were never split (max
    3,395,651 chars, 28,312 distinct terms). The chunk text is untouched; only
    its index vector is truncated. The count truncated SHALL be reported.

V3  Indexes SHALL be sparsevec_ip_ops. Chunk vectors are NOT normalised (BM25
    magnitude is the signal), so cosine ops would change the ranking; inner
    product is the one that equals the numpy score.

V4  Exactness SHALL be measured, not assumed: the same queries run with
    `enable_indexscan = off` (exact) and with the HNSW index at each ef_search,
    and the exact-scan ranking is asserted equal to numpy's before any HNSW
    number is read.

V5  `SET`, never `SET LOCAL`: connections here are autocommit, where SET LOCAL
    is a silent no-op (measured in this repo, playbook T98).

DDL lives HERE via ensure_schema, not in sql/, per entities.py:129-135.
"""
from __future__ import annotations

import time
from collections.abc import Iterable

import psycopg

import config

DSN = config.DSN
SPARSEVEC_MAX_NNZ = 1000        # pgvector 0.8.1 hard limit (V2)


def _table(label: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in label.lower())
    return "lex_chunk_" + safe


def connect(dsn: str = DSN):
    """Write-capable, autocommit (so V5 applies)."""
    return psycopg.connect(dsn, autocommit=True)


def ensure_schema(conn, label: str, dim: int) -> str:
    """V1: vocab table shared, one typed chunk table per label. Idempotent."""
    t = _table(label)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_vocab (
                           label text NOT NULL,
                           col   integer NOT NULL,
                           piece text NOT NULL,
                           idf   real,
                           PRIMARY KEY (label, col),
                           UNIQUE (label, piece))""")
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {t} (
                            ord    integer PRIMARY KEY,
                            doc_id text NOT NULL,
                            source text,
                            nnz    integer NOT NULL,
                            truncated boolean NOT NULL DEFAULT false,
                            vec    sparsevec({int(dim)}) NOT NULL)""")
    return t


def write_vocab(conn, label: str, pieces: list[str], idf=None) -> None:
    """Replace the vocabulary rows for `label`. col is 1-based (sparsevec is)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM lex_vocab WHERE label = %s", (label,))
        with cur.copy("COPY lex_vocab (label, col, piece, idf) FROM STDIN") as cp:
            for j, p in enumerate(pieces):
                cp.write_row((label, j + 1, p,
                              float(idf[j]) if idf is not None else None))


def _literal(entries: dict[int, float] | list[tuple[int, float]], dim: int) -> str:
    """pgvector sparsevec text form: '{i:w,i:w}/dim', 1-based, ascending i."""
    items = sorted(entries.items() if isinstance(entries, dict) else entries)
    return "{" + ",".join("%d:%.6g" % (i + 1, w) for i, w in items) + "}/%d" % dim


def write_chunks(conn, label: str, dim: int,
                 rows: Iterable[tuple[int, str, str, dict[int, float]]]) -> dict:
    """rows = (ord, doc_id, source, {col0: weight}). Truncates per V2. Returns counts."""
    t = _table(label)
    n = trunc = 0
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {t}")
        with cur.copy(f"COPY {t} (ord, doc_id, source, nnz, truncated, vec) FROM STDIN") as cp:
            for ord_, doc_id, source, entries in rows:
                items = list(entries.items())
                cut = len(items) > SPARSEVEC_MAX_NNZ
                if cut:
                    items.sort(key=lambda kv: -kv[1])
                    items = items[:SPARSEVEC_MAX_NNZ]
                    trunc += 1
                if not items:
                    continue          # an all-masked chunk has no vector; skip, count
                cp.write_row((ord_, doc_id, source, len(items), cut,
                              _literal(items, dim)))
                n += 1
    return {"written": n, "truncated": trunc, "table": t}


def create_hnsw(conn, label: str, m: int = 16, ef_construction: int = 64) -> float:
    """V3. Returns build seconds."""
    t = _table(label)
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {t}_hnsw")
        cur.execute(f"CREATE INDEX {t}_hnsw ON {t} USING hnsw (vec sparsevec_ip_ops) "
                    f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})")
    return time.time() - t0


def query(conn, label: str, dim: int, qcols: list[int], k: int,
          exact: bool = False, ef_search: int | None = None) -> list[tuple[int, float]]:
    """V4: top-k (ord, inner_product). Binary query vector over 0-based qcols."""
    if not qcols:
        return []
    t = _table(label)
    q = _literal({c: 1.0 for c in qcols}, dim)
    with conn.cursor() as cur:
        cur.execute("SET enable_indexscan = %s" % ("off" if exact else "on"))   # V5
        if ef_search is not None and not exact:
            cur.execute("SET hnsw.ef_search = %d" % int(ef_search))
        cur.execute(f"SELECT ord, -(vec <#> %s::sparsevec) AS ip FROM {t} "
                    f"ORDER BY vec <#> %s::sparsevec LIMIT %s", (q, q, int(k)))
        return [(r[0], float(r[1])) for r in cur.fetchall()]


def counts(conn, label: str) -> dict:
    t = _table(label)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*), sum(truncated::int), max(nnz), "
                    f"pg_total_relation_size('{t}') FROM {t}")
        n, tr, mx, size = cur.fetchone()
    return {"rows": n, "truncated": tr or 0, "max_nnz": mx, "bytes": size}
