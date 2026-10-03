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

V6  WHERE a corpus is ingested for serving (tools/ingest_arxiv_sparsevec.py), IDF
    SHALL live on the QUERY side only (sparsevec-lexsem-graph R6): the stored vector
    holds saturated tf, `query(weights=idf*qtf)` supplies the rest, and the idf column
    of lex_vocab is that table. <q, d> then equals the idf-baked BM25 the earlier
    diagnostics scored with. Chunk text, section title, is_reference and the business
    key (doc_id, section_idx, chunk_idx) live in lex_chunk_meta, joined on (label, ord).

V7  What the arxiv graph derives SHALL be kept in Postgres beside the chunks, not only in
    .tmp (operator 2026-10-03: "is this data stored in psql?"). Per label: `lex_cos_<label>`
    holds the symmetric doc-doc view (saturated tf * idf, L2 rows) under
    sparsevec_cosine_ops (sparsevec-lexsem-graph R7: cosine for doc-doc, inner product for
    queries); `lex_dense_<label>` holds the mean-centered, L2 dense vectors under
    vector_cosine_ops (R9). One row per build in `lex_build`; its communities, the
    community each chunk landed in (`how` says consensus or nearest-centroid), exemplars
    and model-authored summaries hang off build_id. A summary is a DRAFT over a frozen
    community, never a join key (structure.md, determinism boundary). Builds are
    superseded, never deleted: `live` marks the current one per label.
V9  A paper is identified by its doc_id ('arxiv/<arXiv id without version>', domain_corpora C11) and carries
    its `version` as a secondary key, 1 when none was given. A higher version OVERWRITES the paper
    (delete_papers, then the new chunks are appended); the same or a lower one is never read. This is
    overwrite on purpose, not supersede-and-keep (operator 2026-10-03: "version as overwriting
    mechanism"): the markdown files remain the source of every version.
V8  `lex_chunk_meta.is_junk` flags extraction debris (domain_corpora C10) and `text_md5`
    lets an incremental ingest keep identical chunk texts once, as the full chunker does
    (C8). Junk and reference chunks are stored and flagged, never dropped.

DDL lives HERE via ensure_schema, not in sql/, per entities.py:129-135.
"""
from __future__ import annotations

import time
from collections.abc import Iterable

import psycopg

import config

DSN = config.DSN
SPARSEVEC_MAX_NNZ = 1000        # pgvector 0.8.1 hard limit (V2)


_KIND_PREFIX = {"chunk": "lex_chunk_", "cos": "lex_cos_", "dense": "lex_dense_"}


def _table(label: str, kind: str = "chunk") -> str:
    safe = "".join(c if c.isalnum() else "_" for c in label.lower())
    return _KIND_PREFIX[kind] + safe


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


def reset_label_tables(conn, label: str) -> None:
    """V1, V7. Drop the label's chunk, cosine and dense tables. A full build replaces them because the
    vocabulary IS the sparse dimension, baked into the column type, and a rebuilt vocabulary changes it.
    Chunk metadata, vocabulary rows and build records are not touched here."""
    with conn.cursor() as cur:
        for kind in _KIND_PREFIX:
            cur.execute(f"DROP TABLE IF EXISTS {_table(label, kind)}")


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
                 rows: Iterable[tuple[int, str, str, dict[int, float]]],
                 replace: bool = True, kind: str = "chunk") -> dict:
    """rows = (ord, doc_id, source, {col0: weight}). Truncates per V2. `replace` empties the
    table first (a full build); False appends (an incremental ingest). kind "cos" writes the
    V7 doc-doc view. Returns counts."""
    t = _table(label, kind)
    n = trunc = 0
    with conn.cursor() as cur:
        if replace:
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


_HNSW_OPS = {"chunk": ("vec", "sparsevec_ip_ops"), "cos": ("vec", "sparsevec_cosine_ops"),
             "dense": ("emb", "vector_cosine_ops")}


def create_hnsw(conn, label: str, m: int = 16, ef_construction: int = 64, kind: str = "chunk") -> float:
    """V3, V7. Returns build seconds. kind picks the table and its operator class: the query
    index is inner product, the doc-doc and dense views are cosine."""
    t = _table(label, kind)
    col, ops = _HNSW_OPS[kind]
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {t}_hnsw")
        cur.execute(f"CREATE INDEX {t}_hnsw ON {t} USING hnsw ({col} {ops}) "
                    f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})")
    return time.time() - t0


def ensure_chunk_meta(conn) -> None:
    """V6. One row per chunk: the text and everything retrieval filters or cites by."""
    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_chunk_meta (
                           label         text    NOT NULL,
                           ord           integer NOT NULL,
                           doc_id        text    NOT NULL,
                           section_idx   integer NOT NULL,
                           chunk_idx     integer NOT NULL,
                           section_title text    NOT NULL,
                           is_reference  boolean NOT NULL,
                           text          text    NOT NULL,
                           PRIMARY KEY (label, ord),
                           UNIQUE (label, doc_id, section_idx, chunk_idx))""")
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS "
                    "is_junk boolean NOT NULL DEFAULT false")                                   # V8
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS "
                    "text_md5 text GENERATED ALWAYS AS (md5(text)) STORED")                     # V8
        cur.execute("CREATE INDEX IF NOT EXISTS lex_chunk_meta_md5 ON lex_chunk_meta (label, text_md5)")
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS "
                    "version integer NOT NULL DEFAULT 1")                                       # V9
        cur.execute("CREATE INDEX IF NOT EXISTS lex_chunk_meta_doc ON lex_chunk_meta (label, doc_id)")


_META_COLS = ("label, ord, doc_id, version, section_idx, chunk_idx, section_title, is_reference, is_junk, text")


def _copy_meta(cur, label: str, records: list[dict], first_ord: int) -> None:
    with cur.copy(f"COPY lex_chunk_meta ({_META_COLS}) FROM STDIN") as cp:
        for i, r in enumerate(records):
            cp.write_row((label, first_ord + i, r["doc_id"], int(r.get("version", 1)), r["section_idx"], r["chunk_idx"],
                          r["section_title"], r["is_reference"], bool(r.get("is_junk", False)), r["text"]))


def write_chunk_meta(conn, label: str, records: list[dict]) -> int:
    """V6, V8. Replace the label's rows; ord is the record's position. Returns rows written."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM lex_chunk_meta WHERE label = %s", (label,))
        _copy_meta(cur, label, records, 0)
    return len(records)


def append_chunk_meta(conn, label: str, records: list[dict], min_ord: int = 0) -> int:
    """V8, V9. Add records after the label's last ord, keeping every existing row, never below `min_ord`: pass
    the live build's chunk count so that an overwrite which emptied the top of the ord range cannot hand
    those ords out again (they belong to the build, and the churn count reads them as its own). Returns
    the first new ord (the records occupy first..first+len-1)."""
    with conn.cursor() as cur:
        cur.execute("SELECT greatest(coalesce(max(ord) + 1, 0), %s) FROM lex_chunk_meta WHERE label = %s", (int(min_ord), label))
        first = cur.fetchone()[0]
        _copy_meta(cur, label, records, first)
    return first


def known_text_md5(conn, label: str, texts: list[str]) -> set[str]:
    """V8. Guarantee: the md5 of each of `texts` that the label already stores (C8, across runs)."""
    import hashlib
    want = {hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts}
    if not want:
        return set()
    with conn.cursor() as cur:
        cur.execute("SELECT text_md5 FROM lex_chunk_meta WHERE label = %s AND text_md5 = ANY(%s)",
                    (label, list(want)))
        return {r[0] for r in cur.fetchall()}


def stored_versions(conn, label: str) -> dict[str, int]:
    """V9. Guarantee: {doc_id: version held} for every paper the label stores, so an incremental ingest takes
    only papers that are new or newer."""
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, max(version) FROM lex_chunk_meta WHERE label = %s GROUP BY doc_id", (label,))
        return {r[0]: r[1] for r in cur.fetchall()}


def delete_papers(conn, label: str, doc_ids: list[str], build_id: int | None = None) -> int:
    """V9. Overwrite: remove every chunk of `doc_ids` -- metadata, the three vector tables, and (given the live
    build) its assignments and exemplars -- so a higher version can take the paper's place. Ords below the
    build's size are never handed out again (append_chunk_meta min_ord). Returns the chunks removed. A community that loses an exemplar this way keeps its draft summary
    until the next full build, which re-derives both."""
    if not doc_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT ord FROM lex_chunk_meta WHERE label = %s AND doc_id = ANY(%s)", (label, list(doc_ids)))
        ords = [r[0] for r in cur.fetchall()]
        if not ords:
            return 0
        for kind in _KIND_PREFIX:
            cur.execute("SELECT to_regclass(%s)", (_table(label, kind),))
            if cur.fetchone()[0] is not None:
                cur.execute(f"DELETE FROM {_table(label, kind)} WHERE ord = ANY(%s)", (ords,))
        if build_id is not None:
            for t in ("lex_assign", "lex_exemplar"):
                cur.execute(f"DELETE FROM {t} WHERE build_id = %s AND ord = ANY(%s)", (build_id, ords))
        cur.execute("DELETE FROM lex_chunk_meta WHERE label = %s AND ord = ANY(%s)", (label, ords))
    return len(ords)


_VERSIONED_ID = r"^arxiv/([0-9]{4}_[0-9]{4,5}|[a-z-]+(\.[A-Z]{2})?_[0-9]{7})v[0-9]+(_methods)?$"


def normalise_versioned_doc_ids(conn, label: str) -> int:
    """V9. One-time repair: a paper stored with its version inside the doc_id ('arxiv/2403_19889v1') gets the
    id without it and the version in `version`, in the metadata and in the vector tables' doc_id column, so
    the identity rule (domain_corpora C11) sees it. Refuses, changing nothing, when the stripped id is
    already held by another stored paper: that would merge two papers. Returns chunks changed."""
    with conn.cursor() as cur:
        cur.execute(f"""SELECT DISTINCT doc_id, regexp_replace(doc_id, 'v[0-9]+(_methods)?$', '\\1') FROM lex_chunk_meta
                        WHERE label = %s AND doc_id ~ %s""", (label, _VERSIONED_ID))
        pairs = cur.fetchall()
        if not pairs:
            return 0
        cur.execute("SELECT DISTINCT doc_id FROM lex_chunk_meta WHERE label = %s", (label,))
        held = {r[0] for r in cur.fetchall()}
        clash = [(a, b) for a, b in pairs if b in held]
        assert not clash, "stripping the version would merge stored papers: %s" % clash
        n = 0
        for old, new in pairs:
            cur.execute("""UPDATE lex_chunk_meta SET version = substring(doc_id from 'v([0-9]+)(?:_methods)?$')::int, doc_id = %s
                           WHERE label = %s AND doc_id = %s""", (new, label, old))
            n += cur.rowcount
            for kind in ("chunk", "cos"):
                cur.execute("SELECT to_regclass(%s)", (_table(label, kind),))
                if cur.fetchone()[0] is not None:
                    cur.execute(f"UPDATE {_table(label, kind)} SET doc_id = %s WHERE doc_id = %s", (new, old))
    return n


def ensure_view_schema(conn, label: str, sparse_dim: int, dense_dim: int) -> None:
    """V7. The doc-doc cosine view and the dense view, one typed table each. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {_table(label, 'cos')} (
                            ord    integer PRIMARY KEY,
                            doc_id text NOT NULL,
                            source text,
                            nnz    integer NOT NULL,
                            truncated boolean NOT NULL DEFAULT false,
                            vec    sparsevec({int(sparse_dim)}) NOT NULL)""")
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {_table(label, 'dense')} (
                            ord integer PRIMARY KEY,
                            emb vector({int(dense_dim)}) NOT NULL)""")


def write_dense(conn, label: str, ords, emb, replace: bool = True) -> int:
    """V7. emb: (n, dim) rows, already mean-centered and L2-normalised (R9). Returns rows written."""
    t = _table(label, "dense")
    with conn.cursor() as cur:
        if replace:
            cur.execute(f"TRUNCATE {t}")
        with cur.copy(f"COPY {t} (ord, emb) FROM STDIN") as cp:
            for o, row in zip(ords, emb):
                cp.write_row((int(o), "[" + ",".join("%.6g" % float(x) for x in row) + "]"))
    return len(ords)


def ensure_build_schema(conn) -> None:
    """V7. The build record and what hangs off it. Idempotent."""
    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_build (
                           build_id  bigserial PRIMARY KEY,
                           label     text NOT NULL,
                           created   timestamptz NOT NULL DEFAULT now(),
                           live      boolean NOT NULL DEFAULT true,
                           n_chunks  integer NOT NULL,
                           params    jsonb NOT NULL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_community (
                           build_id bigint  NOT NULL REFERENCES lex_build ON DELETE CASCADE,
                           cid      integer NOT NULL,
                           size     integer NOT NULL,
                           centroid vector,
                           PRIMARY KEY (build_id, cid))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_assign (
                           build_id bigint  NOT NULL REFERENCES lex_build ON DELETE CASCADE,
                           ord      integer NOT NULL,
                           cid      integer NOT NULL,
                           how      text    NOT NULL CHECK (how IN ('consensus', 'nearest-centroid')),
                           x        real,
                           y        real,
                           PRIMARY KEY (build_id, ord))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_exemplar (
                           build_id bigint  NOT NULL REFERENCES lex_build ON DELETE CASCADE,
                           cid      integer NOT NULL,
                           rank     integer NOT NULL,
                           role     text    NOT NULL,
                           z        real    NOT NULL,
                           ord      integer NOT NULL,
                           PRIMARY KEY (build_id, cid, rank))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_summary (
                           build_id bigint  NOT NULL REFERENCES lex_build ON DELETE CASCADE,
                           cid      integer NOT NULL,
                           status   text    NOT NULL CHECK (status IN ('draft', 'failed', 'not_summarized')),
                           title    text,
                           summary  text,
                           model    text,
                           provider text,
                           prompt_tokens integer,
                           completion_tokens integer,
                           cost     real,
                           chunk_keys jsonb NOT NULL,
                           reason   text,
                           PRIMARY KEY (build_id, cid))""")


def new_build(conn, label: str, n_chunks: int, params: dict) -> int:
    """V7. Supersede the label's live build (never delete it) and open the next. Returns build_id."""
    import json
    with conn.cursor() as cur:
        cur.execute("UPDATE lex_build SET live = false WHERE label = %s AND live", (label,))
        cur.execute("INSERT INTO lex_build (label, n_chunks, params) VALUES (%s, %s, %s) RETURNING build_id",
                    (label, int(n_chunks), json.dumps(params)))
        return cur.fetchone()[0]


def live_build(conn, label: str) -> tuple[int, int, dict] | None:
    """V7. (build_id, n_chunks, params) of the label's live build, or None before the first."""
    with conn.cursor() as cur:
        cur.execute("SELECT build_id, n_chunks, params FROM lex_build WHERE label = %s AND live "
                    "ORDER BY build_id DESC LIMIT 1", (label,))
        r = cur.fetchone()
    return None if r is None else (r[0], r[1], r[2])


def update_build_params(conn, build_id: int, params: dict) -> dict:
    """V7. Merge `params` into the build's params (top-level keys replaced) so each stage adds what
    it fixed -- the ingest its BM25 statistics, the map its dense mean. Returns the merged params."""
    import json
    with conn.cursor() as cur:
        cur.execute("UPDATE lex_build SET params = params || %s::jsonb WHERE build_id = %s RETURNING params",
                    (json.dumps(params), build_id))
        r = cur.fetchone()
    assert r is not None, "no build %s" % build_id
    return r[0]


def clear_derived(conn, build_id: int) -> None:
    """V7. Empty a build's communities, assignments and exemplars (not its summaries) so the stage
    that writes them can run again on the same build."""
    with conn.cursor() as cur:
        for t in ("lex_exemplar", "lex_assign", "lex_community"):
            cur.execute(f"DELETE FROM {t} WHERE build_id = %s", (build_id,))


def write_communities(conn, build_id: int, rows: list[tuple[int, int, list[float] | None]]) -> None:
    """V7. rows = (cid, size, centroid or None)."""
    with conn.cursor() as cur, cur.copy("COPY lex_community (build_id, cid, size, centroid) FROM STDIN") as cp:
        for cid, size, cen in rows:
            cp.write_row((build_id, int(cid), int(size),
                          None if cen is None else "[" + ",".join("%.6g" % float(x) for x in cen) + "]"))


def write_assignments(conn, build_id: int, rows: list[tuple[int, int, str, float | None, float | None]]) -> None:
    """V7. rows = (ord, cid, how, x, y); how is 'consensus' or 'nearest-centroid'."""
    with conn.cursor() as cur, cur.copy("COPY lex_assign (build_id, ord, cid, how, x, y) FROM STDIN") as cp:
        for o, cid, how, x, y in rows:
            cp.write_row((build_id, int(o), int(cid), how, x, y))


def write_exemplars(conn, build_id: int, rows: list[tuple[int, int, str, float, int]]) -> None:
    """V7. rows = (cid, rank, role, z, ord)."""
    with conn.cursor() as cur, cur.copy("COPY lex_exemplar (build_id, cid, rank, role, z, ord) FROM STDIN") as cp:
        for cid, rank, role, z, o in rows:
            cp.write_row((build_id, int(cid), int(rank), role, float(z), int(o)))


def write_summaries(conn, build_id: int, recs: list[dict]) -> int:
    """V7. Replace the build's summary rows with `recs` ({community, status, title, summary, model,
    provider, prompt_tokens, completion_tokens, cost, chunk_keys, reason}). Returns rows written."""
    import json
    with conn.cursor() as cur:
        cur.execute("DELETE FROM lex_summary WHERE build_id = %s", (build_id,))
        with cur.copy("COPY lex_summary (build_id, cid, status, title, summary, model, provider, "
                      "prompt_tokens, completion_tokens, cost, chunk_keys, reason) FROM STDIN") as cp:
            for r in recs:
                cp.write_row((build_id, int(r["community"]), r["status"], r.get("title"), r.get("summary"),
                              r.get("model"), r.get("provider"), r.get("prompt_tokens"),
                              r.get("completion_tokens"), r.get("cost"),
                              json.dumps(r["chunk_keys"]), r.get("reason")))
    return len(recs)


def query(conn, label: str, dim: int, qcols: list[int], k: int,
          exact: bool = False, ef_search: int | None = None,
          weights: list[float] | None = None) -> list[tuple[int, float]]:
    """V4: top-k (ord, inner_product). Query vector over 0-based qcols: 1.0 each, or
    `weights[i]` for qcols[i] (V6: idf * qtf when the stored vector holds saturated tf)."""
    if not qcols:
        return []
    t = _table(label)
    ws = [1.0] * len(qcols) if weights is None else weights
    assert len(ws) == len(qcols), "weights must align with qcols"
    q = _literal(dict(zip(qcols, ws)), dim)
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
