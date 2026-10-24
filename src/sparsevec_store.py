"""Store legacy BM25 artifacts and exact sparsevec bridge generations.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| LEGACY | Retain typed chunk/cosine tables and ANN helpers for diagnostic callers. | [opt] | R1 |
| GENERATION | Store build-scoped metadata/vocabulary and record_id-only sparse vector bridges. | [mandatory] | R2 |
| PUBLISH | Validate complete record and vocabulary coverage before atomic live switch. | [mandatory] | R3 |
| QUERY | Exact inner-product scan joined to metadata for the pinned build. | [contract] | R4 |

R1 Legacy helpers remain isolated for older diagnostics and are not called by active section BM25.
R2 Active vectors contain only `(record_id, vec)`; build membership and metadata remain in their owning tables.
R3 Incomplete generations remain non-live; failed publication leaves the prior complete generation live.
R4 Empty encodings are stored as zero sparsevec rows and exact queries never set ANN parameters.
R5 Vectors over SPARSEVEC_MAX_NNZ retain the heaviest weights; diagnostics record truncation.
R6 Query weights carry IDF and query term frequency; stored vectors carry saturated BM25 term frequency.

CLOSED: HNSW and IVFFlat on the active retrieval lane; direct exact vector operations are selected.

Preconditions: PostgreSQL has pgvector sparsevec support. Failure modes: record ID collision/orphan, schema mismatch, invalid dimension, incomplete generation, and database errors abort the operation.

V1  Legacy typed tables retain their vocabulary-specific dimensions. Active exact vectors use
    untyped sparsevec because exact scans do not require a typed ANN index.

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

V6  WHERE a corpus is ingested for serving (src/ingest_arxiv_sparsevec.py), IDF
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
V10 An entity inventory is frozen PER BUILD in `lex_entity` and the chunks it matches in `lex_mention`
    (build_id, ord, ent_id, cnt); entity_derive.py derives it (AE1-AE14), nothing here does. A build
    with no `lex_entity` rows has no entities and the service's entities step does nothing for it.
    Deleting a paper (V9) deletes its mentions (entity_derive AE12). Spec: NO GOVERNING SPEC beyond the
    approved plan derive-arxiv-entities-with-information-theory.md; task T131.

DDL lives HERE via ensure_schema, not in sql/, per entities.py:129-135.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from itertools import zip_longest

import psycopg

import config

DSN = config.DSN
SPARSEVEC_MAX_NNZ = 1000        # pgvector 0.8.1 hard limit (V2)


_KIND_PREFIX = {"chunk": "lex_chunk_", "cos": "lex_cos_", "dense": "lex_dense_", "sparse": "lex_sparse_"}


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
        for kind in ("chunk", "cos", "dense"):
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
                           build_id      bigint  NOT NULL DEFAULT 0,
                           record_id     bigserial NOT NULL UNIQUE,
                           doc_id        text    NOT NULL,
                           section_idx   integer NOT NULL,
                           chunk_idx     integer NOT NULL,
                           section_title text    NOT NULL,
                           is_reference  boolean NOT NULL,
                           nnz           integer NOT NULL DEFAULT 0,
                           truncated     boolean NOT NULL DEFAULT false,
                           text          text    NOT NULL,
                           PRIMARY KEY (label, build_id, ord),
                           UNIQUE (label, build_id, doc_id, section_idx, chunk_idx))""")
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS build_id bigint NOT NULL DEFAULT 0")
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS record_id bigserial")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS lex_chunk_meta_record_id_key ON lex_chunk_meta(record_id)")
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS nnz integer NOT NULL DEFAULT 0")
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS truncated boolean NOT NULL DEFAULT false")
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


def ensure_generation_schema(conn, label: str) -> str:
    """Create/migrate the record_id-only sparse vector bridge and generation metadata.

    Require: label identifies an active sparse retrieval lane.
    Guarantee: vectors contain only record_id and vec; metadata owns build_id and ord.
    Maintain: old tables stay intact until row-for-row migration validation succeeds.
    Assert: each vector row resolves to exactly one metadata row.
    """
    import json
    from psycopg import sql

    ensure_build_schema(conn)
    ensure_chunk_meta(conn)
    vector_table = _table(label, "sparse")
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS build_id bigint NOT NULL DEFAULT 0")
        cur.execute("ALTER TABLE lex_chunk_meta ADD COLUMN IF NOT EXISTS record_id bigserial")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS lex_chunk_meta_record_id_key ON lex_chunk_meta(record_id)")
        constraints = cur.execute(
            "SELECT c.conname, c.contype, array_agg(a.attname ORDER BY k.ord) AS cols "
            "FROM pg_constraint c CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY k(attnum,ord) "
            "JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.attnum "
            "WHERE c.conrelid='lex_chunk_meta'::regclass AND c.contype IN ('p','u') GROUP BY c.conname,c.contype"
        ).fetchall()
        for name, kind, cols in constraints:
            if kind == "u" and list(cols) == ["record_id"]:
                continue
            wanted = ["label", "build_id", "ord"] if kind == "p" else ["label", "build_id", "doc_id", "section_idx", "chunk_idx"]
            if list(cols) != wanted:
                cur.execute(sql.SQL("ALTER TABLE lex_chunk_meta DROP CONSTRAINT {}").format(sql.Identifier(name)))
        cur.execute("CREATE TABLE IF NOT EXISTS lex_build_vocab (build_id bigint NOT NULL REFERENCES lex_build ON DELETE CASCADE, "
                    "col integer NOT NULL, piece text NOT NULL, idf real NOT NULL, PRIMARY KEY(build_id,col), UNIQUE(build_id,piece))")
        cur.execute("CREATE INDEX IF NOT EXISTS lex_chunk_meta_build_lookup_idx ON lex_chunk_meta (label,build_id,doc_id,section_idx)")
        if not any(kind == "p" and list(cols) == ["label", "build_id", "ord"] for _, kind, cols in constraints):
            cur.execute("ALTER TABLE lex_chunk_meta ADD CONSTRAINT lex_chunk_meta_build_pkey PRIMARY KEY (label,build_id,ord)")
        if not any(kind == "u" and list(cols) == ["label", "build_id", "doc_id", "section_idx", "chunk_idx"] for _, kind, cols in constraints):
            cur.execute("ALTER TABLE lex_chunk_meta ADD CONSTRAINT lex_chunk_meta_build_key UNIQUE (label,build_id,doc_id,section_idx,chunk_idx)")

        vector_exists = cur.execute("SELECT to_regclass(%s)", (vector_table,)).fetchone()[0] is not None
        if not vector_exists:
            cur.execute(f"CREATE TABLE {vector_table} (record_id bigint PRIMARY KEY REFERENCES lex_chunk_meta(record_id) ON DELETE CASCADE, vec sparsevec NOT NULL)")
        vector_cols = {r[0] for r in cur.execute(
            "SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped", (vector_table,)
        ).fetchall()}
        if "build_id" in vector_cols:
            backup = vector_table + "_build_scoped"
            if cur.execute("SELECT to_regclass(%s)", (backup,)).fetchone()[0] is not None:
                raise RuntimeError(f"{backup} already exists; inspect prior sparse-vector migration before retrying")
            stage = vector_table + "_record_id_stage"
            cur.execute(sql.SQL("CREATE TABLE {} (record_id bigint PRIMARY KEY REFERENCES lex_chunk_meta(record_id) ON DELETE CASCADE, vec sparsevec NOT NULL)").format(sql.Identifier(stage)))
            old_n = cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(vector_table))).fetchone()[0]
            cur.execute(sql.SQL("""INSERT INTO {} (record_id,vec)
                SELECT m.record_id,v.vec FROM {} v JOIN lex_chunk_meta m
                  ON m.label=%s AND m.build_id=v.build_id AND m.ord=v.record_id""").format(sql.Identifier(stage), sql.Identifier(vector_table)), (label,))
            new_n = cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(stage))).fetchone()[0]
            if old_n != new_n:
                raise RuntimeError(f"sparse vector migration found {old_n-new_n} orphan rows")
            cur.execute(sql.SQL("ALTER TABLE {} RENAME TO {}").format(sql.Identifier(vector_table), sql.Identifier(backup)))
            cur.execute(sql.SQL("ALTER TABLE {} RENAME TO {}").format(sql.Identifier(stage), sql.Identifier(vector_table)))

    live = live_build(conn, label)
    if live is None:
        return vector_table
    build_id, _, params = live
    dim = int(params["bm25"]["dim"])
    legacy = _table(label, "chunk")
    legacy_exists = conn.execute("SELECT to_regclass(%s)", (legacy,)).fetchone()[0] is not None
    if legacy_exists:
        old_count = conn.execute(f"SELECT count(*) FROM {legacy}").fetchone()[0]
        new_count = conn.execute(f"SELECT count(*) FROM {vector_table} v JOIN lex_chunk_meta m ON m.record_id=v.record_id "
                                 "WHERE m.label=%s AND m.build_id=%s", (label, build_id)).fetchone()[0]
        if new_count == 0:
            conn.execute("UPDATE lex_chunk_meta SET build_id=%s WHERE label=%s AND build_id=0", (build_id, label))
            conn.execute(f"INSERT INTO {vector_table} (record_id,vec) SELECT m.record_id,v.vec::sparsevec FROM {legacy} v "
                         "JOIN lex_chunk_meta m ON m.label=%s AND m.build_id=%s AND m.ord=v.ord ON CONFLICT DO NOTHING", (label, build_id))
            conn.execute(f"INSERT INTO {vector_table} (record_id,vec) "
                         "SELECT m.record_id,('{}/' || %s::text)::sparsevec FROM lex_chunk_meta m "
                         "WHERE m.label=%s AND m.build_id=%s AND NOT EXISTS "
                         f"(SELECT 1 FROM {vector_table} v WHERE v.record_id=m.record_id) ON CONFLICT DO NOTHING",
                         (dim, label, build_id))
            conn.execute(f"UPDATE lex_chunk_meta m SET nnz=v.nnz,truncated=v.truncated FROM {legacy} v "
                         "WHERE m.label=%s AND m.build_id=%s AND m.ord=v.ord", (label, build_id))
            conn.execute("INSERT INTO lex_build_vocab (build_id,col,piece,idf) "
                         "SELECT %s,col,piece,coalesce(idf,0) FROM lex_vocab WHERE label=%s ON CONFLICT DO NOTHING",
                         (build_id, label))
            new_count = conn.execute(f"SELECT count(*) FROM {vector_table} v JOIN lex_chunk_meta m ON m.record_id=v.record_id "
                                     "WHERE m.label=%s AND m.build_id=%s", (label, build_id)).fetchone()[0]
            if old_count > new_count:
                raise RuntimeError("legacy vector backfill did not account for every source vector")
        return vector_table
    return vector_table


def write_generation_meta(conn, label: str, build_id: int, records: list[dict]) -> int:
    """Persist source chunk metadata for an unpublished generation.

    Require:
        `build_id` names a non-live generation and `records` have unique paper/section/chunk keys.
    Guarantee:
        All source records are written with ordinals starting at zero for this build.
    Maintain:
        Repeated ordinals in other generations remain valid and untouched.
    Assert:
        The caller supplies text and every required business-key field.

    Args:
        conn: Open BM25 PostgreSQL connection.
        label: Retrieval lane label.
        build_id: Unpublished generation ID.
        records: Chunk text, flags, and section identity records.

    Returns:
        Number of metadata records written.
    """
    cols = "label,ord,build_id,doc_id,version,section_idx,chunk_idx,section_title,is_reference,is_junk,nnz,truncated,text"
    with conn.cursor() as cur, cur.copy(f"COPY lex_chunk_meta ({cols}) FROM STDIN") as cp:
        for ord_, r in enumerate(records):
            cp.write_row((label, ord_, int(build_id), r["doc_id"], int(r.get("version", 1)), r["section_idx"],
                          r["chunk_idx"], r["section_title"], r["is_reference"], bool(r.get("is_junk", False)),
                          0, False, r["text"]))
    return len(records)


def write_generation_vocab(conn, build_id: int, pieces: list[str], idf) -> int:
    """Persist a build-scoped BPE vocabulary and its IDF values."""
    with conn.cursor() as cur, cur.copy("COPY lex_build_vocab (build_id,col,piece,idf) FROM STDIN") as cp:
        for col, piece in enumerate(pieces, 1):
            cp.write_row((int(build_id), col, str(piece), float(idf[col - 1])))
    return len(pieces)


def write_generation_vectors(conn, label: str, build_id: int, dim: int, matrices) -> dict:
    """Write narrow sparse vectors and matching nnz diagnostics in one transaction.

    Require:
        Metadata for the generation exists and each input mapping uses zero-based vocabulary columns.
    Guarantee:
        Every input row, including empty mappings, becomes one vector record joined by globally unique metadata record_id.
    Maintain:
        A failed COPY or diagnostic update rolls back the vector write as a unit.
    Assert:
        Input row order matches metadata ordinal order and sparsevec dimensions match `dim`.

    Args:
        conn: Open BM25 PostgreSQL connection.
        label: Retrieval lane label.
        build_id: Unpublished generation ID.
        dim: Vocabulary dimension.
        matrices: Ordered iterable of sparse zero-based column-to-weight mappings.

    Returns:
        Mapping with written row count, truncated count, and target table name.
    """
    table = _table(label, "sparse")
    written = truncated = 0
    diagnostics = []
    with conn.transaction():
        conn.execute("DROP TABLE IF EXISTS pg_temp.lex_vector_diag")
        conn.execute("CREATE TEMP TABLE lex_vector_diag (ord integer PRIMARY KEY, nnz integer NOT NULL, truncated boolean NOT NULL) ON COMMIT DROP")
        metadata_ids = conn.execute("SELECT ord,record_id FROM lex_chunk_meta WHERE label=%s AND build_id=%s ORDER BY ord",
                                    (label, int(build_id))).fetchall()
        if [row[0] for row in metadata_ids] != list(range(len(metadata_ids))):
            raise RuntimeError("sparse vector inputs do not match contiguous metadata ordinals")
        missing = object()
        with conn.cursor() as cur, cur.copy(f"COPY {table} (record_id,vec) FROM STDIN") as cp:
            for (meta, entries) in zip_longest(metadata_ids, matrices, fillvalue=missing):
                if meta is missing:
                    raise RuntimeError("sparse vector input exceeds metadata rows")
                if entries is missing:
                    break
                ord_, global_id = meta
                items = list(entries.items())
                cut = len(items) > SPARSEVEC_MAX_NNZ
                if cut:
                    items.sort(key=lambda kv: (-kv[1], kv[0]))
                    items = items[:SPARSEVEC_MAX_NNZ]
                    truncated += 1
                cp.write_row((int(global_id), _literal(items, dim)))
                diagnostics.append((int(ord_), len(items), cut))
                written += 1
        with conn.cursor() as cur, cur.copy("COPY lex_vector_diag (ord,nnz,truncated) FROM STDIN") as diag:
            for row in diagnostics:
                diag.write_row(row)
        conn.execute(
            "UPDATE lex_chunk_meta m SET nnz=d.nnz,truncated=d.truncated "
            "FROM lex_vector_diag d WHERE m.label=%s AND m.build_id=%s AND m.ord=d.ord",
            (label, int(build_id)),
        )
    return {"written": written, "truncated": truncated, "table": table}


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


def known_text_md5(conn, label: str, texts: list[str], exclude_doc_ids: list[str] | None = None) -> set[str]:
    """V8. Find matching stored text hashes without counting rows being replaced.

    Args:
        conn: Open autocommit Postgres connection.
        label: Sparsevec label whose chunk metadata is checked.
        texts: Candidate text values.
        exclude_doc_ids: Document IDs excluded during replacement deduplication.

    Returns:
        MD5 hashes present on other stored chunks.

    Failure modes:
        Database errors propagate to the caller before any mutation.
    """
    import hashlib
    want = {hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts}
    if not want:
        return set()
    with conn.cursor() as cur:
        if exclude_doc_ids:
            cur.execute("SELECT text_md5 FROM lex_chunk_meta WHERE label = %s AND text_md5 = ANY(%s) AND NOT (doc_id = ANY(%s))",
                        (label, list(want), exclude_doc_ids))
        else:
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
        query = "SELECT build_id,ord,record_id FROM lex_chunk_meta WHERE label=%s AND doc_id=ANY(%s)"
        args = [label, list(doc_ids)]
        if build_id is not None:
            query += " AND build_id=%s"
            args.append(int(build_id))
        selected = cur.execute(query, args).fetchall()
        if not selected and build_id is not None:
            selected = cur.execute("SELECT build_id,ord,record_id FROM lex_chunk_meta WHERE label=%s AND doc_id=ANY(%s)",
                                   (label, list(doc_ids))).fetchall()
        if not selected:
            return 0
        ords = list(dict.fromkeys(r[1] for r in selected))
        record_ids = [r[2] for r in selected]
        builds = sorted({r[0] for r in selected} | ({int(build_id)} if build_id is not None else set()))
        for kind in ("chunk", "cos", "dense"):
            cur.execute("SELECT to_regclass(%s)", (_table(label, kind),))
            if cur.fetchone()[0] is not None:
                cur.execute(f"DELETE FROM {_table(label, kind)} WHERE ord = ANY(%s)", (ords,))
        sparse_table = _table(label, "sparse")
        cur.execute("SELECT to_regclass(%s)", (sparse_table,))
        if cur.fetchone()[0] is not None:
            vector_cols = {r[0] for r in cur.execute(
                "SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped", (sparse_table,)
            ).fetchall()}
            if vector_cols == {"record_id", "vec"}:
                cur.execute(f"DELETE FROM {sparse_table} WHERE record_id=ANY(%s)", (record_ids,))
        for affected_build in builds:
            build_ords = [r[1] for r in selected if r[0] == affected_build]
            if build_id is not None and affected_build == int(build_id) and not build_ords:
                build_ords = ords
            for t in ("lex_assign", "lex_exemplar", "lex_mention"):                       # V10: mentions go with the chunk
                cur.execute("SELECT to_regclass(%s)", (t,))
                if cur.fetchone()[0] is not None:
                    cur.execute(f"DELETE FROM {t} WHERE build_id = %s AND ord = ANY(%s)", (affected_build, build_ords))
        cur.execute("DELETE FROM lex_chunk_meta WHERE label=%s AND record_id=ANY(%s)", (label, record_ids))
    return len(selected)


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
                           ready     boolean NOT NULL DEFAULT true,
                           n_chunks  integer NOT NULL,
                           params    jsonb NOT NULL)""")
        cur.execute("ALTER TABLE lex_build ADD COLUMN IF NOT EXISTS ready boolean NOT NULL DEFAULT true")
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
        cur.execute("SELECT build_id, n_chunks, params FROM lex_build WHERE label = %s AND live AND ready "
                    "ORDER BY build_id DESC LIMIT 1", (label,))
        r = cur.fetchone()
    return None if r is None else (r[0], r[1], r[2])


def open_generation(conn, label: str, n_chunks: int, params: dict) -> int:
    """Create an unpublished build while leaving the previous ready generation live.

    Require: build schema exists and params records its source graph build.
    Guarantee: a unique build ID with `ready=false` and `live=false`.
    Maintain: opening a failed generation never changes the serving pointer.
    """
    import json
    with conn.transaction():
        row = conn.execute(
            "INSERT INTO lex_build (label, live, ready, n_chunks, params) VALUES (%s, false, false, %s, %s) RETURNING build_id",
            (label, int(n_chunks), json.dumps(params)),
        ).fetchone()
    return int(row[0])


def publish_generation(conn, label: str, build_id: int, expected_rows: int, dim: int) -> None:
    """Validate vector-only storage and atomically make a complete BM25 build live.

    Require: metadata, one sparse vector per record, and one vocabulary row per dimension are persisted.
    Guarantee: this complete generation becomes live in one transaction.
    Maintain: a failed validation leaves the prior ready generation live.
    """
    sparse_table = _table(label, "sparse")
    with conn.transaction():
        build = conn.execute(
            "SELECT n_chunks, ready, params FROM lex_build WHERE build_id=%s AND label=%s FOR UPDATE",
            (int(build_id), label),
        ).fetchone()
        if build is None or build[1]:
            raise RuntimeError("BM25 generation is absent or already published")
        bm25_params = build[2].get("bm25", {})
        meta_n = conn.execute(
            "SELECT count(*) FROM lex_chunk_meta WHERE label=%s AND build_id=%s",
            (label, int(build_id)),
        ).fetchone()[0]
        vector_n = conn.execute(
            f"SELECT count(*) FROM {sparse_table} v JOIN lex_chunk_meta m ON m.record_id=v.record_id "
            "WHERE m.label=%s AND m.build_id=%s", (label, int(build_id))
        ).fetchone()[0]
        orphan_n = conn.execute(
            f"SELECT count(*) FROM {sparse_table} v LEFT JOIN lex_chunk_meta m ON m.record_id=v.record_id "
            "WHERE m.record_id IS NULL"
        ).fetchone()[0]
        vocab_n = conn.execute(
            "SELECT count(*) FROM lex_build_vocab WHERE build_id=%s", (int(build_id),)
        ).fetchone()[0]
        vocab_bounds = conn.execute(
            "SELECT min(col), max(col) FROM lex_build_vocab WHERE build_id=%s", (int(build_id),)
        ).fetchone()
        if (int(build[0]) != expected_rows or meta_n != expected_rows or vector_n != expected_rows or orphan_n
                or vocab_n != dim or vocab_bounds != (1, dim)):
            raise RuntimeError(
                "incomplete BM25 generation: expected rows=%d vocab=%d, found metadata=%d vectors=%d vocab=%d"
                % (expected_rows, dim, meta_n, vector_n, vocab_n)
            )
        sample_view = bm25_params.get("sample_view")
        sample_expected = bm25_params.get("df_sample_size")
        if sample_view and sample_expected is not None:
            expected_view = "lex_sample_" + "".join(ch if ch.isalnum() else "_" for ch in label.lower())
            if sample_view != expected_view:
                raise RuntimeError("BM25 sample view does not belong to this label")
            sample_n = conn.execute(
                f"SELECT count(*) FROM {expected_view} WHERE build_id=%s", (int(build_id),)
            ).fetchone()[0]
            if sample_n != int(sample_expected):
                raise RuntimeError("BM25 sample row count does not match build provenance")
        anns = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname=current_schema() AND tablename=%s "
            "AND (indexdef ILIKE '%%USING hnsw%%' OR indexdef ILIKE '%%USING ivfflat%%')",
            (sparse_table,),
        ).fetchall()
        if anns:
            raise RuntimeError("ANN index found on exact sparse-vector table")
        conn.execute("UPDATE lex_build SET live=false WHERE label=%s AND live", (label,))
        conn.execute("UPDATE lex_build SET live=true, ready=true WHERE build_id=%s", (int(build_id),))


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


def ensure_entity_schema(conn) -> None:
    """V10. The frozen inventory and its mentions. Idempotent."""
    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_entity (
                           build_id    bigint   NOT NULL REFERENCES lex_build ON DELETE CASCADE,
                           ent_id      integer  NOT NULL,
                           fold        text     NOT NULL,
                           surface     text     NOT NULL,
                           n           smallint NOT NULL,
                           score       real     NOT NULL,
                           df_papers   integer  NOT NULL,
                           df_chunks   integer  NOT NULL,
                           tf          integer  NOT NULL,
                           canonical   integer  NOT NULL,
                           PRIMARY KEY (build_id, ent_id))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS lex_mention (
                           build_id bigint  NOT NULL REFERENCES lex_build ON DELETE CASCADE,
                           ord      integer NOT NULL,
                           ent_id   integer NOT NULL,
                           cnt      integer NOT NULL,
                           PRIMARY KEY (build_id, ord, ent_id))""")
        cur.execute("CREATE INDEX IF NOT EXISTS lex_mention_ent ON lex_mention (build_id, ent_id)")


def write_entities(conn, build_id: int, rows: list[tuple]) -> None:
    """V10. Replace the build's inventory. rows = (ent_id, fold, surface, n, score, df_papers, df_chunks, tf, canonical).
    Its mentions are cleared with it: a mention names an entity of THIS inventory only."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM lex_mention WHERE build_id = %s", (build_id,))
        cur.execute("DELETE FROM lex_entity WHERE build_id = %s", (build_id,))
        with cur.copy("COPY lex_entity (build_id, ent_id, fold, surface, n, score, df_papers, df_chunks, tf, canonical) FROM STDIN") as cp:
            for e, fold, surf, n, sc, dp, dc, tf, can in rows:
                cp.write_row((build_id, int(e), fold, surf, int(n), float(sc), int(dp), int(dc), int(tf), int(can)))


def read_entities(conn, build_id: int) -> list[tuple]:
    """V10. Guarantee: the build's inventory rows in ent_id order (same shape write_entities takes), [] when none was frozen."""
    with conn.cursor() as cur:
        cur.execute("SELECT ent_id, fold, surface, n, score, df_papers, df_chunks, tf, canonical FROM lex_entity "
                    "WHERE build_id = %s ORDER BY ent_id", (build_id,))
        return cur.fetchall()


def replace_mentions(conn, build_id: int, ords: Iterable[int], rows: list[tuple[int, int, int]]) -> int:
    """V10. For the chunks `ords`, drop their mentions and write `rows` = (ord, ent_id, cnt). Returns rows written. A chunk
    re-matched is never counted twice."""
    ords = [int(o) for o in ords]
    with conn.cursor() as cur:
        if ords:
            cur.execute("DELETE FROM lex_mention WHERE build_id = %s AND ord = ANY(%s)", (build_id, ords))
        with cur.copy("COPY lex_mention (build_id, ord, ent_id, cnt) FROM STDIN") as cp:
            for o, e, c in rows:
                cp.write_row((build_id, int(o), int(e), int(c)))
    return len(rows)


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


def query_generation(conn, label: str, build_id: int, dim: int, qcols: list[int], k: int,
                     weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Return exact top-k `(ord, inner_product)` hits from one BM25 generation.

    Require:
        `build_id` names a complete generation and `qcols` are zero-based vocabulary columns.
    Guarantee:
        The query joins vectors to build-scoped metadata and breaks score ties by global record ID.
    Maintain:
        No ANN session setting or metadata-field predicate affects the vector scan.
    Assert:
        Query weights align with query columns and each column fits the stored dimension.
    """
    if not qcols or k <= 0:
        return []
    ws = [1.0] * len(qcols) if weights is None else weights
    assert len(ws) == len(qcols), "weights must align with qcols"
    assert all(0 <= int(col) < int(dim) for col in qcols), "query column exceeds generation dimension"
    q = _literal(dict(zip(qcols, ws)), dim)
    table = _table(label, "sparse")
    rows = conn.execute(
        f"SELECT m.ord, -(v.vec <#> %s::sparsevec) AS ip FROM {table} v "
        "JOIN lex_chunk_meta m ON m.record_id=v.record_id "
        "WHERE m.label=%s AND m.build_id=%s ORDER BY v.vec <#> %s::sparsevec, v.record_id LIMIT %s",
        (q, label, int(build_id), q, int(k)),
    ).fetchall()
    return [(int(ord_), float(score)) for ord_, score in rows]


def counts(conn, label: str) -> dict:
    t = _table(label)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*), sum(truncated::int), max(nnz), "
                    f"pg_total_relation_size('{t}') FROM {t}")
        n, tr, mx, size = cur.fetchone()
    return {"rows": n, "truncated": tr or 0, "max_nnz": mx, "bytes": size}
