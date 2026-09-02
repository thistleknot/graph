"""Persist a fitted ChunkGraph into Postgres (jsonb + pgvector).

Require: a ChunkGraph on which fit() has already run.
Guarantee: one graph_run row plus its nodes/edges/communities/embeddings,
committed atomically; any prior live run under the same label is superseded,
never deleted. Node payloads carry `source` and edge rows carry the source
pair when the run was fit with labelled sources; absent otherwise
(design.md §6.14 R20).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

DSN = os.environ.get("CHUNKGRAPH_DSN",
                     "postgresql://graph:graph@localhost:5433/graph")

# Params worth reproducing a run from; mirrors ChunkGraph.__init__.
_PARAM_ATTRS = ("k_sigma", "M", "H", "K", "LAM", "EPS", "phrases", "chunk_params")   # R17


def _hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _vec_literal(row: np.ndarray) -> str:
    # pgvector's text input format; COPY parses it straight into vector.
    return "[" + ",".join(f"{float(x):.7g}" for x in row) + "]"


def _params(cg) -> dict:
    p = {a: getattr(cg, a) for a in _PARAM_ATTRS if hasattr(cg, a)}
    p["vocab_restricted"] = cg.vocab is not None
    p["has_dense"] = getattr(cg, "sim_dense", None) is not None
    p["avgdl"] = float(getattr(cg, "avgdl", 0.0))
    return p


def save(cg, label: str, dsn: str = DSN, min_size: int = 5) -> str:
    """Returns the new run_id. Idempotent per label: re-running supersedes."""
    if not hasattr(cg, "chunks"):
        raise ValueError("ChunkGraph has not been fit()")

    E = getattr(cg, "E", None)                 # absent when embed_fn is None
    embed_dim = int(E.shape[1]) if E is not None else None
    edges = cg.edges()
    comms = cg.communities(min_size=min_size)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Retire the incumbent FIRST: graph_run_live_label allows only one
            # live run per label, so inserting before superseding would trip
            # the unique index. Same transaction, so this is atomic.
            cur.execute("SELECT supersede_label(%s, NULL, now())", (label,))

            cur.execute(
                """INSERT INTO graph_run
                       (label, ingested_at, n_chunks, embed_dim, params, diagnostics)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING run_id""",
                (label, cg.ingested_at, cg.n, embed_dim,
                 Jsonb(_params(cg)), Jsonb(cg.diagnostics)),
            )
            run_id = cur.fetchone()[0]

            # ---- nodes
            with cur.copy(
                "COPY node (run_id, ord, doc_id, chunk_hash, body, attrs) "
                "FROM STDIN"
            ) as cp:
                sources = getattr(cg, "source", None)
                for i, body in enumerate(cg.chunks):
                    attrs = {
                        "n_tok": len(cg.docs_tok[i]),
                        "qterms": list(cg.qterms[i]),
                        "disc": sorted(cg.disc[i]),
                        "tf": {t: int(f) for t, f in cg.tfs[i].items()},
                    }
                    src = sources[i] if sources is not None else None
                    if isinstance(src, str) and src:
                        attrs["source"] = src                                  # R20
                    cp.write_row((run_id, i, cg.doc_id[i], _hash(body),
                                  body, Jsonb(attrs)))

            # ---- edges (already canonical src < dst from np.triu)
            with cur.copy(
                "COPY edge (run_id, src, dst, provenance, strength, sim_sparse, "
                "sim_dense, src_doc, dst_doc, valid_from, valid_to, ingested_at, "
                "attrs) FROM STDIN"
            ) as cp:
                for e in edges:
                    edge_attrs = {}                                            # R20
                    src_source = e.get("src_source")
                    dst_source = e.get("dst_source")
                    if isinstance(src_source, str) and src_source:
                        edge_attrs["src_source"] = src_source
                    if isinstance(dst_source, str) and dst_source:
                        edge_attrs["dst_source"] = dst_source
                    cp.write_row((
                        run_id, e["src"], e["dst"], e["provenance"],
                        e["strength"], e["sim_sparse"], e["sim_dense"],
                        e["src_doc"], e["dst_doc"],
                        e["valid_from"], e["valid_to"], e["ingested_at"],
                        Jsonb(edge_attrs),
                    ))

            # ---- communities
            if comms:
                with cur.copy(
                    "COPY community (run_id, cid, members, keywords, medoid, "
                    "medoid_text) FROM STDIN"
                ) as cp:
                    for c in comms:
                        cp.write_row((
                            run_id, int(c["id"]),
                            [int(v) for v in c["members"]],
                            list(c["keywords"]),
                            int(c["medoid"]), c["medoid_text"],
                        ))

            # ---- embeddings
            if E is not None:
                _fix_embedding_dim(cur, embed_dim)
                with cur.copy(
                    "COPY node_embedding (run_id, ord, embedding) FROM STDIN"
                ) as cp:
                    for i in range(E.shape[0]):
                        cp.write_row((run_id, i, _vec_literal(E[i])))

            cur.execute("SELECT supersede_label(%s, %s, now())", (label, run_id))
        conn.commit()

    if embed_dim is not None:
        _ensure_hnsw(dsn)
    return str(run_id)


def _fix_embedding_dim(cur, dim: int) -> None:
    """The column ships as unconstrained `vector`; stamp the real dimension
    once so it can carry an HNSW index."""
    cur.execute("""SELECT format_type(a.atttypid, a.atttypmod)
                     FROM pg_attribute a
                    WHERE a.attrelid = 'node_embedding'::regclass
                      AND a.attname  = 'embedding'""")
    if cur.fetchone()[0] == "vector":          # no dimension yet
        cur.execute(
            f"ALTER TABLE node_embedding ALTER COLUMN embedding TYPE vector({dim})")


def _ensure_hnsw(dsn: str) -> None:
    """cg.E is L2-normalized, so inner product ranks identically to cosine
    and vector_ip_ops is cheaper. Built outside the ingest transaction."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("""CREATE INDEX IF NOT EXISTS node_embedding_hnsw
                          ON node_embedding USING hnsw (embedding vector_ip_ops)""")


def load_edges(run_label: str, dsn: str = DSN) -> list[dict]:
    """Round-trips back into the shape ChunkGraph.edges() produced."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT e.src, e.dst, e.src_doc, e.dst_doc, e.provenance::text,
                      e.strength, e.sim_sparse, e.sim_dense,
                      e.valid_from, e.valid_to, e.ingested_at,
                      e.attrs ->> 'src_source' AS src_source,
                      e.attrs ->> 'dst_source' AS dst_source
                 FROM edge e JOIN live_run r USING (run_id)
                WHERE r.label = %s AND e.valid_to IS NULL
                ORDER BY e.src, e.dst""",
            (run_label,),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
